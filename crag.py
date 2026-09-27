from typing import List, TypedDict
from pydantic import BaseModel
import re
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
def create_retriever(pdf_files):

    start = time.time()

    # -------------------------
    # Load PDF
    # -------------------------

    docs = []

    for pdf_file in pdf_files:
        docs.extend(
            PyPDFLoader(pdf_file).load()
        )

    print(
        f"PDF loading: {time.time() - start:.2f} seconds"
    )

    # -------------------------
    # Chunking
    # -------------------------

    chunk_start = time.time()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    ).split_documents(docs)

    print(
        f"Chunking: {time.time() - chunk_start:.2f} seconds"
    )

    print(
        f"Pages: {len(docs)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    # -------------------------
    # Embeddings
    # -------------------------

    embedding_start = time.time()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print(
        f"Embedding + FAISS: "
        f"{time.time() - embedding_start:.2f} seconds"
    )

    # -------------------------
    # Retriever
    # -------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print(
        f"TOTAL: {time.time() - start:.2f} seconds"
    )

    return retriever
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
UPPER_TH = 0.7
LOWER_TH = 0.3
# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    
    web_docs: List[Document]

    answer: str
    retriever: object
def retrieve_node(state: State) -> State:
    q = state["question"]
    retriever = state["retriever"]

    return {
        "docs": retriever.invoke(q)
    }
# -----------------------------
# Score-based doc evaluator
# -----------------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)

        # Keep any doc above LOWER_TH as "weakly relevant"
        if out.score > LOWER_TH:
            good.append(d)

        # -----------------------------
    # CORRECT
    # -----------------------------
    if any(s > UPPER_TH for s in scores):

        max_score = max(scores)

        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": (
                f"A retrieved chunk was highly relevant to the question "
                f"(highest relevance score: {max_score:.2f}). "
                f"{len(good)} relevant chunk(s) were selected for refinement."
            ),
        }

    # -----------------------------
    # INCORRECT
    # -----------------------------
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):

        max_score = max(scores)

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": (
                f"None of the retrieved chunks were sufficiently relevant "
                f"to the question (highest relevance score: {max_score:.2f}). "
                f"Web search was triggered to find additional information."
            ),
        }

    # -----------------------------
    # AMBIGUOUS
    # -----------------------------
    max_score = max(scores) if scores else 0.0

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": (
            f"The retrieved context was partially relevant but did not "
            f"contain a highly relevant chunk "
            f"(highest relevance score: {max_score:.2f}). "
            f"Web search was used to supplement the retrieved documents."
        ),
    }

# -----------------------------
# Sentence-level DECOMPOSER
# -----------------------------
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# -----------------------------
# FILTER (LLM judge)
# -----------------------------
class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


# -----------------------------
# Knowledge refinement
# (CORRECT => internal only)
# (INCORRECT => web only)
# (AMBIGUOUS => internal + web)
# -----------------------------
def refine(state: State) -> State:

    # --------------------------------
    # Select documents
    # --------------------------------

    if state["verdict"] == "CORRECT":

        docs = state["good_docs"]

    elif state["verdict"] == "INCORRECT":

        docs = state["web_docs"]

    else:

        docs = state["good_docs"] + state["web_docs"]


    # --------------------------------
    # Build context
    # --------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # --------------------------------
    # Handle empty context
    # --------------------------------

    if not context.strip():

        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": ""
        }


    # --------------------------------
    # One-shot refinement
    # --------------------------------

    refine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a context refinement system.

Given the user's question and the retrieved context:

1. Keep only information relevant to answering the question.
2. Remove irrelevant information.
3. Do not add new information.
4. Do not change facts.
5. Return only the relevant context.
"""
            ),
            (
                "human",
                """
Question:
{question}

Retrieved Context:
{context}
"""
            )
        ]
    )


    refine_chain = refine_prompt | llm


    out = refine_chain.invoke(
        {
            "question": state["question"],
            "context": context
        }
    )


    refined_context = out.content


    # --------------------------------
    # Return
    # --------------------------------

    return {
        "strips": [],
        "kept_strips": [],
        "refined_context": refined_context
    }
# -----------------------------
# Query rewrite for web search
# -----------------------------
class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


# -----------------------------
# Web search node: uses web_query
# -----------------------------
tavily = TavilySearchResults(max_results=5)


def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})

    web_docs: List[Document] = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}

# -----------------------------
# Generate
# -----------------------------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def generate(state: State) -> State:

    print(state["question"])
    print(state["verdict"])
    print(len(state["good_docs"]))
    print(state["refined_context"])

    out = (
        answer_prompt | llm
    ).invoke(
        {
            "question": state["question"],
            "context": state["refined_context"]
        }
    )

    return {
        "answer": out.content
    }

# -----------------------------
# Routing
# CORRECT => refine
# INCORRECT / AMBIGUOUS => rewrite -> web_search -> refine -> generate
# -----------------------------
def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

# -----------------------------
# Build graph
# -----------------------------

def timed_node(name, func):

    def wrapper(state):

        start = time.time()

        result = func(state)

        elapsed = time.time() - start

        print(f"{name}: {elapsed:.2f} sec")

        return result

    return wrapper

g = StateGraph(State)

g.add_node(
    "retrieve",
    timed_node("RETRIEVE", retrieve_node)
)

g.add_node(
    "eval_each_doc",
    timed_node("EVALUATE DOCS", eval_each_doc_node)
)

g.add_node(
    "rewrite_query",
    timed_node("REWRITE QUERY", rewrite_query_node)
)

g.add_node(
    "web_search",
    timed_node("WEB SEARCH", web_search_node)
)

g.add_node(
    "refine",
    timed_node("REFINE", refine)
)

g.add_node(
    "generate",
    timed_node("GENERATE", generate)
)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)

# non-correct path
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")

# correct path already goes to refine
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()


def ask_question(question, retriever):

    start = time.time()

    result = app.invoke(
        {
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_query": "",
            "web_docs": [],
            "answer": "",
            "retriever": retriever,
        }
    )

    print(
        "Total CRAG time:",
        round(time.time() - start, 2),
        "seconds"
    )

    return result