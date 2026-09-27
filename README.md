# Corrective RAG (CRAG)

A Corrective Retrieval-Augmented Generation (CRAG) system designed to improve the reliability of RAG applications by evaluating retrieved information and taking corrective action when the retrieved context is not relevant enough.

## 🚀 Overview

Traditional RAG systems retrieve documents from a knowledge base and directly pass them to an LLM for answer generation.

The problem is that retrieval is not always accurate. If irrelevant or insufficient documents are retrieved, the LLM may generate an incorrect or poorly grounded response.

This project introduces a corrective retrieval workflow that evaluates the quality of retrieved context before generating the final answer.

## 🔄 Workflow

```text
User Query
     ↓
Document Retriever
     ↓
Retrieved Documents
     ↓
Relevance Evaluation
     ↓
 ┌───────────────┐
 │ Relevant?     │
 └───────┬───────┘
      Yes│        No
         │         │
         ↓         ↓
   Generate     Corrective
     Answer     Retrieval
                   │
                   ↓
             Improved Context
                   │
                   ↓
             Generate Answer

✨ Key Features

Retrieval-Augmented Generation pipeline
Semantic document retrieval
Retrieved-context relevance evaluation
Corrective retrieval mechanism
LLM-based answer generation
Conditional workflow using LangGraph
Improved handling of irrelevant retrieved context
Modular RAG architecture

🛠️ Tech Stack

Python
LangChain
LangGraph
LLM
Embeddings
Vector Database
RAG
Web Search / External Retrieval
Streamlit (if used for the interface)

🧠 How It Works
1. Query

The user submits a question to the system.

2. Retrieval

The system searches the vector database and retrieves documents relevant to the query.

3. Retrieval Evaluation

The retrieved documents are evaluated to determine whether they contain useful information for answering the query.

4. Corrective Step

If the retrieved context is considered insufficient or irrelevant, the system performs an additional retrieval step instead of blindly using the original context.

5. Answer Generation

The final context is passed to the LLM, which generates the answer based on the available information.

📊 RAG vs Corrective RAG

Traditional RAG	Corrective RAG
Retrieves documents	Retrieves documents
Directly uses retrieved context	Evaluates retrieved context
No explicit correction step	Uses corrective retrieval
Can depend heavily on retrieval quality	Handles poor retrieval more carefully
Simple pipeline	Conditional workflow

