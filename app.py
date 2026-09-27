import streamlit as st
import tempfile
import os

from crag import create_retriever, ask_question


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="CRAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Session State
# -----------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False


# -----------------------------
# Title
# -----------------------------

st.title("🤖 CRAG Chatbot")

st.caption(
    "Upload your PDFs once and ask unlimited questions."
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} PDF(s) selected**"
        )

        if st.button(
            "🚀 Process Documents",
            use_container_width=True
        ):

            temp_paths = []

            try:

                with st.status(
                    "Processing documents...",
                    expanded=True
                ):

                    st.write("📖 Loading PDFs...")

                    for uploaded_file in uploaded_files:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as tmp:

                            tmp.write(
                                uploaded_file.getbuffer()
                            )

                            temp_paths.append(tmp.name)

                    st.write("✂️ Creating document chunks...")

                    st.write("🧠 Creating embeddings...")

                    retriever = create_retriever(
                        temp_paths
                    )

                    st.session_state.retriever = retriever

                    st.session_state.documents_processed = True

                    # Clear previous conversation
                    st.session_state.messages = []

                    st.write("✅ FAISS vector store ready!")

            finally:

                for path in temp_paths:

                    if os.path.exists(path):
                        os.remove(path)


    # -------------------------
    # Status
    # -------------------------

    st.divider()

    if st.session_state.documents_processed:

        st.success("🟢 Documents ready")

        if st.button(
            "🗑️ Clear Documents",
            use_container_width=True
        ):

            st.session_state.retriever = None
            st.session_state.documents_processed = False
            st.session_state.messages = []

            st.rerun()

    else:

        st.info("Upload PDFs to start.")


# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # -------------------------
    # Check documents
    # -------------------------

    if st.session_state.retriever is None:

        st.warning(
            "Please upload and process your PDFs first."
        )

        st.stop()


    # -------------------------
    # User message
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # CRAG
    # -------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Thinking..."
        ):

            result = ask_question(
                question,
                st.session_state.retriever
            )

        answer = result["answer"]

        st.markdown(answer)

        # -----------------------------
        # CRAG Decision
        # -----------------------------

    verdict = result.get("verdict", "")
    reason = result.get("reason", "")
    web_query = result.get("web_query", "")


    if verdict == "CORRECT":

        source_used = "📄 Documents"

    elif verdict == "INCORRECT":

        source_used = "🌐 Web Search"

    elif verdict == "AMBIGUOUS":

        source_used = "📄 Documents + 🌐 Web Search"

    else:

        source_used = "Unknown"


    with st.expander("🧠 CRAG Decision"):

        st.markdown(
            f"**Verdict:** `{verdict}`"
        )

        st.markdown(
            f"**Reason:** {reason}"
        )

        st.markdown(
            f"**Sources Used:** {source_used}"
        )

        if web_query:

            st.markdown(
                f"**Web Query:** `{web_query}`"
            )


    # -------------------------
    # Save answer
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )