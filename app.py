import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    process_and_index_documents,
    load_persisted_knowledge_base,
    generate_grounded_answer,
    DEFAULT_GROQ_MODEL,
)
from vector_store import DEFAULT_EMBEDDING_MODEL
from prompt import EXACT_FALLBACK_RESPONSE

load_dotenv()

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


def init_session_state() -> None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_persisted_knowledge_base()
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_documents_and_vector_store() -> None:
    st.session_state.vector_store = None
    st.session_state.processed_files = []
    st.session_state.messages = []


def clear_chat_history() -> None:
    st.session_state.messages = []


def main() -> None:
    init_session_state()

    st.title("📚 Domain-Specific RAG Chatbot")
    st.caption("Grounded PDF Question Answering with Page-Level Source Attribution")

    # High-stakes warning disclaimer
    st.warning(
        "⚠️ **Responsible AI Disclaimer**: AI-generated responses are based on retrieved document passages. "
        "Please verify all critical or high-stakes information against original official documents."
    )

    with st.expander("ℹ️ How this RAG System Works (PDF Guidance Architecture)", expanded=False):
        st.markdown(
            """
            1. **Upload PDF Files**: Select one or multiple PDF documents.  
            2. **Text Extraction**: Read page-by-page using `pypdf`, capturing document name and page number metadata.  
            3. **Text Chunking**: Split text with LangChain's `RecursiveCharacterTextSplitter` (700–1000 chars, 100–150 overlap).  
            4. **Embeddings**: Generate dense vectors using Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`).  
            5. **FAISS Vector Store**: Store embeddings in FAISS index with local disk persistence (`vector_store/saved_index/`).  
            6. **Similarity Retrieval**: Query FAISS to retrieve top 3–5 most relevant chunks.  
            7. **Grounded LLM Answer**: Groq LLM generates answers strictly from retrieved context with prompt injection defenses.  
            8. **Exact Refusal & Sources**: Displays exact fallback if missing, along with document name and page citations.
            """
        )

    # --- SIDEBAR UI ---
    with st.sidebar:
        st.header("📄 Document Upload & Setup")

        # PDF Uploader in Sidebar
        uploaded_files = st.file_uploader(
            "Upload PDF Document(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF files (Max 25 MB per file).",
        )

        if uploaded_files:
            st.markdown("**Uploaded Files:**")
            for f in uploaded_files:
                st.caption(f"• `{f.name}` ({f.size / (1024*1024):.2f} MB)")

        # Process Documents Button directly underneath PDF upload area (immediately visible)
        process_button = st.button(
            "⚡ Process Documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

        st.divider()

        chunk_size = st.slider("Chunk Size (characters)", 400, 1500, 800, 50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 300, 120, 10)
        top_k = st.slider("Chunks to Retrieve (top-k)", 1, 8, 4)

        if process_button and uploaded_files:
            if chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                try:
                    with st.spinner("Extracting text, chunking, and indexing in FAISS..."):
                        vector_store, num_chunks, doc_names = process_and_index_documents(
                            uploaded_files,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            persist=True,
                        )

                    st.session_state.vector_store = vector_store
                    st.session_state.processed_files = doc_names
                    st.session_state.messages = []
                    st.success(f"Successfully processed {len(doc_names)} file(s) into {num_chunks} chunks!")
                except Exception as err:
                    st.error(f"Processing Error: {err}")

        st.divider()

        st.header("⚙️ Model & API Settings")

        # Secure API Key Handling (NEVER displays raw API key string in UI)
        env_api_key = os.getenv("GROQ_API_KEY", "")
        if env_api_key:
            st.success("🔒 Groq API: Configured (.env)")
            override_key = st.text_input(
                "Override Groq API Key (Optional)",
                type="password",
                value="",
                help="Leave blank to use GROQ_API_KEY from .env",
            )
            api_key = override_key.strip() if override_key.strip() else env_api_key
        else:
            st.warning("⚠️ Groq API Key Not Found in .env")
            api_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                value="",
                help="Enter your Groq API key (e.g. gsk_...)",
            ).strip()

        model_name = st.selectbox(
            "Groq Model",
            ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound"],
            index=0,
        )

        st.caption(f"Embedding Model: `{DEFAULT_EMBEDDING_MODEL}`")

        st.divider()

        if st.button("🗑️ Reset Documents & Knowledge Base", use_container_width=True):
            reset_documents_and_vector_store()
            st.rerun()

    # --- MAIN AREA ---
    if st.session_state.vector_store is None:
        st.info("👈 Please upload PDF document(s) in the sidebar and click **Process Documents** to start.")
        return

    # Knowledge Base Metrics Banner
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Indexed Documents", len(st.session_state.processed_files) or "Loaded Index")
    m_col2.metric("Total Text Chunks", len(st.session_state.vector_store.chunks))
    m_col3.metric("Vector Database", "FAISS (Local Disk)")

    # Action Toolbar with Dedicated Clear Chat Button
    c_col1, c_col2 = st.columns([4, 1])
    with c_col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Retrieved Source Passages & Page References"):
                    for src in msg["sources"]:
                        st.markdown(f"**📄 Document:** `{src['source']}` | **Page:** `{src['page_number']}` | **Chunk ID:** `{src['chunk_id']}`")
                        st.caption(f"Similarity Score: {src['score']:.4f}")
                        st.text(src["text"])

    # Chat Input Box
    question = st.chat_input("Ask a question based on the uploaded documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS index and generating grounded response..."):
                try:
                    answer, sources = generate_grounded_answer(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        top_k=top_k,
                        api_key=api_key,
                        model_name=model_name,
                    )
                    st.markdown(answer)

                    if sources and answer != EXACT_FALLBACK_RESPONSE:
                        with st.expander("📌 Retrieved Source Passages & Page References"):
                            for src in sources:
                                st.markdown(f"**📄 Document:** `{src['source']}` | **Page:** `{src['page_number']}` | **Chunk ID:** `{src['chunk_id']}`")
                                st.caption(f"Similarity Score: {src['score']:.4f}")
                                st.text(src["text"])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as err:
                    st.error(f"Error generating answer: {err}")


if __name__ == "__main__":
    main()