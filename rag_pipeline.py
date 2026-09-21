from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Optional
from groq import Groq

from document_loader import load_documents, DocumentPage
from vector_store import (
    VectorStore,
    build_vector_store as create_vector_store,
    save_vector_store as save_index,
    load_vector_store as load_index,
    retrieve_relevant_chunks,
    DEFAULT_EMBEDDING_MODEL,
)
from prompt import SYSTEM_GUARDRAIL_PROMPT, build_rag_prompt, EXACT_FALLBACK_RESPONSE

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def process_and_index_documents(
    uploaded_files: Iterable,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    persist: bool = True,
) -> Tuple[VectorStore, int, List[str]]:
    """
    Complete document processing pipeline:
    1. Extract text page by page with pypdf and capture metadata (source, page).
    2. Chunk text using LangChain RecursiveCharacterTextSplitter.
    3. Generate Sentence Transformer embeddings.
    4. Store in FAISS vector database.
    5. Optionally persist FAISS index to disk (vector_store/saved_index/).
    """
    pages = load_documents(uploaded_files)
    vector_store = create_vector_store(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if persist:
        save_index(vector_store)

    doc_names = list({p.source for p in pages})
    return vector_store, len(vector_store.chunks), doc_names


def load_persisted_knowledge_base() -> Optional[VectorStore]:
    """Attempt to load saved FAISS vector store from local disk."""
    return load_index()


def generate_grounded_answer(
    question: str,
    vector_store: VectorStore,
    top_k: int = 4,
    api_key: str = "",
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Tuple[str, List[dict]]:
    """
    RAG Query Pipeline:
    1. Retrieve top-k relevant chunks from FAISS vector store.
    2. Format retrieved context with source document and page number.
    3. Query Groq LLM with strict system guardrails and refusal fallback.
    4. Return answer and retrieved source chunks.
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("Groq API key missing. Please provide GROQ_API_KEY in .env or the sidebar.")

    sources = retrieve_relevant_chunks(question, vector_store, top_k=top_k)

    if not sources:
        return EXACT_FALLBACK_RESPONSE, []

    # Format context with document name and page number citations
    context_blocks = []
    for item in sources:
        header = f"[Document: {item['source']} | Page: {item['page_number']} | Chunk: {item['chunk_id']}]"
        context_blocks.append(f"{header}\n{item['text']}")

    formatted_context = "\n\n---\n\n".join(context_blocks)
    user_prompt = build_rag_prompt(formatted_context, question)

    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_GUARDRAIL_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_completion_tokens=700,
        )

        answer = completion.choices[0].message.content
        if not answer or not answer.strip():
            return EXACT_FALLBACK_RESPONSE, sources

        clean_answer = answer.strip()
        return clean_answer, sources

    except Exception as e:
        raise RuntimeError(f"Error querying Groq LLM API: {e}")
