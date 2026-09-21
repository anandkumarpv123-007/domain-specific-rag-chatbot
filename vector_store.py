from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple, Optional

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from document_loader import DocumentPage

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SAVE_DIR = os.path.join("vector_store", "saved_index")


@dataclass
class TextChunk:
    text: str
    source: str
    page_number: int
    chunk_id: int


@dataclass
class VectorStore:
    model: SentenceTransformer
    index: faiss.Index
    chunks: List[TextChunk]


def chunk_documents(
    pages: Sequence[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[TextChunk]:
    """
    Split document pages into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Preserves document source and page number metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[TextChunk] = []
    global_chunk_id = 1

    for page in pages:
        split_texts = splitter.split_text(page.text)
        for text in split_texts:
            clean_chunk = text.strip()
            if clean_chunk:
                chunks.append(
                    TextChunk(
                        text=clean_chunk,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=global_chunk_id,
                    )
                )
                global_chunk_id += 1

    if not chunks:
        raise ValueError("No text chunks were generated from the uploaded documents.")

    return chunks


def build_vector_store(
    pages: Sequence[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VectorStore:
    """
    Chunk documents, embed text with Sentence Transformers (all-MiniLM-L6-v2),
    and build a FAISS vector index.
    """
    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    model = SentenceTransformer(embedding_model_name)

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return VectorStore(model=model, index=index, chunks=chunks)


def save_vector_store(vector_store: VectorStore, directory_path: str = DEFAULT_SAVE_DIR) -> str:
    """Save FAISS index and metadata to disk."""
    os.makedirs(directory_path, exist_ok=True)

    index_path = os.path.join(directory_path, "faiss_index.bin")
    metadata_path = os.path.join(directory_path, "metadata.pkl")

    faiss.write_index(vector_store.index, index_path)

    serializable_chunks = [asdict(c) for c in vector_store.chunks]
    with open(metadata_path, "wb") as f:
        pickle.dump(serializable_chunks, f)

    return directory_path


def load_vector_store(
    directory_path: str = DEFAULT_SAVE_DIR,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[VectorStore]:
    """Load FAISS index and metadata from disk if available."""
    index_path = os.path.join(directory_path, "faiss_index.bin")
    metadata_path = os.path.join(directory_path, "metadata.pkl")

    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        return None

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        raw_chunks = pickle.load(f)

    chunks = [TextChunk(**c) for c in raw_chunks]
    model = SentenceTransformer(embedding_model_name)

    return VectorStore(model=model, index=index, chunks=chunks)


def retrieve_relevant_chunks(
    query: str,
    vector_store: VectorStore,
    top_k: int = 4,
) -> List[dict]:
    """
    Convert user query to embedding and perform similarity search in FAISS vector store.
    Returns top_k most relevant chunks with metadata and similarity score.
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    query_vector = vector_store.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(max(top_k, 1), len(vector_store.chunks))
    scores, indices = vector_store.index.search(query_vector, k)

    results: List[dict] = []
    for score, pos in zip(scores[0], indices[0]):
        if pos < 0 or pos >= len(vector_store.chunks):
            continue
        chunk = vector_store.chunks[int(pos)]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
            }
        )

    return results
