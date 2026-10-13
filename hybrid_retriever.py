"""
Hybrid Retrieval Engine (BM25 Sparse + ChromaDB Dense)
======================================================
Combines dense semantic embeddings (ChromaDB + BGE) with sparse lexical keyword matching
(BM25) via LangChain's EnsembleRetriever (Reciprocal Rank Fusion / Weighted Scoring).
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain_community.retrievers.bm25 import BM25Retriever

from langchain_community.vectorstores import Chroma

# Reuse the optimized embedding wrapper and chunking pipelines
from build_vector_store import (
    BgeEmbeddings,
    build_conditional_splitters,
    load_and_chunk_corpus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid_retriever")

WORKSPACE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = WORKSPACE_DIR / "Database" / "chroma_db"
CORPUS_FILE = WORKSPACE_DIR / "Dataset" / "Processed_DATA_NLP" / "ragtruth_cleaned_corpus.jsonl"
BM25_CACHE_FILE = WORKSPACE_DIR / "Database" / "bm25_index.pkl"


# =====================================================================
# 1. Dense Retriever Setup (ChromaDB + BGE)
# =====================================================================

def get_dense_retriever(k: int = 3) -> Any:
    """
    Connects to the persistent local ChromaDB instance and creates a Dense retriever.
    """
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DB_DIR}. "
            "Please run build_vector_store.py first."
        )

    logger.info("Initializing Dense ChromaDB retriever with local BGE embeddings...")
    embedding_model = BgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    chroma_store = Chroma(
        collection_name="ragtruth_knowledge_base",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return chroma_store.as_retriever(search_kwargs={"k": k})


# =====================================================================
# 2. Sparse Retriever Setup (BM25 on Cleaned Corpus)
# =====================================================================

def get_sparse_bm25_retriever(k: int = 3, use_cache: bool = True) -> BM25Retriever:
    """
    Constructs a BM25 sparse retriever across the exact same chunked documents
    from ragtruth_cleaned_corpus.jsonl. Supports caching the index to avoid re-chunking.
    """
    if use_cache and BM25_CACHE_FILE.exists():
        logger.info(f"Loading cached BM25 index from {BM25_CACHE_FILE.name}...")
        try:
            with open(BM25_CACHE_FILE, "rb") as f:
                bm25_retriever = pickle.load(f)
                bm25_retriever.k = k
                return bm25_retriever
        except Exception as e:
            logger.warning(f"Failed to load cached BM25 index ({e}). Rebuilding from corpus...")

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Cleaned corpus not found at {CORPUS_FILE}")

    logger.info("Chunking corpus for BM25 sparse keyword index...")
    splitters = build_conditional_splitters()
    chunks: List[Document] = load_and_chunk_corpus(CORPUS_FILE, splitters=splitters)
    logger.info(f"Loaded {len(chunks)} chunks. Building BM25 index...")

    bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=k)

    # Persist index cache to disk for fast subsequent loads
    try:
        BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_CACHE_FILE, "wb") as f:
            pickle.dump(bm25_retriever, f)
        logger.info(f"Saved BM25 index cache to {BM25_CACHE_FILE.resolve()}")
    except Exception as e:
        logger.warning(f"Could not persist BM25 cache: {e}")

    return bm25_retriever


# =====================================================================
# 3. Hybrid Ensemble Retriever Assembly
# =====================================================================

def build_hybrid_retriever(
    k: int = 3,
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7
) -> EnsembleRetriever:
    """
    Creates an EnsembleRetriever combining:
      - Sparse lexical retrieval (BM25) with weight bm25_weight
      - Dense semantic retrieval (ChromaDB) with weight dense_weight
    """
    sparse_retriever = get_sparse_bm25_retriever(k=k)
    dense_retriever = get_dense_retriever(k=k)

    logger.info(
        f"Assembling Hybrid EnsembleRetriever (weights: BM25={bm25_weight}, Dense={dense_weight}, k={k})..."
    )
    ensemble = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight]
    )
    return ensemble


# =====================================================================
# 4. Standalone Test Entrypoint
# =====================================================================

def test_hybrid_search(query: str = "Apna Indian Kitchen hours", k: int = 3):
    """Executes a hybrid search query and prints the retrieved chunks with provenance."""
    print("=" * 80)
    print(f"HYBRID RETRIEVAL TEST: \"{query}\" (k={k})")
    print("=" * 80)

    hybrid = build_hybrid_retriever(k=k, bm25_weight=0.3, dense_weight=0.7)

    # Invoke hybrid retriever
    results: List[Document] = hybrid.invoke(query)

    print(f"\nRetrieved {len(results)} chunks via Hybrid Fusion:\n")
    for idx, doc in enumerate(results, start=1):
        meta = doc.metadata
        doc_id = meta.get("doc_id", "Unknown")
        chunk_idx = meta.get("chunk_index", 0)
        task = meta.get("task_type", "Unknown")
        domain = meta.get("source_domain", "Unknown")
        snippet = doc.page_content.strip()

        print(f"[{idx}] Doc: {doc_id} | Chunk: {chunk_idx} | Task: {task} | Domain: {domain}")
        print(f"    Content: {snippet[:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "Apna Indian Kitchen hours"
    test_hybrid_search(test_q)
