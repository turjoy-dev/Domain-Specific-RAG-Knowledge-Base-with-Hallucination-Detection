"""
Vector Database Ingestion Pipeline
==================================
Builds a local persistent ChromaDB vector store from the cleaned RAGTruth corpus.
Applies task-specific conditional chunking (Summary, QA, Data2txt) and BGE embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# LangChain text splitting & vector store imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_vector_store")


class BgeEmbeddings(Embeddings):
    """
    Direct Hugging Face Transformers wrapper for BAAI/bge-small-en-v1.5.
    Computes normalized CLS pooled embeddings via PyTorch.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs[0][:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                results.extend(embeddings.cpu().tolist())
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# =====================================================================
# 1. Configuration & Conditional Splitters
# =====================================================================

def build_conditional_splitters() -> Dict[str, RecursiveCharacterTextSplitter]:
    """
    Constructs specialized RecursiveCharacterTextSplitter instances
    tailored to each task type with the required boundaries and parameters.
    """
    separators = ["\n\n", "\n", ". ", " "]

    return {
        "Summary": RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=separators,
        ),
        "QA": RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=30,
            separators=separators,
        ),
        "Data2txt": RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=0,
            separators=separators,
        ),
    }


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> Embeddings:
    """Initializes and returns the local BgeEmbeddings model."""
    return BgeEmbeddings(model_name=model_name, device="cpu")


# =====================================================================
# 2. Corpus Loading & Conditional Chunking
# =====================================================================

def load_and_chunk_corpus(
    input_file: Path,
    splitters: Dict[str, RecursiveCharacterTextSplitter],
    limit_docs: Optional[int] = None
) -> List[Document]:
    """
    Loads ragtruth_cleaned_corpus.jsonl, applies conditional chunking,
    and returns LangChain Document objects with preserved metadata.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned corpus not found at {input_file}")

    documents_to_ingest: List[Document] = []
    task_chunk_counts: Dict[str, int] = {}
    docs_read = 0

    logger.info(f"Reading cleaned corpus from: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            doc_id = record.get("doc_id", f"doc_{line_num}")
            task_type = record.get("task_type", "Summary")
            source_domain = record.get("source_domain", "Unknown")
            clean_text = record.get("clean_text", "")

            if not clean_text:
                continue

            # Select splitter matching the task_type (fallback to Summary)
            splitter = splitters.get(task_type, splitters["Summary"])
            text_chunks = splitter.split_text(clean_text)

            for idx, chunk in enumerate(text_chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                doc_obj = Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "task_type": task_type,
                        "source_domain": source_domain,
                        "chunk_index": idx,
                        "chunk_id": chunk_id,
                    }
                )
                documents_to_ingest.append(doc_obj)
                task_chunk_counts[task_type] = task_chunk_counts.get(task_type, 0) + 1

            docs_read += 1
            if limit_docs and docs_read >= limit_docs:
                break

    logger.info(f"Loaded {docs_read} source documents.")
    logger.info(f"Total chunks created: {len(documents_to_ingest)}")
    logger.info(f"Chunks generated per task type: {task_chunk_counts}")
    return documents_to_ingest


# =====================================================================
# 3. Vector Database Ingestion
# =====================================================================

def ingest_documents_to_chroma(
    documents: List[Document],
    db_directory: Path,
    embedding_model: HuggingFaceBgeEmbeddings,
    collection_name: str = "ragtruth_knowledge_base",
    batch_size: int = 500
) -> Chroma:
    """
    Ingests chunked documents into a persistent local ChromaDB instance
    using batched insertion.
    """
    try:
        db_directory.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.error(f"Failed to create vector database directory {db_directory}: {err}")
        raise

    logger.info(f"Target ChromaDB directory: {db_directory}")
    logger.info(f"Initializing collection '{collection_name}' with {len(documents)} chunks...")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(db_directory),
    )

    total_chunks = len(documents)
    # Batch insertion to optimize memory usage and adhere to SQLite parameter limits
    for start_idx in range(0, total_chunks, batch_size):
        end_idx = min(start_idx + batch_size, total_chunks)
        batch = documents[start_idx:end_idx]

        ids = [doc.metadata["chunk_id"] for doc in batch]
        vector_store.add_documents(documents=batch, ids=ids)

        logger.info(f"Ingested chunks [{start_idx + 1} - {end_idx}] / {total_chunks}")

    logger.info(f"Successfully persisted {total_chunks} chunks to {db_directory}.")
    return vector_store


# =====================================================================
# 4. Pipeline Execution
# =====================================================================

def main():
    # Dynamic workspace resolution
    workspace_dir = Path(__file__).resolve().parent
    input_file = workspace_dir / "Dataset" / "Processed_DATA_NLP" / "ragtruth_cleaned_corpus.jsonl"
    chroma_db_dir = workspace_dir / "Database" / "chroma_db"

    splitters = build_conditional_splitters()
    documents = load_and_chunk_corpus(input_file, splitters)

    embedding_model = get_embedding_model("BAAI/bge-small-en-v1.5")
    ingest_documents_to_chroma(
        documents=documents,
        db_directory=chroma_db_dir,
        embedding_model=embedding_model,
        collection_name="ragtruth_knowledge_base",
        batch_size=500
    )


if __name__ == "__main__":
    main()
