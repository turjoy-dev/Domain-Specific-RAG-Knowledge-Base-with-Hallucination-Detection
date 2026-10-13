"""
MCP Server: RAGTruth Knowledge Base
===================================
Exposes the local ChromaDB vector store as a query tool via Model Context Protocol (MCP).
"""

import logging
from pathlib import Path
from typing import Optional

# Support both MCP 2.x (MCPServer) and MCP 1.x (FastMCP)
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP

from langchain_community.vectorstores import Chroma

# Set up logging for standard error (so it doesn't corrupt stdout for MCP)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("ragtruth-knowledge-base")

# Resolve workspace paths dynamically
WORKSPACE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = WORKSPACE_DIR / "Database" / "chroma_db"

# Global variables for lazy initialization
_embedding_model = None
_vector_store = None


def get_vector_store() -> Chroma:
    """Lazy-loads the embedding model and ChromaDB connection."""
    global _embedding_model, _vector_store

    if _vector_store is None:
        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIR}")

        logger.info("Loading embedding model BAAI/bge-small-en-v1.5...")
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            _embedding_model = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as err:
            logger.warning(
                f"Standard HuggingFaceBgeEmbeddings unavailable ({err}). "
                "Falling back to native Transformers BgeEmbeddings."
            )
            from build_vector_store import BgeEmbeddings
            _embedding_model = BgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", device="cpu")

        logger.info(f"Connecting to ChromaDB at {CHROMA_DB_DIR}...")
        _vector_store = Chroma(
            collection_name="ragtruth_knowledge_base",
            embedding_function=_embedding_model,
            persist_directory=str(CHROMA_DB_DIR),
        )
    return _vector_store


@mcp.tool()
def query_vector_db(query: str, task_type_filter: Optional[str] = None) -> str:
    """
    Queries the local RAGTruth vector database for the most relevant context chunks.

    Args:
        query: The search query string.
        task_type_filter: Optional filter for specific task types (e.g., 'Summary', 'QA', 'Data2txt').

    Returns:
        A formatted string containing the top 3 most relevant context chunks and their metadata.
    """
    try:
        vs = get_vector_store()

        # Build metadata filter if specified
        search_filter = None
        if task_type_filter and task_type_filter.strip():
            filter_val = task_type_filter.strip()
            valid_tasks = {"summary": "Summary", "qa": "QA", "data2txt": "Data2txt"}
            canonical_task = valid_tasks.get(filter_val.lower(), filter_val)
            search_filter = {"task_type": canonical_task}

        # Retrieve top 3 most relevant chunks
        results = vs.similarity_search(query.strip(), k=3, filter=search_filter)

        if not results:
            return "No relevant chunks found."

        # Format the output into structured markdown for the calling LLM
        formatted_results = []
        for i, doc in enumerate(results):
            meta = doc.metadata
            doc_id = meta.get("doc_id", "Unknown")
            domain = meta.get("source_domain", "Unknown")
            task = meta.get("task_type", "Unknown")
            chunk_idx = meta.get("chunk_index", 0)

            header = f"--- Result {i+1} [Doc ID: {doc_id} | Task: {task} | Domain: {domain} | Chunk: {chunk_idx}] ---"
            formatted_results.append(f"{header}\n{doc.page_content}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        return f"Error executing query: {str(e)}"


if __name__ == "__main__":
    # Start the MCP server using standard input/output (stdio) transport
    # Note: FastMCP automatically handles the JSON-RPC stdio wrapping natively
    mcp.run(transport="stdio")
