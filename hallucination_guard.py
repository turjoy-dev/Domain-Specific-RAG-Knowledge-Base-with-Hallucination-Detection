"""
Hallucination Detection & Self-Refinement Guard Loop
=====================================================
Implements a dual-stage RAG chain:
- Stage 1: Strictly grounded generation from local ChromaDB context.
- Stage 2: Independent LLM hallucination judge.
- Self-Refinement: Automated correction loop with safe fallback upon judge failure.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

# Local embedding wrapper established in Phase 2
from build_vector_store import BgeEmbeddings
from hybrid_retriever import build_hybrid_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hallucination_guard")

WORKSPACE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = WORKSPACE_DIR / "Database" / "chroma_db"


# =====================================================================
# 1. Base LLM Placeholder & Component Factory
# =====================================================================

def get_base_llm(
    provider: str = "google",
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:11434"
) -> Any:
    """
    Initializes a LangChain-compatible Chat Model with zero temperature for deterministic reasoning.
    Supports:
    - 'ollama': Local zero-cost inference via Ollama HTTP API (e.g., qwen2.5-coder:14b, gemma4:e4b)
    - 'google': Cloud inference via ChatGoogleGenerativeAI (Gemini)
    - 'openai': Cloud inference via ChatOpenAI
    - 'mock': Offline dry-run placeholder
    """
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        return OllamaChatWrapper(
            model=model_name if model_name != "gemini-1.5-flash" else "qwen2.5-coder:14b",
            base_url=base_url,
            temperature=temperature
        )

    elif provider_lower == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=key
            )
        except ImportError:
            logger.warning(
                "langchain_google_genai is not installed. "
                "Returning mock LLM placeholder for pipeline scaffolding."
            )
            return _MockChatModel()

    elif provider_lower == "openai":
        try:
            from langchain_openai import ChatOpenAI
            key = api_key or os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=key
            )
        except ImportError:
            logger.warning(
                "langchain_openai is not installed. "
                "Returning mock LLM placeholder for pipeline scaffolding."
            )
            return _MockChatModel()

    else:
        return _MockChatModel()


class OllamaChatWrapper:
    """Lightweight, zero-dependency wrapper for local Ollama chat inference."""

    def __init__(self, model: str = "qwen2.5-coder:14b", base_url: str = "http://localhost:11434", temperature: float = 0.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def invoke(self, prompt: Any) -> Any:
        import requests

        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "options": {"temperature": self.temperature}
        }

        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama API request failed ({self.model}): {e}")
            raise

        class _OllamaResponse:
            def __init__(self, c: str):
                self.content = c
            def __str__(self):
                return self.content

        return _OllamaResponse(content)


class _MockChatModel:
    """Mock LLM placeholder used for offline syntax validation and pipeline testing."""
    def invoke(self, prompt: Any) -> Any:
        class _MockResponse:
            content = "PASS"
        return _MockResponse()


def load_vector_store() -> Chroma:
    """Connects to the local persistent ChromaDB instance created in Phase 2."""
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DB_DIR}. "
            "Please run build_vector_store.py first."
        )

    embedding_model = BgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    return Chroma(
        collection_name="ragtruth_knowledge_base",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DB_DIR),
    )


# =====================================================================
# 2. Prompt Templates
# =====================================================================

STAGE_1_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are a strictly grounded, factual AI assistant.
Use ONLY the following context to answer the question.
If the context does not contain the answer, say 'Insufficient context.'
Do NOT infer, speculate, or introduce any outside facts.

Context:
{context}

Question:
{question}

Answer:"""
)

STAGE_2_JUDGE_PROMPT = PromptTemplate.from_template(
    """You are an impartial, strict factual accuracy evaluator and hallucination detector.
Verify if the answer is 100% factually supported by the provided context.
If it introduces outside facts, unsupported claims, or unmentioned entity details, it is a hallucination.

Context:
{context}

Question:
{question}

Answer to Evaluate:
{answer}

Evaluation Instructions:
- If every single claim in the answer is completely supported by the context, output exactly:
PASS
- If the answer honestly states 'Insufficient context' (or indicates that the provided context does not contain enough information to answer the question) because the context genuinely lacks the required facts, output exactly:
PASS
- If the answer contains ANY ungrounded claims, invented facts, or external knowledge not supported by the context, output:
FAIL: [Provide a concise, 1-2 sentence explanation of the unsupported facts]"""
)

REFINEMENT_PROMPT = PromptTemplate.from_template(
    """You previously generated an answer that contained hallucinated or unsupported statements flagged by our factual verification judge.

Context:
{context}

Original Question:
{question}

Previous Draft:
{draft_answer}

Judge Feedback:
{feedback}

Task:
Rewrite the answer so that it is 100% faithful to the context and addresses the judge's feedback.
Remove all hallucinated or external statements.
If the context is insufficient to answer any part of the question faithfully, state 'Insufficient context.'

Refined Answer:"""
)


# =====================================================================
# 3. Two-Stage Execution Engine & Self-Refinement Loop
# =====================================================================

class HallucinationGuard:
    """Orchestrates retrieval, grounded generation, judge verification, and self-refinement."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        vector_store: Optional[Chroma] = None,
        llm: Optional[Any] = None,
        k: int = 3
    ):
        self.k = k
        if retriever is not None:
            self.retriever = retriever
        elif vector_store is not None:
            self.retriever = vector_store.as_retriever(search_kwargs={"k": k})
        else:
            self.retriever = build_hybrid_retriever(k=k)
        self.llm = llm or get_base_llm()

    def retrieve_context(self, query: str, task_type_filter: Optional[str] = None) -> Tuple[str, List[Document]]:
        """Retrieves top-k chunks using hybrid retrieval and formats them with provenance metadata."""
        docs = self.retriever.invoke(query.strip())
        if task_type_filter:
            docs = [d for d in docs if d.metadata.get("task_type") == task_type_filter]
        if self.k and len(docs) > self.k:
            docs = docs[:self.k]

        formatted_chunks = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata
            header = f"[Chunk {i} | Doc: {meta.get('doc_id', 'Unknown')} | Task: {meta.get('task_type', 'Unknown')} | Domain: {meta.get('source_domain', 'Unknown')}]"
            formatted_chunks.append(f"{header}\n{d.page_content.strip()}")

        context_str = "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant context found."
        return context_str, docs

    def generate_initial_answer(self, query: str, context: str) -> str:
        """Stage 1: Generates candidate answer strictly constrained to context."""
        prompt = STAGE_1_GENERATION_PROMPT.format(question=query, context=context)
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def judge_answer(self, query: str, context: str, answer: str) -> Tuple[bool, str]:
        """
        Stage 2: Hallucination Judge evaluates candidate answer.
        Returns: (passed: bool, verdict_or_reason: str)
        """
        prompt = STAGE_2_JUDGE_PROMPT.format(question=query, context=context, answer=answer)
        response = self.llm.invoke(prompt)
        raw_verdict = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if raw_verdict.upper().startswith("PASS"):
            return True, "PASS"
        else:
            reason = raw_verdict
            if reason.upper().startswith("FAIL:"):
                reason = reason[5:].strip()
            return False, reason

    def refine_answer(
        self,
        query: str,
        context: str,
        draft_answer: str,
        feedback: str
    ) -> str:
        """Refinement Pass: Instructs LLM to surgically remove hallucinated facts."""
        prompt = REFINEMENT_PROMPT.format(
            question=query,
            context=context,
            draft_answer=draft_answer,
            feedback=feedback
        )
        response = self.llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def ask_with_guard(
        self,
        query: str,
        max_refinements: int = 1,
        task_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the guarded loop:
        1. Context Retrieval (top-3 chunks) with optional task_type filter
        2. Stage 1: Grounded Generation
        3. Stage 2: Hallucination Verification
        4. If FAIL: Self-Refinement pass with re-verification or safe fallback
        """
        logger.info(f"Initiating guarded query: '{query}' (filter={task_type_filter})")
        context_str, raw_docs = self.retrieve_context(query, task_type_filter=task_type_filter)

        # Stage 1: Generate initial answer
        initial_answer = self.generate_initial_answer(query, context_str)
        logger.info("Stage 1 completed. Initial answer generated.")

        # Stage 2: Hallucination Judge
        is_grounded, judge_output = self.judge_answer(query, context_str, initial_answer)
        logger.info(f"Stage 2 Judge Verdict: {'PASS' if is_grounded else 'FAIL'}")

        audit_log = [{
            "attempt": 1,
            "type": "initial_generation",
            "answer": initial_answer,
            "judge_verdict": "PASS" if is_grounded else "FAIL",
            "judge_reason": judge_output if not is_grounded else None
        }]

        if is_grounded:
            return {
                "query": query,
                "final_answer": initial_answer,
                "status": "VERIFIED_PASS",
                "retrieved_context": context_str,
                "num_chunks_retrieved": len(raw_docs),
                "audit_trail": audit_log,
            }

        # Failure branch: Self-Refinement Loop
        logger.warning(f"Hallucination detected by Stage 2: {judge_output}. Triggering refinement...")
        current_draft = initial_answer
        feedback = judge_output

        for attempt in range(1, max_refinements + 1):
            refined_answer = self.refine_answer(query, context_str, current_draft, feedback)
            re_judged, re_verdict = self.judge_answer(query, context_str, refined_answer)

            audit_log.append({
                "attempt": attempt + 1,
                "type": f"refinement_pass_{attempt}",
                "answer": refined_answer,
                "judge_verdict": "PASS" if re_judged else "FAIL",
                "judge_reason": re_verdict if not re_judged else None
            })

            if re_judged:
                logger.info("Refined answer successfully passed secondary verification.")
                return {
                    "query": query,
                    "final_answer": refined_answer,
                    "status": "REFINED_PASS",
                    "retrieved_context": context_str,
                    "num_chunks_retrieved": len(raw_docs),
                    "audit_trail": audit_log,
                }

            current_draft = refined_answer
            feedback = re_verdict

        # Safe fallback if refinements fail to clear hallucination
        logger.error("Self-refinement failed to eliminate hallucinations. Triggering safe fallback.")
        fallback_msg = (
            "Insufficient context: I cannot provide a fully factually grounded answer "
            f"based strictly on the available documents. (Reason: {feedback})"
        )
        return {
            "query": query,
            "final_answer": fallback_msg,
            "status": "FALLBACK_TRIGGERED",
            "retrieved_context": context_str,
            "num_chunks_retrieved": len(raw_docs),
            "audit_trail": audit_log,
        }


# =====================================================================
# 4. Standalone Convenience Wrapper
# =====================================================================

def ask_with_guard(query: str, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Convenience functional wrapper for single-line invocations."""
    guard = HallucinationGuard(llm=llm)
    return guard.ask_with_guard(query)


if __name__ == "__main__":
    # Note: Do not execute without explicit approval
    print("hallucination_guard.py scaffold loaded. Awaiting user execution authorization.")
