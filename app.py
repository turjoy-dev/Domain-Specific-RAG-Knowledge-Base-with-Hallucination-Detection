"""
Unified Application Entrypoint & CLI Interface
===============================================
Wraps the HallucinationGuard engine into an interactive terminal CLI and batch query processor.

Modes:
  1. Interactive REPL Mode (default):
     - Prompt loop for user questions.
     - Displays:
         [1] Retrieved Context Chunks (with provenance metadata)
         [2] Initial Draft Answer (Stage 1)
         [3] Judge Verdict & Reason (Stage 2)
         [4] Final Verified / Refined Answer
  2. Batch Processing Mode:
     - Accepts a text/JSON file of questions via --batch <file_path>.
     - Executes guarded RAG across all inputs.
     - Exports structured audit metrics to an output JSON file.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from hallucination_guard import HallucinationGuard, get_base_llm

WORKSPACE_DIR = Path(__file__).resolve().parent

# Configure console logger
logging.basicConfig(
    level=logging.WARNING,  # Keep CLI clean
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("app_cli")


def print_banner(model_name: str, provider: str):
    banner = f"""
================================================================================
   RAGTRUTH DOMAIN-SPECIFIC KNOWLEDGE BASE & HALLUCINATION GUARD
   Provider: {provider.upper()} | Model: {model_name} | Local-First Architecture
================================================================================
 Type your question below, or type 'exit' or 'quit' to end the session.
 Options: Prefix with '/filter <TaskType>' to filter (Summary, QA, Data2txt).
================================================================================
"""
    print(banner)


def format_audit_step(attempt_info: Dict[str, Any]) -> str:
    attempt = attempt_info.get("attempt", 1)
    step_type = attempt_info.get("type", "generation")
    verdict = attempt_info.get("judge_verdict", "N/A")
    reason = attempt_info.get("judge_reason")

    verdict_badge = f"\033[92m[{verdict}]\033[0m" if verdict == "PASS" else f"\033[91m[{verdict}]\033[0m"
    res = f"  * Attempt {attempt} ({step_type}): {verdict_badge}"
    if reason:
        res += f"\n    Reason: {reason}"
    return res


def process_single_query(
    guard: HallucinationGuard,
    user_input: str,
    task_type_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Executes a single guarded query and prints the 4-stage pipeline output."""
    # Sanitize user input by stripping any copied prompt prefixes like 'Query >' or '[QA] Query >'
    clean_query = re.sub(r"^(?:\[[^\]]+\]\s*)?(?:Query\s*>\s*)+", "", user_input).strip()
    if not clean_query:
        clean_query = user_input.strip()

    start_t = time.time()
    print("\n" + "-" * 78, flush=True)
    filter_label = f" [Filter: {task_type_filter}]" if task_type_filter else ""
    print(f"[*] Processing Query{filter_label}: \"{clean_query}\"", flush=True)
    print("-" * 78, flush=True)

    # 1. Retrieval
    context_str, docs = guard.retrieve_context(clean_query, task_type_filter=task_type_filter)

    print("\n\033[1m[1] RETRIEVED CONTEXT CHUNKS:\033[0m", flush=True)
    if docs:
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata
            doc_id = meta.get("doc_id", "Unknown")
            task = meta.get("task_type", "Unknown")
            domain = meta.get("source_domain", "Unknown")
            chunk_idx = meta.get("chunk_index", 0)
            snippet = doc.page_content.strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            print(f"  • Chunk {idx} | Doc: {doc_id} (idx:{chunk_idx}) | Task: {task} | Domain: {domain}", flush=True)
            print(f"    Content: \"{snippet}\"", flush=True)
    else:
        print("  (No relevant chunks found in vector database)", flush=True)

    # 2. Execute Guard Loop
    output = guard.ask_with_guard(clean_query, task_type_filter=task_type_filter)
    audit_trail = output.get("audit_trail", [])

    # Initial Draft
    initial_draft = audit_trail[0].get("answer", "") if audit_trail else ""
    print(f"\n\033[1m[2] INITIAL DRAFT ANSWER (Stage 1):\033[0m", flush=True)
    print(f"  {initial_draft}", flush=True)

    # Judge Verdict
    initial_verdict = audit_trail[0].get("judge_verdict", "N/A") if audit_trail else "N/A"
    initial_reason = audit_trail[0].get("judge_reason") if audit_trail else None
    badge = f"\033[92m{initial_verdict}\033[0m" if initial_verdict == "PASS" else f"\033[91m{initial_verdict}\033[0m"

    print(f"\n\033[1m[3] JUDGE VERDICT (Stage 2):\033[0m {badge}", flush=True)
    if initial_reason:
        print(f"  Feedback: {initial_reason}", flush=True)

    # Refinement steps if any
    if len(audit_trail) > 1:
        print("\n\033[1m[+] SELF-REFINEMENT PASSES:\033[0m", flush=True)
        for step in audit_trail[1:]:
            print(format_audit_step(step), flush=True)

    # Final Output
    final_status = output.get("status", "UNKNOWN")
    final_ans = output.get("final_answer", "")
    elapsed = time.time() - start_t

    print(f"\n\033[1m[4] FINAL GROUNDED ANSWER [{final_status}]:\033[0m", flush=True)
    print(f"  {final_ans}", flush=True)
    print(f"\n  [Latency: {elapsed:.2f}s]", flush=True)
    print("-" * 78, flush=True)
    return output


def run_interactive(guard: HallucinationGuard):
    """Executes the interactive REPL prompt loop."""
    print_banner(
        model_name=getattr(guard.llm, "model", "default"),
        provider=getattr(guard.llm, "__class__", type(guard.llm)).__name__
    )

    task_filter: Optional[str] = None

    while True:
        try:
            prompt_label = f"[{task_filter}] Query > " if task_filter else "Query > "
            user_input = input(f"\n{prompt_label}").strip()

            if not user_input:
                continue

            # Strip accidental copy-pasted prefixes like 'Query >' or '[QA] Query >'
            clean_input = re.sub(r"^(?:\[[^\]]+\]\s*)?(?:Query\s*>\s*)+", "", user_input).strip()
            if not clean_input:
                continue

            if clean_input.lower() in ("exit", "quit", "q"):
                print("\nExiting session. Goodbye!", flush=True)
                break

            if clean_input.startswith("/filter"):
                parts = clean_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    task_filter = parts[1].strip()
                    print(f"Task filter active: '{task_filter}'", flush=True)
                else:
                    task_filter = None
                    print("Task filter cleared.", flush=True)
                continue

            process_single_query(guard, clean_input, task_type_filter=task_filter)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.", flush=True)
            break
        except Exception as e:
            print(f"\n\033[91m[Error]\033[0m Failed to process query: {e}", flush=True)


def run_batch(guard: HallucinationGuard, input_file: Path, output_file: Path):
    """Executes guarded queries in batch from an input file and saves results."""
    if not input_file.exists():
        print(f"[Error] Batch input file not found: {input_file}")
        sys.exit(1)

    queries: List[str] = []
    if input_file.suffix.lower() == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        queries.append(item)
                    elif isinstance(item, dict) and "query" in item:
                        queries.append(item["query"])
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Loaded {len(queries)} queries for batch execution from {input_file.name}")
    results = []
    total_start = time.time()

    for idx, q in enumerate(queries, start=1):
        print(f"[{idx}/{len(queries)}] Processing: \"{q[:60]}...\"")
        t0 = time.time()
        res = guard.ask_with_guard(q)
        dt = time.time() - t0
        res["latency_seconds"] = round(dt, 2)
        results.append(res)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_start
    print(f"\nBatch processing complete! Processed {len(queries)} queries in {total_time:.2f}s.")
    print(f"Saved audit log to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for RAGTruth Knowledge Base with Hallucination Guard"
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "google", "openai", "mock"],
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
        help="Model identifier (default: qwen2.5-coder:14b)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Inference temperature (default: 0.0 for deterministic grounding)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks (default: 3)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Execute a single query and exit (useful for scripted or headless execution)"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to batch input file containing questions (.txt or .json)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Evaluation/batch_results.json",
        help="Path to save batch evaluation results"
    )

    args = parser.parse_args()

    # Initialize components
    llm = get_base_llm(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature
    )
    guard = HallucinationGuard(llm=llm, k=args.top_k)

    if args.query:
        process_single_query(guard, args.query)
    elif args.batch:
        run_batch(guard, Path(args.batch), WORKSPACE_DIR / args.out)
    else:
        run_interactive(guard)


if __name__ == "__main__":
    main()
