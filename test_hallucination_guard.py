"""
Test Suite: Hallucination Guard (Phase 3)
==========================================
Executes 3 test cases against the local Ollama LLM and ChromaDB vector store:
- Case A: Factually Supported (Apna Indian Kitchen operating hours)
- Case B: Hallucination-Inducing / Semi-Factual (Hours + Head Chef / Owner)
- Case C: Completely Out-of-Domain (Mars Rover Perseverance mission)
"""

import json
import logging
import time
from pathlib import Path

from hallucination_guard import HallucinationGuard, get_base_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_suite")

TEST_CASES = [
    {
        "id": "Case A",
        "name": "Factually Supported",
        "query": "Where is Apna Indian Kitchen located, and what is its average rating?",
        "expected_behavior": "Should retrieve chunk 0 of doc ragtruth_13599 and produce grounded answer passing Judge on attempt 1."
    },
    {
        "id": "Case B",
        "name": "Hallucination-Inducing / Semi-Factual",
        "query": "Where is Apna Indian Kitchen located, and who is the head chef and founder?",
        "expected_behavior": "Entity location exists in DB but chef/founder does not. Model must state insufficient context for the chef/founder or trigger refinement if ungrounded claims are made."
    },
    {
        "id": "Case C",
        "name": "Completely Out-of-Domain",
        "query": "What were the primary scientific instruments aboard NASA's Perseverance rover during its 2021 Mars landing mission?",
        "expected_behavior": "Context has zero info on Perseverance rover. Model should state 'Insufficient context' and pass factual verification."
    }
]


def run_tests(provider: str = "ollama", model_name: str = "qwen2.5-coder:14b"):
    print("=" * 80)
    print(f"RUNNING PHASE 3 HALLUCINATION GUARD TEST SUITE ({provider} / {model_name})")
    print("=" * 80)

    llm = get_base_llm(provider=provider, model_name=model_name, temperature=0.0)
    guard = HallucinationGuard(llm=llm, k=3)

    results = []

    for case in TEST_CASES:
        print("\n" + "-" * 70)
        print(f"Executing {case['id']}: {case['name']}")
        print(f"Query: \"{case['query']}\"")
        print("-" * 70)

        start_time = time.time()
        output = guard.ask_with_guard(case["query"])
        elapsed = time.time() - start_time

        audit_trail = output.get("audit_trail", [])
        initial_step = audit_trail[0] if audit_trail else {}
        initial_answer = initial_step.get("answer", "")
        judge_verdict = initial_step.get("judge_verdict", "UNKNOWN")
        judge_reason = initial_step.get("judge_reason")

        final_answer = output.get("final_answer", "")
        status = output.get("status", "UNKNOWN")

        print(f"\n[Initial Generation (Attempt 1)]:\n{initial_answer}")
        print(f"\n[Judge Verdict]: {judge_verdict}" + (f" -> Reason: {judge_reason}" if judge_reason else ""))
        if len(audit_trail) > 1:
            print(f"\n[Refinements Triggered]: {len(audit_trail) - 1} iteration(s)")
            for ref in audit_trail[1:]:
                print(f"  - {ref.get('type')}: {ref.get('judge_verdict')} (Answer: {ref.get('answer')[:120]}...)")
        print(f"\n[Status]: {status}")
        print(f"[Final Output]:\n{final_answer}")
        print(f"[Latency]: {elapsed:.2f}s")

        results.append({
            "case_id": case["id"],
            "name": case["name"],
            "query": case["query"],
            "initial_answer": initial_answer,
            "judge_verdict": judge_verdict,
            "judge_reason": judge_reason,
            "status": status,
            "final_answer": final_answer,
            "num_attempts": len(audit_trail),
            "latency_seconds": round(elapsed, 2)
        })

    # Save structured audit results
    output_path = Path(__file__).resolve().parent / "test_hallucination_guard_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All {len(TEST_CASES)} test cases completed. Saved audit logs to {output_path.name}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_tests()
