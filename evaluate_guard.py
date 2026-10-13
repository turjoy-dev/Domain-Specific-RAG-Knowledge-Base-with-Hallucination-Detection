"""
Quantitative Evaluation Runner for Hallucination Guard
======================================================
Evaluates the factual verification Judge and self-refinement loop against
annotated ground-truth test splits in the RAGTruth dataset.

Ground-Truth Annotations:
  - Dataset/Raw_DATA_NLP/ragtruth/response.jsonl
    * Faithful: len(labels) == 0
    * Hallucinated: len(labels) > 0 (contains Evident Conflict, Baseless Info, etc.)

Metrics Computed:
  1. Judge Detection Rate / Sensitivity (Recall on Hallucinations): TP / (TP + FN)
  2. False Positive Rate on Faithful Samples: FP / (FP + TN)
  3. Judge Precision on Flagged Hallucinations: TP / (TP + FP)
  4. Overall Judge Classification Accuracy: (TP + TN) / Total
  5. Refinement Recovery Rate: (% of flagged drafts successfully grounded via refinement)
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hallucination_guard import HallucinationGuard, get_base_llm

WORKSPACE_DIR = Path(__file__).resolve().parent
RESPONSE_FILE = WORKSPACE_DIR / "Dataset" / "Raw_DATA_NLP" / "ragtruth" / "response.jsonl"
CORPUS_FILE = WORKSPACE_DIR / "Dataset" / "Processed_DATA_NLP" / "ragtruth_cleaned_corpus.jsonl"
OUTPUT_METRICS_FILE = WORKSPACE_DIR / "Evaluation" / "phase4_metrics.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eval_runner")


def load_cleaned_corpus() -> Dict[str, Dict[str, Any]]:
    """Loads the preprocessed RAGTruth corpus indexed by doc_id."""
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Cleaned corpus not found at {CORPUS_FILE}")

    corpus = {}
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                corpus[item["doc_id"]] = item
    logger.info(f"Loaded {len(corpus)} documents from cleaned corpus.")
    return corpus


def sample_balanced_responses(
    sample_size: int = 50,
    seed: int = 42,
    corpus: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Samples balanced faithful and hallucinated responses from RAGTruth response.jsonl.
    Ensures matching source documents exist in our cleaned corpus.
    """
    if not RESPONSE_FILE.exists():
        raise FileNotFoundError(f"Response file not found at {RESPONSE_FILE}")

    faithful_pool: List[Dict[str, Any]] = []
    hallucinated_pool: List[Dict[str, Any]] = []

    with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            source_id = str(rec.get("source_id", ""))
            doc_id = f"ragtruth_{source_id}"

            # Only sample if source doc exists in our cleaned corpus
            if corpus and doc_id not in corpus:
                continue

            labels = rec.get("labels", [])
            if len(labels) == 0:
                faithful_pool.append(rec)
            else:
                hallucinated_pool.append(rec)

    random.seed(seed)
    half = sample_size // 2
    sampled_faithful = random.sample(faithful_pool, min(half, len(faithful_pool)))
    sampled_hallucinated = random.sample(hallucinated_pool, min(half, len(hallucinated_pool)))

    logger.info(
        f"Sampled {len(sampled_faithful)} faithful and "
        f"{len(sampled_hallucinated)} hallucinated responses (seed={seed})."
    )
    return sampled_faithful, sampled_hallucinated


def run_evaluation(
    guard: HallucinationGuard,
    sample_size: int = 50,
    seed: int = 42,
    max_refinements: int = 1
) -> Dict[str, Any]:
    """Executes quantitative evaluation of Judge and Refinement on sampled records."""
    corpus = load_cleaned_corpus()
    faithful_samples, hallucinated_samples = sample_balanced_responses(
        sample_size=sample_size,
        seed=seed,
        corpus=corpus
    )

    all_samples = []
    for s in faithful_samples:
        all_samples.append({"record": s, "is_hallucinated": False})
    for s in hallucinated_samples:
        all_samples.append({"record": s, "is_hallucinated": True})

    # Shuffle for mixed execution order
    random.seed(seed)
    random.shuffle(all_samples)

    tp = 0  # Ground truth hallucinated, Judge flags FAIL
    fn = 0  # Ground truth hallucinated, Judge says PASS
    tn = 0  # Ground truth faithful, Judge says PASS
    fp = 0  # Ground truth faithful, Judge flags FAIL

    refinement_attempts = 0
    refinement_successes = 0

    detailed_eval_records = []
    start_total_time = time.time()

    print("=" * 80)
    print(f"STARTING PHASE 4 QUANTITATIVE BENCHMARK EVALUATION (N={len(all_samples)})")
    print("=" * 80)

    for idx, item in enumerate(all_samples, start=1):
        rec = item["record"]
        is_ground_truth_hallucinated = item["is_hallucinated"]
        source_id = str(rec["source_id"])
        doc_id = f"ragtruth_{source_id}"
        source_doc = corpus.get(doc_id, {})
        source_context = source_doc.get("clean_text", "")
        task_type = source_doc.get("task_type", "Unknown")
        domain = source_doc.get("source_domain", "Unknown")
        candidate_response = rec.get("response", "")

        t0 = time.time()
        # Stage 2 Judge Evaluation
        passed, judge_feedback = guard.judge_answer(
            query=f"Task: {task_type} for document {doc_id}",
            context=source_context,
            answer=candidate_response
        )
        judge_predicted_hallucinated = not passed

        # Confusion Matrix updates
        if is_ground_truth_hallucinated and judge_predicted_hallucinated:
            tp += 1
            verdict_match = "CORRECT_DETECTION (TP)"
        elif is_ground_truth_hallucinated and not judge_predicted_hallucinated:
            fn += 1
            verdict_match = "MISSED_HALLUCINATION (FN)"
        elif not is_ground_truth_hallucinated and not judge_predicted_hallucinated:
            tn += 1
            verdict_match = "CORRECT_FAITHFUL (TN)"
        else:
            fp += 1
            verdict_match = "FALSE_ALARM (FP)"

        # Self-Refinement Test (if flagged as hallucination)
        refined_output = None
        refined_passed = None
        if judge_predicted_hallucinated:
            refinement_attempts += 1
            refined_output = guard.refine_answer(
                query=f"Task: {task_type} for document {doc_id}",
                context=source_context,
                draft_answer=candidate_response,
                feedback=judge_feedback
            )
            refined_passed, _ = guard.judge_answer(
                query=f"Task: {task_type} for document {doc_id}",
                context=source_context,
                answer=refined_output
            )
            if refined_passed:
                refinement_successes += 1

        dt = time.time() - t0

        print(
            f"[{idx}/{len(all_samples)}] Doc: {doc_id} | "
            f"GroundTruth: {'HALLUCINATED' if is_ground_truth_hallucinated else 'FAITHFUL'} | "
            f"Judge: {'FAIL' if judge_predicted_hallucinated else 'PASS'} -> {verdict_match} ({dt:.2f}s)"
        )

        detailed_eval_records.append({
            "sample_index": idx,
            "response_id": rec.get("id"),
            "doc_id": doc_id,
            "task_type": task_type,
            "domain": domain,
            "ground_truth_hallucinated": is_ground_truth_hallucinated,
            "ground_truth_labels": rec.get("labels", []),
            "judge_passed": passed,
            "judge_feedback": judge_feedback if not passed else None,
            "verdict_match": verdict_match,
            "refinement_attempted": judge_predicted_hallucinated,
            "refined_passed": refined_passed,
            "latency_seconds": round(dt, 2)
        })

    total_time = time.time() - start_total_time

    # Calculate final aggregate metrics
    detection_rate_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(all_samples) if all_samples else 0.0
    f1 = (2 * precision * detection_rate_recall) / (precision + detection_rate_recall) if (precision + detection_rate_recall) > 0 else 0.0
    recovery_rate = refinement_successes / refinement_attempts if refinement_attempts > 0 else 0.0

    metrics_summary = {
        "benchmark_metadata": {
            "dataset": "RAGTruth",
            "sample_size": len(all_samples),
            "faithful_count": tn + fp,
            "hallucinated_count": tp + fn,
            "total_execution_time_seconds": round(total_time, 2),
            "avg_latency_per_sample_seconds": round(total_time / len(all_samples), 2) if all_samples else 0.0
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp
        },
        "judge_metrics": {
            "detection_rate_recall": round(detection_rate_recall, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4)
        },
        "refinement_metrics": {
            "total_refinements_attempted": refinement_attempts,
            "successful_recoveries": refinement_successes,
            "recovery_rate": round(recovery_rate, 4)
        },
        "samples": detailed_eval_records
    }

    OUTPUT_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4 QUANTITATIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Samples Evaluated:            {len(all_samples)}")
    print(f"Hallucination Detection Rate (TPR): {detection_rate_recall * 100:.2f}% (TP={tp}, FN={fn})")
    print(f"False Positive Rate on Faithful:    {false_positive_rate * 100:.2f}% (FP={fp}, TN={tn})")
    print(f"Judge Precision on Flagged:         {precision * 100:.2f}%")
    print(f"Overall Classification Accuracy:    {accuracy * 100:.2f}%")
    print(f"F1 Score:                           {f1 * 100:.2f}%")
    print(f"Refinement Recovery Rate:           {recovery_rate * 100:.2f}% ({refinement_successes}/{refinement_attempts})")
    print(f"Saved complete evaluation report to: {OUTPUT_METRICS_FILE.resolve()}")
    print("=" * 80)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="RAGTruth Quantitative Benchmark Evaluator")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of records to evaluate (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "google", "openai", "mock"])
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock LLM for rapid pipeline test")

    args = parser.parse_args()

    provider = "mock" if args.dry_run else args.provider
    llm = get_base_llm(provider=provider, model_name=args.model, temperature=0.0)
    guard = HallucinationGuard(llm=llm)

    run_evaluation(guard, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
