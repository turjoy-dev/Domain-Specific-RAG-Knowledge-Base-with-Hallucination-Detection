"""
RAGTruth Knowledge Base Preprocessing Pipeline
==============================================
Parses, linearizes, normalizes, and cleans source documents from RAGTruth
to produce standardized, chunk-ready documents for vector store ingestion.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ragtruth_preprocessor")


# =====================================================================
# 1. Boilerplate Stripping & Text Normalization
# =====================================================================

BOILERPLATE_PATTERNS = [
    r"^Summarize the following (news|article|text)?\s*(within|in)?\s*\d*\s*words?:?\s*",
    r"^Briefly answer the following question:?\s*",
    r"^Instruction:\s*",
    r"^Overview:\s*",
    r"\n+Bear in mind that your response should be[^\n]*",
    r"\n+If no relevant information is found[^\n]*",
    r"\n+output:\s*$",
    r"^output:\s*",
]

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\n{3,}")


def strip_boilerplate(text: str) -> str:
    """Removes LLM task-instruction prefixes, directives, and output flags."""
    cleaned = text.strip()
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    """
    Cleans raw text:
    - Normalizes non-breaking spaces (\u00a0) and carriage returns
    - Strips residual HTML tags
    - Collapses excessive horizontal and vertical whitespace
    - Removes instruction boilerplate
    """
    if not text:
        return ""

    # Replace non-breaking spaces and unify newline characters
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Collapse horizontal whitespace
    text = WHITESPACE_PATTERN.sub(" ", text)

    # Collapse excessive blank lines (max 2 consecutive newlines)
    text = NEWLINE_PATTERN.sub("\n\n", text)

    # Remove task instruction wrappers
    text = strip_boilerplate(text)

    return text.strip()


# =====================================================================
# 2. Polymorphic Task Parsers
# =====================================================================

def parse_summary_task(source_info: Any) -> str:
    """
    Handles 'Summary' tasks (e.g., CNN/DailyMail news articles).
    Plain string is treated as standard reference document.
    """
    if isinstance(source_info, str):
        return normalize_text(source_info)
    elif isinstance(source_info, dict):
        return normalize_text(str(source_info.get("text", "")))
    return normalize_text(str(source_info))


def parse_qa_task(source_info: Any) -> str:
    """
    Handles 'QA' tasks.
    Extracts and concatenates passages, cleanly removing 'passage X:' tags.
    """
    if not isinstance(source_info, dict):
        return normalize_text(str(source_info))

    passages_raw = source_info.get("passages", "")
    extracted_passages: List[str] = []

    if isinstance(passages_raw, str):
        # Split on passage markers while capturing the body
        segments = re.split(r"(?i)passage\s+\d+:\s*", passages_raw)
        for seg in segments:
            cleaned_seg = normalize_text(seg)
            if cleaned_seg:
                extracted_passages.append(cleaned_seg)
    elif isinstance(passages_raw, list):
        for p in passages_raw:
            cleaned_seg = normalize_text(str(p))
            if cleaned_seg:
                extracted_passages.append(cleaned_seg)

    if not extracted_passages:
        return normalize_text(str(passages_raw))

    # Join passages with clear paragraph boundaries
    return "\n\n".join(extracted_passages)


def linearize_hours(hours_dict: Dict[str, Any]) -> str:
    """Formats operating hours into natural language."""
    if not isinstance(hours_dict, dict) or not hours_dict:
        return ""
    schedule = [f"{day}: {hours}" for day, hours in hours_dict.items() if hours]
    if not schedule:
        return ""
    return f"Operating hours are {', '.join(schedule)}."


def linearize_attributes(attrs_dict: Dict[str, Any]) -> str:
    """
    Formats business attributes into natural language sentences.
    Sanitizes segments to prevent double periods and punctuation conflicts.
    """
    if not isinstance(attrs_dict, dict) or not attrs_dict:
        return ""

    segments: List[str] = []
    for key, val in attrs_dict.items():
        if val is None:
            continue

        raw_segment = ""
        human_key = re.sub(r"([A-Z])", r" \1", key).strip()

        if isinstance(val, dict):
            # E.g. Ambience dict
            sub_items = [k for k, v in val.items() if v is True]
            if sub_items:
                raw_segment = f"The {human_key.lower()} is {', '.join(sub_items)}"
        elif val is True:
            raw_segment = f"Offers {human_key.lower()}"
        elif val is False:
            raw_segment = f"Does not offer {human_key.lower()}"
        elif isinstance(val, str) and val.strip().lower() not in ("none", ""):
            cleaned_val = val.strip().rstrip(".:;,")
            raw_segment = f"{human_key}: {cleaned_val}"

        # Clean trailing punctuation from segment to prevent double periods
        raw_segment = raw_segment.strip().rstrip(".:;,")
        if raw_segment:
            segments.append(raw_segment)

    if not segments:
        return ""

    # Join cleanly and terminate with a single period
    return ". ".join(segments) + "."


def linearize_reviews(reviews_list: List[Dict[str, Any]], max_reviews: int = 5) -> str:
    """Converts customer review records into readable review summaries."""
    if not isinstance(reviews_list, list) or not reviews_list:
        return ""

    review_texts: List[str] = []
    for r in reviews_list[:max_reviews]:
        if not isinstance(r, dict):
            continue
        stars = r.get("review_stars", "")
        text = normalize_text(r.get("review_text", ""))
        if text:
            star_prefix = f"Rated {stars} stars: " if stars else ""
            review_texts.append(f"{star_prefix}\"{text}\"")

    if not review_texts:
        return ""
    return "Customer reviews: " + " | ".join(review_texts)


def parse_data2txt_task(source_info: Any) -> str:
    """
    Handles 'Data2txt' tasks (Yelp business entity records).
    Linearizes structured metadata into coherent, natural sentences.
    """
    if not isinstance(source_info, dict):
        return normalize_text(str(source_info))

    name = source_info.get("name", "Unknown Business")
    address = source_info.get("address", "")
    city = source_info.get("city", "")
    state = source_info.get("state", "")
    categories = source_info.get("categories", "")
    stars = source_info.get("business_stars")

    components: List[str] = []

    # 1. Identity & Location
    location_parts = [p.strip().rstrip(".,") for p in [address, city, state] if p and p.strip()]
    location_str = f" located at {', '.join(location_parts)}" if location_parts else ""
    categories_str = f" categorized under {categories.strip().rstrip('.')}" if categories else ""
    components.append(f"{name} is a business{location_str}{categories_str}.")

    # 2. Rating
    if stars is not None:
        components.append(f"It has an average rating of {stars} out of 5 stars.")

    # 3. Hours
    hours_str = linearize_hours(source_info.get("hours", {}))
    if hours_str:
        components.append(hours_str)

    # 4. Attributes (sanitized against double punctuation)
    attrs_str = linearize_attributes(source_info.get("attributes", {}))
    if attrs_str:
        components.append(attrs_str)

    # 5. Reviews
    reviews_str = linearize_reviews(source_info.get("review_info", []))
    if reviews_str:
        components.append(reviews_str)

    combined = " ".join(components)
    return normalize_text(combined)


# =====================================================================
# 3. Main Preprocessing Engine
# =====================================================================

def process_record(raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dispatches a single JSONL record to the appropriate task parser."""
    source_id = raw_record.get("source_id", "")
    task_type = raw_record.get("task_type", "Unknown")
    source_domain = raw_record.get("source", "Unknown")
    raw_source_info = raw_record.get("source_info")

    if raw_source_info is None:
        logger.warning(f"Skipping record {source_id}: missing 'source_info'.")
        return None

    # Polymorphic dispatch
    if task_type == "Summary":
        clean_text = parse_summary_task(raw_source_info)
    elif task_type == "QA":
        clean_text = parse_qa_task(raw_source_info)
    elif task_type == "Data2txt":
        clean_text = parse_data2txt_task(raw_source_info)
    else:
        clean_text = normalize_text(str(raw_source_info))

    if not clean_text:
        logger.warning(f"Record {source_id} yielded empty text after cleaning.")
        return None

    return {
        "doc_id": f"ragtruth_{source_id}",
        "source_id": str(source_id),
        "task_type": task_type,
        "source_domain": source_domain,
        "clean_text": clean_text,
        "char_count": len(clean_text),
        "word_count": len(clean_text.split()),
        "metadata": {
            "task_type": task_type,
            "source": source_domain,
            "original_keys": list(raw_record.keys()),
        }
    }


def run_pipeline(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None
) -> Tuple[int, int]:
    """
    Reads RAGTruth source_info.jsonl, processes all records,
    and writes standardized JSONL records ready for chunking & embedding.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    # Ensure parent output directory exists gracefully
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.error(f"Failed to create output directory {output_path.parent}: {err}")
        raise

    total_records = 0
    successful_records = 0
    task_counts: Dict[str, int] = {}

    logger.info(f"Starting ingestion from: {input_path}")
    logger.info(f"Target output file: {output_path}")

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:

        for line_num, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1
            raw_record = json.loads(line)
            processed = process_record(raw_record)

            if processed:
                out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                successful_records += 1
                tt = processed["task_type"]
                task_counts[tt] = task_counts.get(tt, 0) + 1

            if limit and successful_records >= limit:
                break

    logger.info(
        f"Processing complete: {successful_records}/{total_records} records written successfully to {output_path}."
    )
    logger.info(f"Task Breakdown: {task_counts}")
    return total_records, successful_records


if __name__ == "__main__":
    # Dynamic relative path resolution
    WORKSPACE = Path(__file__).resolve().parent
    INPUT_FILE = WORKSPACE / "Dataset" / "Raw_DATA_NLP" / "ragtruth" / "source_info.jsonl"
    OUTPUT_FILE = WORKSPACE / "Dataset" / "Processed_DATA_NLP" / "ragtruth_cleaned_corpus.jsonl"

    print(f"Executing Preprocessing Pipeline...")
    total, successful = run_pipeline(INPUT_FILE, OUTPUT_FILE)
    print(f"DONE: {successful}/{total} records processed.")
