import os
import json
import re
import csv
import statistics
from collections import Counter

WORKSPACE_DIR = r"d:\Turjoy Edu\Sem-09\NLP\Project_NLP\Domain-Specific RAG Knowledge Base with Hallucination Detection"
RAW_DATA_DIR = os.path.join(WORKSPACE_DIR, "Dataset", "Raw_DATA_NLP")

def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())

def check_html(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"<[^>]+>", text))

def check_excessive_whitespace(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[ \t]{3,}|\n{3,}", text))

def check_mojibake(text):
    if not isinstance(text, str):
        return False
    # Common mojibake patterns from utf-8 misinterpreted as latin1/windows-1252
    patterns = [r"Ã[\x80-\xbf]", r"â€[™œžœ]", r"\ufffd", r"&amp;", r"&lt;", r"&gt;"]
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def check_unparsed_escapes(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\\n|\\t|\\r|\\u[0-9a-fA-F]{4}", text))

def diagnose_delucionqa():
    delucion_dir = os.path.join(RAW_DATA_DIR, "Delucionqa")
    files = ["train.csv", "dev.csv", "test.csv"]
    results = {}

    for fname in files:
        fpath = os.path.join(delucion_dir, fname)
        if not os.path.exists(fpath):
            continue

        stats = {
            "file": fname,
            "row_count": 0,
            "columns": [],
            "missing_values": {},
            "empty_strings": {},
            "encoding_issues": 0,
            "mojibake_count": 0,
            "unparsed_escapes": 0,
            "html_tag_rows": 0,
            "excessive_whitespace_rows": 0,
            "context_is_token_ids": 0,
            "word_counts": {
                "Question": [],
                "Context": [],
                "Answer": []
            },
            "labels_dist": Counter(),
            "anomalies": []
        }

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                stats["columns"] = reader.fieldnames or []
                for col in stats["columns"]:
                    stats["missing_values"][col] = 0
                    stats["empty_strings"][col] = 0

                for idx, row in enumerate(reader):
                    stats["row_count"] += 1
                    for col in stats["columns"]:
                        val = row.get(col)
                        if val is None:
                            stats["missing_values"][col] += 1
                        elif val.strip() == "":
                            stats["empty_strings"][col] += 1

                    q = row.get("Question", "") or ""
                    ctx = row.get("Context", "") or ""
                    ans = row.get("Answer", "") or ""
                    lbl = row.get("Label", "") or ""

                    stats["labels_dist"][lbl] += 1

                    # Check Context format: Is it a list of integers?
                    if ctx.strip().startswith("[") and ctx.strip().endswith("]"):
                        # check if numbers
                        inner = ctx.strip()[1:-1].strip()
                        if inner and all(part.strip().isdigit() for part in inner.split(",")[:10]):
                            stats["context_is_token_ids"] += 1

                    for text_field, name in [(q, "Question"), (ctx, "Context"), (ans, "Answer")]:
                        stats["word_counts"][name].append(count_words(text_field))
                        if check_mojibake(text_field):
                            stats["mojibake_count"] += 1
                        if check_unparsed_escapes(text_field):
                            stats["unparsed_escapes"] += 1

                    combined_text = f"{q} {ans}"
                    if check_html(combined_text):
                        stats["html_tag_rows"] += 1
                    if check_excessive_whitespace(combined_text):
                        stats["excessive_whitespace_rows"] += 1

        except UnicodeDecodeError as e:
            stats["encoding_issues"] += 1
            stats["anomalies"].append(f"UnicodeDecodeError: {str(e)}")

        results[fname] = stats

    return results

def diagnose_ragtruth():
    rag_dir = os.path.join(RAW_DATA_DIR, "ragtruth")
    files = ["source_info.jsonl", "response.jsonl"]
    results = {}

    for fname in files:
        fpath = os.path.join(rag_dir, fname)
        if not os.path.exists(fpath):
            continue

        stats = {
            "file": fname,
            "row_count": 0,
            "keys": set(),
            "missing_fields": Counter(),
            "empty_fields": Counter(),
            "malformed_lines": 0,
            "mojibake_count": 0,
            "unparsed_escapes": 0,
            "html_tag_rows": 0,
            "excessive_whitespace_rows": 0,
            "word_counts": {},
            "task_types": Counter(),
            "models": Counter(),
            "hallucination_counts": Counter(),
            "anomalies": []
        }

        with open(fpath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    stats["malformed_lines"] += 1
                    continue

                stats["row_count"] += 1
                for k in data.keys():
                    stats["keys"].add(k)

                if fname == "source_info.jsonl":
                    stats["task_types"][data.get("task_type", "UNKNOWN")] += 1
                    src_text = data.get("source_info", "") or ""
                    prompt = data.get("prompt", "") or ""

                    for field_name, raw_val in [("source_info", src_text), ("prompt", prompt)]:
                        val_type = type(raw_val).__name__
                        if "types_" + field_name not in stats:
                            stats["types_" + field_name] = Counter()
                        stats["types_" + field_name][val_type] += 1

                        if isinstance(raw_val, dict) or isinstance(raw_val, list):
                            text = json.dumps(raw_val)
                        elif isinstance(raw_val, str):
                            text = raw_val
                        else:
                            text = str(raw_val) if raw_val is not None else ""

                        if field_name not in stats["word_counts"]:
                            stats["word_counts"][field_name] = []
                        stats["word_counts"][field_name].append(count_words(text))
                        if not text.strip():
                            stats["empty_fields"][field_name] += 1
                        if check_mojibake(text):
                            stats["mojibake_count"] += 1
                        if check_unparsed_escapes(text):
                            stats["unparsed_escapes"] += 1
                        if check_html(text):
                            stats["html_tag_rows"] += 1
                        if check_excessive_whitespace(text):
                            stats["excessive_whitespace_rows"] += 1

                elif fname == "response.jsonl":
                    stats["models"][data.get("model", "UNKNOWN")] += 1
                    resp = data.get("response", "") or ""
                    labels = data.get("labels", [])

                    has_hallucination = len(labels) > 0
                    stats["hallucination_counts"]["hallucinated" if has_hallucination else "faithful"] += 1

                    if "response" not in stats["word_counts"]:
                        stats["word_counts"]["response"] = []
                    stats["word_counts"]["response"].append(count_words(resp))

                    if not resp.strip():
                        stats["empty_fields"]["response"] += 1
                    if check_mojibake(resp):
                        stats["mojibake_count"] += 1
                    if check_unparsed_escapes(resp):
                        stats["unparsed_escapes"] += 1
                    if check_html(resp):
                        stats["html_tag_rows"] += 1
                    if check_excessive_whitespace(resp):
                        stats["excessive_whitespace_rows"] += 1

        results[fname] = stats

    return results

if __name__ == "__main__":
    del_res = diagnose_delucionqa()
    rag_res = diagnose_ragtruth()

    output = {
        "delucionqa": del_res,
        "ragtruth": {
            k: {
                **v,
                "keys": list(v["keys"]),
                "missing_fields": dict(v["missing_fields"]),
                "empty_fields": dict(v["empty_fields"]),
                "task_types": dict(v["task_types"]),
                "models": dict(v["models"]),
                "hallucination_counts": dict(v["hallucination_counts"])
            } for k, v in rag_res.items()
        }
    }

    # Print summary metrics
    print("=== DELUCIONQA SUMMARY ===")
    for fname, d in del_res.items():
        print(f"\n--- {fname} ---")
        print(f"Total Rows: {d['row_count']}")
        print(f"Columns: {d['columns']}")
        print(f"Missing Values: {d['missing_values']}")
        print(f"Empty Strings: {d['empty_strings']}")
        print(f"Context is Token IDs: {d['context_is_token_ids']} / {d['row_count']}")
        print(f"Mojibake hits: {d['mojibake_count']}")
        print(f"Unparsed escapes hits: {d['unparsed_escapes']}")
        print(f"HTML tag rows: {d['html_tag_rows']}")
        print(f"Excessive whitespace rows: {d['excessive_whitespace_rows']}")
        print(f"Labels: {dict(d['labels_dist'])}")
        for col, wc in d["word_counts"].items():
            if wc:
                print(f"Word count [{col}]: mean={statistics.mean(wc):.1f}, min={min(wc)}, max={max(wc)}, median={statistics.median(wc)}")

    print("\n=== RAGTRUTH SUMMARY ===")
    for fname, r in rag_res.items():
        print(f"\n--- {fname} ---")
        print(f"Total Lines: {r['row_count']}")
        print(f"Malformed Lines: {r['malformed_lines']}")
        print(f"Keys: {list(r['keys'])}")
        print(f"Empty Fields: {dict(r['empty_fields'])}")
        print(f"Mojibake hits: {r['mojibake_count']}")
        print(f"Unparsed escapes hits: {r['unparsed_escapes']}")
        print(f"HTML tag rows: {r['html_tag_rows']}")
        print(f"Excessive whitespace rows: {r['excessive_whitespace_rows']}")
        if r["task_types"]:
            print(f"Task types: {dict(r['task_types'])}")
        if r["models"]:
            print(f"Models: {dict(r['models'])}")
        if r["hallucination_counts"]:
            print(f"Hallucination Distribution: {dict(r['hallucination_counts'])}")
        if "types_source_info" in r:
            print(f"source_info types: {dict(r['types_source_info'])}")
        if "types_prompt" in r:
            print(f"prompt types: {dict(r['types_prompt'])}")
        for col, wc in r["word_counts"].items():
            if wc:
                print(f"Word count [{col}]: mean={statistics.mean(wc):.1f}, min={min(wc)}, max={max(wc)}, median={statistics.median(wc)}")
