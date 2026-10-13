# Domain-Specific RAG Knowledge Base with Hallucination Detection & Self-Refinement

A production-grade, local-first Retrieval-Augmented Generation (RAG) system engineered for high-factual fidelity on domain-specific corpora. Features a dual-stage generation and verification pipeline, automated self-refinement guard loops, Model Context Protocol (MCP) tool exposure, and quantitative benchmark evaluation grounded on the **RAGTruth** dataset.

---

## 1. System Architecture

```
                          +------------------------------------------+
                          |             User Query / API             |
                          +------------------------------------------+
                                               |
                                               v
                         +--------------------------------------------+
                         |       Stage 0: Vector Retrieval            |
                         |   (BAAI/bge-small-en-v1.5 + ChromaDB)      |
                         +--------------------------------------------+
                                               |
                                        Top-k Chunks &
                                        Provenance Meta
                                               |
                                               v
                         +--------------------------------------------+
                         |       Stage 1: Grounded Generation         |
                         |        Strict Context-Only Prompt          |
                         |          (Default: Insufficient Context)  |
                         +--------------------------------------------+
                                               |
                                         Initial Draft
                                               |
                                               v
                         +--------------------------------------------+
                         |    Stage 2: Independent Hallucination      |
                         |            Verification Judge              |
                         +--------------------------------------------+
                                               |
                         +---------------------+----------------------+
                         |                                            |
                    [VERDICT: PASS]                              [VERDICT: FAIL]
                         |                                            |
                         v                                            v
               +-------------------+                     +--------------------------+
               |  Verified Output  |                     | Stage 3: Self-Refinement |
               |   (Zero Outside   |                     | Loop with Judge Feedback |
               |     Claims)       |                     +--------------------------+
               +-------------------+                                  |
                                                          +-----------+-----------+
                                                          |                       |
                                                     [Re-Judge: PASS]        [Re-Judge: FAIL]
                                                          |                       |
                                                          v                       v
                                                 +-------------------+  +-------------------+
                                                 |  Refined Grounded |  |   Safe Fallback   |
                                                 |      Output       |  |   (Insufficient   |
                                                 +-------------------+  |     Context)      |
                                                                        +-------------------+
```

---

## 2. Directory Tree Structure

```
Domain-Specific RAG Knowledge Base with Hallucination Detection/
├── Dataset/
│   ├── Raw_DATA_NLP/
│   │   ├── delucionqa/               # Excluded (raw numerical token IDs)
│   │   │   ├── train.csv
│   │   │   ├── dev.csv
│   │   │   └── test.csv
│   │   └── ragtruth/                 # Ground truth benchmark dataset
│   │       ├── source_info.jsonl     # Raw source texts across QA, Summary, Data2txt
│   │       └── response.jsonl        # Model responses with fine-grained hallucination labels
│   └── Processed_DATA_NLP/
│       └── ragtruth_cleaned_corpus.jsonl  # Preprocessed & normalized corpus (2,965 docs)
│
├── Database/
│   └── chroma_db/                    # Persistent vector database (26,700 indexed chunks)
│
├── Evaluation/
│   ├── phase4_metrics.json           # Quantitative benchmark results (Precision, Recall, Recovery)
│   └── batch_results.json            # Batch execution audit logs
│
├── preprocess_ragtruth.py            # Phase 1: Cleans, linearizes & standardizes source documents
├── build_vector_store.py             # Phase 2: Conditional text splitter & ChromaDB ingestion
├── mcp_server.py                     # Phase 2: Model Context Protocol (FastMCP) server
├── test_mcp_client.py                # Phase 2: Automated test client for MCP tool verification
├── hallucination_guard.py            # Phase 3: Dual-stage generation, judge & self-refinement core
├── test_hallucination_guard.py       # Phase 3: 3-case empirical test suite (Supported, Semi, Out)
├── app.py                            # Phase 4: Unified CLI entrypoint (Interactive REPL & Batch modes)
├── evaluate_guard.py                 # Phase 4: 50-sample balanced benchmark evaluation runner
└── README.md                         # Project documentation and orchestration guide
```

---

## 3. Key Design Decisions & Safety Constraints

1. **Local-First & Offline Resilience:**
   - **Vector Store:** Local ChromaDB instance with zero external database dependencies.
   - **Embeddings:** Native PyTorch + Hugging Face Transformers wrapper (`BgeEmbeddings`) running `BAAI/bge-small-en-v1.5` on CPU with `local_files_only=True` caching. Bypasses Windows Application Control restrictions on `_uarray.pyd` in `scipy`.
   - **LLM Inference:** Local zero-cost inference via Ollama (`qwen2.5-coder:14b` or `gemma4:e4b`) on `http://localhost:11434`, with optional cloud adapters for Google Gemini and OpenAI.
2. **Context Refusal Discernment:**
   - The Stage 2 Judge explicitly recognizes honest refusals (`"Insufficient context"`) as valid, faithful outputs rather than penalizing models when information is genuinely missing.
3. **Task-Specific Adaptive Chunking:**
   - **Summary Tasks:** `chunk_size=512, chunk_overlap=50` (captures discourse structure).
   - **QA Tasks:** `chunk_size=256, chunk_overlap=30` (pinpoints answer boundaries).
   - **Data2txt Tasks:** `chunk_size=256, chunk_overlap=0` (isolates linearized entity attributes).

---

## 4. Step-by-Step Reproduction Guide

### Step 1: Environment Setup
Ensure Python 3.10+ (tested on Python 3.14 on Windows) is active and install the core dependencies:

```bash
pip install langchain langchain-community langchain-text-splitters chromadb transformers torch mcp requests
```

Ensure Ollama is running locally with your desired model:
```bash
ollama run qwen2.5-coder:14b
```

---

### Step 2: Data Preprocessing (Phase 1)
Parses raw polymorphic structures from `Dataset/Raw_DATA_NLP/ragtruth/source_info.jsonl`, strips task-instruction boilerplate, linearizes JSON key-values into natural language, and writes the standardized corpus:

```bash
python preprocess_ragtruth.py
```
*Output: `Dataset/Processed_DATA_NLP/ragtruth_cleaned_corpus.jsonl` (2,965 documents, 7.06 MB).*

---

### Step 3: Vector Database Ingestion (Phase 2)
Chunks documents according to their `task_type` metadata, computes BGE embeddings, and populates the local ChromaDB database:

```bash
python build_vector_store.py
```
*Output: `Database/chroma_db` (26,700 chunks across Summary, QA, and Data2txt domains).*

---

### Step 4: Model Context Protocol (MCP) Server
Launch the MCP server exposing the `query_vector_db` tool for AI agents or IDE extensions:

```bash
python mcp_server.py
```

To run the automated MCP client verification test:
```bash
python test_mcp_client.py
```

---

### Step 5: Hallucination Guard Test Suite (Phase 3)
Run the 3-case empirical test suite verifying factually supported queries, hallucination-inducing semi-factual prompts, and out-of-domain queries:

```bash
python test_hallucination_guard.py
```

---

### Step 6: Unified Interactive CLI (Phase 4)
Launch the interactive terminal application:

```bash
python app.py
```

#### CLI Usage Examples:
* **Ask a factual question:**
  ```text
  Query > Where is Apna Indian Kitchen located, and what is its average rating?
  ```
* **Filter by task domain:**
  ```text
  Query > /filter Data2txt
  [Data2txt] Query > What are the operating hours of Apna Indian Kitchen?
  ```
* **Run a batch of questions from a file:**
  ```bash
  python app.py --batch questions.txt --out Evaluation/batch_results.json
  ```

---

### Step 7: Quantitative Benchmark Evaluation (Phase 4)
Run quantitative factual evaluation across 50 balanced samples from RAGTruth:

```bash
# Dry-run validation (instant syntax & pipeline check):
python evaluate_guard.py --dry-run

# Full benchmark evaluation (balanced 50 samples):
python evaluate_guard.py --sample-size 50 --provider ollama --model qwen2.5-coder:14b
```

---

## 5. Evaluation Metrics & Artifacts

The quantitative benchmark measures four core dimensions of factual integrity:
1. **Hallucination Detection Rate (Recall / TPR):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ &mdash; Sensitivity to known fabricated or conflicting claims.
2. **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ &mdash; Rate at which truly faithful answers are unnecessarily flagged.
3. **Judge Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ &mdash; Reliability of the Judge's flagged violations.
4. **Refinement Recovery Rate:** $\frac{\text{Recovered PASS}}{\text{Attempted Refinements}}$ &mdash; Ability of the self-correction loop to surgically excise hallucinations without discarding valid context.

Detailed benchmark runs are exported to [`Evaluation/phase4_metrics.json`](file:///d:/Turjoy%20Edu/Sem-09/NLP/Project_NLP/Domain-Specific%20RAG%20Knowledge%20Base%20with%20Hallucination%20Detection/Evaluation/phase4_metrics.json).
