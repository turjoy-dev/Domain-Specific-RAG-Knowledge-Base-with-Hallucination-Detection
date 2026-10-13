"""
Streamlit Web UI: Domain-Specific RAG with Hallucination Detection
==================================================================
Interactive web application wrapping the Hybrid Retrieval (BM25 + ChromaDB)
and Dual-Stage Hallucination Guard pipeline.
"""

import time
from typing import Optional

import streamlit as st
from langchain_core.documents import Document

from hallucination_guard import HallucinationGuard, get_base_llm

# =====================================================================
# 1. Page Configuration & Custom Styling
# =====================================================================

st.set_page_config(
    page_title="Domain-Specific RAG with Hallucination Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-refined {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-fallback {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. Pipeline Initialization (Cached for High Performance)
# =====================================================================

@st.cache_resource(show_spinner="Initializing Hybrid Retrieval Engine & Local Models...")
def load_guard_pipeline(provider: str = "ollama", model_name: str = "qwen2.5-coder:14b") -> HallucinationGuard:
    """Loads and caches the Hybrid Retriever (BM25 + ChromaDB) and Base LLM."""
    llm = get_base_llm(provider=provider, model_name=model_name, temperature=0.0)
    return HallucinationGuard(llm=llm, k=3)


# =====================================================================
# 3. Sidebar Configuration
# =====================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.title("Settings & Filters")
    st.markdown("Configure domain constraints and retrieval parameters.")

    task_filter_selection = st.selectbox(
        "Domain Task Filter:",
        options=["All", "Summary", "QA", "Data2txt"],
        index=0,
        help="Filter the retrieval corpus to specific task modalities from RAGTruth."
    )

    task_type_filter: Optional[str] = None if task_filter_selection == "All" else task_filter_selection

    st.divider()

    st.markdown("### ⚙️ Engine Details")
    st.markdown("- **Dense Store:** ChromaDB (`BAAI/bge-small-en-v1.5`)")
    st.markdown("- **Sparse Store:** BM25 (26,700 Corpus Chunks)")
    st.markdown("- **Fusion Model:** `EnsembleRetriever` (BM25: 0.3, Dense: 0.7)")
    st.markdown("- **Inference Backend:** Ollama (`qwen2.5-coder:14b`)")
    st.markdown("- **Factual Verification:** Zero-Temperature Dual-Stage Judge")

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =====================================================================
# 4. Main Area & Conversation State
# =====================================================================

st.markdown('<div class="main-header">Domain-Specific RAG with Hallucination Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid Semantic Search (BM25 + Dense ChromaDB) with Self-Refinement & Grounding Guard</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Instantiate the pipeline
guard = load_guard_pipeline()

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Display Final Answer
            st.markdown(f"### {msg['content']}")

            # Status Badge
            status = msg.get("status", "VERIFIED_PASS")
            if status == "VERIFIED_PASS":
                st.markdown('<span class="status-badge-pass">✅ Factually Grounded (PASS)</span>', unsafe_allow_html=True)
            elif status == "REFINED_PASS":
                st.markdown('<span class="status-badge-refined">⚠️ Self-Refined & Verified</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge-fallback">🚫 Fallback Triggered (Insufficient Context)</span>', unsafe_allow_html=True)

            # Expander 1: Retrieved Context
            with st.expander("🔍 Retrieved Context (Hybrid Search)", expanded=False):
                raw_docs = msg.get("raw_docs", [])
                if raw_docs:
                    for i, d in enumerate(raw_docs, start=1):
                        meta = d.get("metadata", {})
                        st.markdown(f"""
                        <div class="chunk-card">
                            <b>Chunk {i}</b> | <code>Doc: {meta.get('doc_id', 'N/A')}</code> |
                            Task: <code>{meta.get('task_type', 'N/A')}</code> |
                            Domain: <code>{meta.get('source_domain', 'N/A')}</code><br>
                            <p style="margin-top: 6px; font-size: 0.95rem; color: #334155;">{d.get('page_content', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No documents retrieved.")

            # Expander 2: Judge Audit Trail
            with st.expander("⚖️ Judge Audit Trail", expanded=False):
                audit_trail = msg.get("audit_trail", [])
                for step in audit_trail:
                    attempt = step.get("attempt", 1)
                    step_type = step.get("type", "generation")
                    verdict = step.get("judge_verdict", "N/A")
                    answer = step.get("answer", "")
                    reason = step.get("judge_reason")

                    st.markdown(f"#### Attempt {attempt}: `{step_type}`")
                    st.markdown(f"**Draft Answer:**\n> {answer}")
                    if verdict == "PASS":
                        st.success(f"**Stage 2 Verdict:** `{verdict}`")
                    else:
                        st.error(f"**Stage 2 Verdict:** `{verdict}`")
                        if reason:
                            st.warning(f"**Judge Feedback:** {reason}")
                    st.divider()

# Handle new user input
user_query = st.chat_input("Ask a domain-specific question (e.g., 'What are the operating hours for Apna Indian Kitchen?')...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query with spinner
    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Retrieval & Hallucination Guard..."):
            t_start = time.time()

            # Retrieve context chunks directly to preserve Document objects for UI rendering
            context_str, docs = guard.retrieve_context(user_query, task_type_filter=task_type_filter)

            # Execute guard loop
            result = guard.ask_with_guard(user_query, task_type_filter=task_type_filter)
            elapsed = time.time() - t_start

            final_answer = result.get("final_answer", "")
            status = result.get("status", "VERIFIED_PASS")
            audit_trail = result.get("audit_trail", [])

            # 1. Final Answer Display
            st.markdown(f"### {final_answer}")

            # Status Badge
            if status == "VERIFIED_PASS":
                st.markdown('<span class="status-badge-pass">✅ Factually Grounded (PASS)</span>', unsafe_allow_html=True)
            elif status == "REFINED_PASS":
                st.markdown('<span class="status-badge-refined">⚠️ Self-Refined & Verified</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge-fallback">🚫 Fallback Triggered (Insufficient Context)</span>', unsafe_allow_html=True)

            st.caption(f"⏱️ Total Response Latency: {elapsed:.2f}s | Filter: `{task_filter_selection}`")

            # 2. Expander: Retrieved Context (Hybrid Search)
            with st.expander("🔍 Retrieved Context (Hybrid Search)", expanded=True):
                if docs:
                    for i, d in enumerate(docs, start=1):
                        meta = d.metadata
                        st.markdown(f"""
                        <div class="chunk-card">
                            <b>Chunk {i}</b> | <code>Doc: {meta.get('doc_id', 'N/A')}</code> |
                            Task: <code>{meta.get('task_type', 'N/A')}</code> |
                            Domain: <code>{meta.get('source_domain', 'N/A')}</code><br>
                            <p style="margin-top: 6px; font-size: 0.95rem; color: #334155;">{d.page_content.strip()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No documents retrieved.")

            # 3. Expander: Judge Audit Trail
            with st.expander("⚖️ Judge Audit Trail", expanded=True):
                for step in audit_trail:
                    attempt = step.get("attempt", 1)
                    step_type = step.get("type", "generation")
                    verdict = step.get("judge_verdict", "N/A")
                    answer = step.get("answer", "")
                    reason = step.get("judge_reason")

                    st.markdown(f"#### Attempt {attempt}: `{step_type}`")
                    st.markdown(f"**Draft Answer:**\n> {answer}")
                    if verdict == "PASS":
                        st.success(f"**Stage 2 Verdict:** `{verdict}`")
                    else:
                        st.error(f"**Stage 2 Verdict:** `{verdict}`")
                        if reason:
                            st.warning(f"**Judge Feedback:** {reason}")
                    st.divider()

            # Record assistant response in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "status": status,
                "raw_docs": [{"metadata": d.metadata, "page_content": d.page_content} for d in docs],
                "audit_trail": audit_trail
            })
