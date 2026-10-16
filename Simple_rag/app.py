"""
Podcast Vector RAG - Streamlit Interface
Dense Vector (ChromaDB) + Lexical BM25 + Cross-Encoder Reranking

Run: streamlit run app.py
"""

import os
import re
import math
import uuid
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
ASSETS_DIR = BASE_DIR / "assets"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vector RAG - Podcast Intelligence",
    page_icon=":material/dataset:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Design System (Vector RAG Theme: Indigo & Slate) ──────────────────────
st.markdown("""
<style>
    /* Global Typography & Palette */
    :root {
        --rag-primary: #4f46e5;
        --rag-primary-dark: #3730a3;
        --rag-accent: #818cf8;
        --rag-bg-tint: #f5f3ff;
        --rag-border: #e0e7ff;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #3730a3 0%, #4f46e5 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .system-badge-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }

    .system-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #4f46e5;
        color: #ffffff;
    }

    .system-subbadge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        background: #e0e7ff;
        color: #3730a3;
    }

    .subtitle {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .arch-card {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }

    .chunk-box {
        background: #fbfbfd;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #4f46e5;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1f2937;
        margin-bottom: 0.6rem;
        line-height: 1.5;
    }

    .timestamp-badge {
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .source-badge {
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .score-badge {
        background: #f3f4f6;
        color: #6b7280;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .source-tag {
        display: inline-block;
        background: #4f46e5;
        color: #ffffff;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.74rem;
        font-weight: 500;
        margin: 2px;
    }

    .srt-cue-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    .srt-cue-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .srt-cue-id {
        font-size: 0.75rem;
        font-weight: 700;
        color: #9ca3af;
    }

    .srt-cue-text {
        font-size: 0.92rem;
        color: #111827;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "rag": None,
        "ingested": False,
        "ingest_stats_list": [],
        "active_sources": [],
        "groq_api_key_input": os.getenv("groq_api_key", "") or os.getenv("GROQ_API_KEY", ""),
        "persist_dir": str(Path(tempfile.gettempdir()) / f"rag_store_{uuid.uuid4().hex}"),
        "srt_current_file": None,
        "srt_page_num": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─── SRT Helpers ──────────────────────────────────────────────────────────────
def get_available_srt_files():
    """Discover all .srt transcript files in standard directories."""
    candidates = [
        TRANSCRIPTS_DIR,
        BASE_DIR,
        BASE_DIR.parent / "graph_rag" / "transcripts",
    ]
    discovered = []
    seen = set()
    for directory in candidates:
        if directory.exists() and directory.is_dir():
            for srt_path in sorted(directory.glob("*.srt")):
                if srt_path.name not in seen:
                    seen.add(srt_path.name)
                    discovered.append(srt_path)
    return discovered


def parse_srt(file_path: Path):
    """Parse an .srt file into structured subtitle cues."""
    content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        cue_id = lines[0] if time_index > 0 else str(len(cues) + 1)
        time_line = lines[time_index]
        parts = time_line.split("-->")
        start_ts = parts[0].strip() if len(parts) > 0 else ""
        end_ts = parts[1].strip() if len(parts) > 1 else ""
        text = " ".join(lines[time_index + 1:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            cues.append({
                "id": cue_id,
                "start": start_ts,
                "end": end_ts,
                "text": text,
            })
    return cues


# ─── LLM Caller ───────────────────────────────────────────────────────────────
def call_llm(messages: list, context: str, api_key: str) -> str:
    """Generate an answer using Groq."""
    from groq import Groq

    active_sources = st.session_state.get("active_sources", [])
    num_sources = len(active_sources)
    source_list = ", ".join(active_sources) if active_sources else "unknown"

    if not context or context.strip() == "No relevant excerpts found.":
        return "I could not find relevant excerpts in the transcripts to answer this question."

    if not api_key or not api_key.strip():
        raise ValueError(
            "Groq API key is missing. Add groq_api_key=... to the .env file "
            "or enter your key in the sidebar."
        )

    system_prompt = f"""You are an AI research assistant analyzing podcast transcripts using a Vector RAG system.
Sources currently indexed ({num_sources}): {source_list}.

RULES:
1. ONLY use information explicitly present in the TRANSCRIPT EXCERPTS below.
2. If the excerpts do not contain sufficient evidence, state: "This topic is not covered in the available transcripts."
3. Cite the specific source transcript and approximate timestamp for every claim (e.g. "In [source] around 12:34...").
4. If multiple transcripts discuss the subject, contrast and synthesize their insights.
5. Do NOT guess, hallucinate, or extrapolate beyond the provided text.
6. Maintain an objective, concise, and grounded tone.

TRANSCRIPT EXCERPTS:
{context}"""

    conversation = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)
    ]

    client = Groq(api_key=api_key.strip())
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "system", "content": system_prompt}] + conversation,
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ─── Ingestion Helper ─────────────────────────────────────────────────────────
def run_ingest(file_path: Path, chunk_size: int, overlap: int, top_k: int, top_n: int):
    """Ingest a single transcript file into the PodcastRAG vector index."""
    if st.session_state.rag is None:
        from rag_pipeline import PodcastRAG
        st.session_state.rag = PodcastRAG(
            persist_dir=st.session_state.persist_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k_retrieve=top_k,
            top_n_rerank=top_n,
        )

    stats = st.session_state.rag.ingest(str(file_path))
    st.session_state.ingest_stats_list.append(stats)
    st.session_state.ingested = bool(st.session_state.rag._ingested_sources)
    st.session_state.active_sources = list(st.session_state.rag._ingested_sources.keys())
    return stats


# ─── Sidebar Navigation & Controls ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    current_page = st.radio(
        "Select View",
        [
            "Vector RAG Assistant",
            "SRT Transcripts Explorer",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Configuration")
    st.caption("LLM Provider: Groq API")
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key_input,
        type="password",
        placeholder="Loaded from .env or enter key...",
        help="Uses groq_api_key from .env by default.",
    )
    if api_key_input:
        st.session_state.groq_api_key_input = api_key_input

    st.divider()
    with st.expander("Vector RAG Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size (tokens)", 200, 800, 500, 50)
        overlap = st.slider("Chunk Overlap (tokens)", 20, 150, 80, 10)
        top_k = st.slider("Vector Retrieve Top-K", 5, 30, 20, 5)
        top_n = st.slider("Cross-Encoder Rerank Top-N", 2, 10, 5, 1)

    st.divider()
    st.markdown("### Upload New Transcripts")
    uploaded_files = st.file_uploader(
        "Upload .srt / .txt files",
        type=["srt", "txt"],
        accept_multiple_files=True,
        help="Upload podcast transcript files to index into ChromaDB.",
    )

    if uploaded_files:
        already = set(st.session_state.active_sources)
        pending = [f for f in uploaded_files if Path(f.name).stem not in already]
        btn_label = f"Process {len(pending)} New Transcript(s)" if pending else "All Uploads Loaded"

        if st.button(btn_label, use_container_width=True, type="primary", disabled=not pending):
            progress_bar = st.progress(0.0, text="Initializing ingestion...")
            new_stats = []
            for i, uf in enumerate(pending):
                progress_bar.progress((i + 0.5) / len(pending), text=f"Processing {uf.name}")
                saved_path = TRANSCRIPTS_DIR / Path(uf.name).name
                saved_path.write_bytes(uf.getvalue())
                try:
                    stats = run_ingest(saved_path, chunk_size, overlap, top_k, top_n)
                    new_stats.append(stats)
                except Exception as exc:
                    st.error(f"Error processing {uf.name}: {exc}")

            progress_bar.progress(1.0, text="Ingestion complete")
            st.success(f"{len(new_stats)} transcript(s) added to vector index")
            st.rerun()

    if st.session_state.active_sources:
        st.divider()
        st.markdown("### Active Vector Sources")
        for src in st.session_state.active_sources:
            st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

    if st.session_state.ingest_stats_list:
        st.divider()
        st.markdown("### Vector Index Stats")
        total_chunks = sum(s.get("chunks", 0) for s in st.session_state.ingest_stats_list)
        col_s1, col_s2 = st.columns(2)
        col_s1.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(st.session_state.active_sources)}</div>'
            '<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
        col_s2.markdown(
            f'<div class="stat-box"><div class="stat-number">{total_chunks}</div>'
            '<div class="stat-label">Embed Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    col_btn_a, col_btn_b = st.columns(2)
    if col_btn_a.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col_btn_b.button("Reset Index", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag = None
        st.session_state.ingested = False
        st.session_state.ingest_stats_list = []
        st.session_state.active_sources = []
        st.session_state.persist_dir = str(Path(tempfile.gettempdir()) / f"rag_store_{uuid.uuid4().hex}")
        st.rerun()


# ─── Top Identity Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="system-badge-container">
    <span class="system-badge">Vector RAG</span>
    <span class="system-subbadge">Dense Embeddings + BM25 + Cross-Encoder</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: VECTOR RAG ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
if current_page == "Vector RAG Assistant":
    st.markdown('<div class="main-header">Podcast Vector RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Multi-transcript semantic search with ChromaDB, BM25 sparse scoring, and neural cross-encoder reranking.</div>',
        unsafe_allow_html=True,
    )

    # Architecture Overview Expander with Visual Image
    arch_img_path = ASSETS_DIR / "vector_rag_architecture.jpg"
    with st.expander("Vector RAG Architecture Diagram & Pipeline Overview", expanded=False):
        if arch_img_path.exists():
            st.image(str(arch_img_path), caption="Vector RAG Pipeline: Chunking -> Dense Embeddings + BM25 -> ChromaDB -> Neural Reranker -> Groq LLM", use_container_width=True)
        st.markdown("""
        **How Vector RAG Operates:**
        1. **Chunking**: Transcripts are segmented into token windows with overlapping boundaries.
        2. **Dense & Sparse Indexing**: Each chunk is embedded via `sentence-transformers` and indexed in BM25 for exact keyword hits.
        3. **Hybrid Retrieval**: Queries execute dense similarity search combined with BM25 lexical ranking.
        4. **Cross-Encoder Reranking**: The top candidates are rescored with a cross-encoder before context is fed to the Groq LLM.
        """)

    # When no documents are ingested yet
    if not st.session_state.ingested:
        st.markdown("---")
        st.info("No transcripts currently loaded in the vector index. You can load available SRT transcripts below or upload in the sidebar.")

        available_srts = get_available_srt_files()
        if available_srts:
            st.markdown("#### Available Local Transcripts")
            st.caption("Click to index directly into the Vector RAG pipeline:")
            cols = st.columns(min(len(available_srts), 3))
            for idx, srt_file in enumerate(available_srts):
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(f"**{srt_file.stem}**")
                    st.caption(f"File size: {srt_file.stat().st_size / 1024:.1f} KB")
                    if st.button(f"Index {srt_file.stem}", key=f"quick_index_{idx}", use_container_width=True):
                        with st.spinner(f"Indexing {srt_file.name}..."):
                            run_ingest(srt_file, chunk_size, overlap, top_k, top_n)
                        st.success(f"Indexed {srt_file.name}")
                        st.rerun()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Upload or Select**\n\nChoose an SRT transcript from the library or upload via the sidebar.")
        c2.markdown("**2. Vector Indexing**\n\nDocuments are tokenized, embedded into dense vectors, and stored in ChromaDB.")
        c3.markdown("**3. Grounded Chat**\n\nAsk questions with precise source citations and timestamped playback intervals.")

    else:
        sources = st.session_state.active_sources
        src_str = " · ".join(f"`{s}`" for s in sources)
        st.markdown(f"**Indexed Sources ({len(sources)}):** {src_str}")
        st.markdown("---")

        # Chat Message History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"{len(msg['sources'])} Retrieved Excerpts", expanded=False):
                        for text, meta, score in msg["sources"]:
                            src = meta.get("source", "Transcript")
                            ts = f"{meta.get('start_time', '00:00:00')} → {meta.get('end_time', '00:00:00')}"
                            st.markdown(
                                f'<div class="chunk-box">'
                                f'<span class="source-badge">Source: {src}</span> '
                                f'<span class="timestamp-badge">Time: {ts}</span> '
                                f'<span class="score-badge">Rerank Score: {score:.3f}</span>'
                                f'<br><br>{text}</div>',
                                unsafe_allow_html=True,
                            )

        # Suggested Starters
        if not st.session_state.messages:
            st.markdown("**Suggested Queries:**")
            suggestions = [
                "What are the central themes discussed across all episodes?",
                "How do the guests compare in their perspectives on artificial intelligence?",
                "What was said regarding autonomous systems and self-driving vehicles?",
                "What major technological hurdles were highlighted for the next decade?",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state["_prefill"] = sug
                    st.rerun()

        # Chat Input Bar
        user_input = st.chat_input("Ask any question across the indexed transcripts...")
        if not user_input and "_prefill" in st.session_state:
            user_input = st.session_state.pop("_prefill")

        if user_input:
            api_key = (
                st.session_state.get("groq_api_key_input")
                or os.getenv("groq_api_key")
                or os.getenv("GROQ_API_KEY")
                or ""
            ).strip()

            if not api_key:
                st.error("Please provide your Groq API key in the sidebar or in your .env file.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Executing hybrid vector & BM25 retrieval..."):
                        hits = st.session_state.rag.retrieve(user_input)
                        context = st.session_state.rag.format_context(hits)

                    with st.spinner("Generating grounded response with Groq LLM..."):
                        answer = call_llm(st.session_state.messages, context, api_key)

                    st.markdown(answer)

                    if hits:
                        with st.expander(f"{len(hits)} Source Excerpts", expanded=False):
                            for text, meta, score in hits:
                                src = meta.get("source", "Transcript")
                                ts = f"{meta.get('start_time', '00:00:00')} → {meta.get('end_time', '00:00:00')}"
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="source-badge">Source: {src}</span> '
                                    f'<span class="timestamp-badge">Time: {ts}</span> '
                                    f'<span class="score-badge">Score: {score:.3f}</span>'
                                    f'<br><br>{text}</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": hits,
                    })

                except Exception as exc:
                    error_msg = f"Inference Error ({type(exc).__name__}): {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SRT TRANSCRIPTS EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif current_page == "SRT Transcripts Explorer":
    st.markdown('<div class="main-header">SRT Transcripts Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Inspect, search, and download podcast SRT subtitle files, or index them directly into Vector RAG.</div>',
        unsafe_allow_html=True,
    )

    available_srts = get_available_srt_files()

    if not available_srts:
        st.warning("No .srt transcript files found in the transcripts directory.")
        st.info("Upload .srt files using the uploader below to populate this library.")
    else:
        # Top-level Library Metrics
        total_files = len(available_srts)
        total_kb = sum(f.stat().st_size for f in available_srts) / 1024

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Available SRT Files", total_files)
        m_col2.metric("Total Library Storage", f"{total_kb:.1f} KB")
        m_col3.metric("Indexed in Vector RAG", len(st.session_state.active_sources))

        st.markdown("---")

        # Select Transcript File
        file_names = [f.name for f in available_srts]
        selected_file_name = st.selectbox(
            "Select Transcript to Inspect",
            file_names,
            index=0,
            help="Choose an SRT file to view its subtitles, search content, or load into RAG.",
        )

        selected_path = next(f for f in available_srts if f.name == selected_file_name)
        cues = parse_srt(selected_path)
        source_stem = selected_path.stem
        is_already_ingested = source_stem in st.session_state.active_sources

        # Action Bar for the selected file
        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

        with col_act1:
            if is_already_ingested:
                st.success(f"Status: Already indexed in Vector RAG ({source_stem})")
            else:
                if st.button(f"Index {source_stem} into Vector RAG", type="primary", use_container_width=True):
                    with st.spinner(f"Ingesting {selected_file_name}..."):
                        run_ingest(selected_path, chunk_size, overlap, top_k, top_n)
                    st.success(f"Successfully indexed {selected_file_name}")
                    st.rerun()

        with col_act2:
            st.download_button(
                label="Download SRT",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="text/plain",
                use_container_width=True,
            )

        with col_act3:
            st.caption(f"Total Cues: {len(cues):,}")

        # Search within Transcript
        st.markdown("#### Search Dialogue")
        search_query = st.text_input(
            "Filter dialogue cues by keyword or phrase",
            placeholder="Search keywords (e.g., intelligence, physics, rockets, venture)...",
        )

        filtered_cues = cues
        if search_query.strip():
            query_lower = search_query.strip().lower()
            filtered_cues = [c for c in cues if query_lower in c["text"].lower()]
            st.caption(f"Found {len(filtered_cues)} matching subtitle cues out of {len(cues)} total.")

        # Display Mode Tabs
        tab_timeline, tab_reading, tab_raw = st.tabs([
            "Timeline Cues View",
            "Continuous Reading View",
            "Raw SRT Source",
        ])

        with tab_timeline:
            if not filtered_cues:
                st.info("No subtitle cues match the search query.")
            else:
                cues_per_page = 40
                total_pages = max(1, math.ceil(len(filtered_cues) / cues_per_page))

                p_col1, p_col2 = st.columns([1, 3])
                page_num = p_col1.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    help="Navigate through pages of subtitle cues.",
                )
                p_col2.caption(f"Showing page {page_num} of {total_pages} ({len(filtered_cues)} cues total)")

                start_idx = (page_num - 1) * cues_per_page
                page_cues = filtered_cues[start_idx : start_idx + cues_per_page]

                for cue in page_cues:
                    st.markdown(
                        f'<div class="srt-cue-card">'
                        f'<div class="srt-cue-header">'
                        f'<span class="srt-cue-id">Cue #{cue["id"]}</span>'
                        f'<span class="timestamp-badge">{cue["start"]} → {cue["end"]}</span>'
                        f'</div>'
                        f'<div class="srt-cue-text">{cue["text"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with tab_reading:
            if not filtered_cues:
                st.info("No matching dialogue.")
            else:
                aggregated_text = "\n\n".join(cue["text"] for cue in filtered_cues)
                st.text_area(
                    "Spoken Dialogue (Plain Text)",
                    value=aggregated_text,
                    height=500,
                    help="Clean concatenated speech dialogue from subtitle cues.",
                )

        with tab_raw:
            raw_content = selected_path.read_text(encoding="utf-8-sig", errors="replace")
            st.code(raw_content[:20000] + ("\n... [truncated]" if len(raw_content) > 20000 else ""), language="text")

    st.markdown("---")
    st.markdown("#### Upload Additional SRT Files")
    new_upload = st.file_uploader(
        "Upload a new .srt file to store in the transcripts library",
        type=["srt"],
        key="srt_page_uploader",
    )
    if new_upload:
        dest = TRANSCRIPTS_DIR / new_upload.name
        dest.write_bytes(new_upload.getvalue())
        st.success(f"Saved {new_upload.name} to transcripts directory.")
        st.rerun()
