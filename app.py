import streamlit as st
import faiss
import json
import numpy as np
import os
import anthropic
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RailYatra Query Engine",
    page_icon="🚆",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — Clean Government/Professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0a3d62 0%, #1a6ca8 60%, #2980b9 100%);
    border-radius: 12px;
    padding: 28px 32px 22px;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(10,61,98,0.18);
    display: flex;
    align-items: center;
    gap: 18px;
}
.hero-icon { font-size: 3.2rem; line-height: 1; }
.hero-text h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px;
    letter-spacing: 0.5px;
}
.hero-text p {
    font-size: 0.93rem;
    color: #b8d4eb;
    margin: 0;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #1a6ca8;
    margin-bottom: 8px;
}

/* ── Answer card ── */
.answer-card {
    background: #f0f7ff;
    border-left: 4px solid #1a6ca8;
    border-radius: 0 10px 10px 0;
    padding: 20px 22px;
    margin-top: 12px;
    font-size: 1rem;
    line-height: 1.7;
    color: #1a2e45;
}

/* ── Source chip ── */
.source-row { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.source-chip {
    background: #e8f2fb;
    border: 1px solid #aad0ef;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #1a6ca8;
    font-weight: 600;
}

/* ── Info badge ── */
.info-badge {
    background: #fff8e1;
    border: 1px solid #ffd54f;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #795548;
    margin-bottom: 16px;
}

/* ── Context expander text ── */
.ctx-block {
    background: #f8f8f8;
    border-left: 3px solid #ccc;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #444;
    margin-bottom: 10px;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a3d62;
}
[data-testid="stSidebar"] * {
    color: #d6eaf8 !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #1a5276;
    border: 1px solid #2980b9;
    color: #ffffff !important;
}

/* ── Button ── */
.stButton > button {
    background: #0a3d62;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1a6ca8;
}

/* ── Sample question pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px; }
.pill {
    background: #e8f2fb;
    border: 1px solid #aad0ef;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #0a3d62;
    cursor: pointer;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR — API Key
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**RailYatra Query Engine** answers passenger queries based strictly on "
        "official Indian Railways documents — Citizen Charter, Refund Rules, "
        "and Passenger Rights."
    )
    st.markdown("---")
    top_k = st.slider("Retrieved chunks (top-k)", min_value=2, max_value=8, value=4)
    st.markdown("---")
    st.markdown(
        "<small style='color:#7fb3d3'>Built with FAISS · Claude AI · Streamlit</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🚆</div>
  <div class="hero-text">
    <h1>RailYatra Query Engine</h1>
    <p>Instant, document-grounded answers from official Indian Railways rules &amp; passenger rights</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading vector store…")
def load_data():
    index = faiss.read_index("data/railway_faiss.index")
    with open("data/railway_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

embed_model = load_embed_model()

try:
    index, metadata = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(
        f"⚠️ Could not load vector store: `{e}`\n\n"
        "Please run the pipeline notebooks first to generate `data/railway_faiss.index` "
        "and `data/railway_metadata.json`, then place them in the `data/` folder."
    )

# ─────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────
def retrieve_context(query: str, top_k: int = 4):
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    chunks, sources = [], []
    for idx in indices[0]:
        chunks.append(metadata[idx]["text"])
        sources.append(metadata[idx].get("source", "Railway Document"))
    return chunks, sources

# ─────────────────────────────────────────────
#  ANSWER GENERATION — Claude API
# ─────────────────────────────────────────────
def generate_answer(query: str, api_key: str, top_k: int = 4):
    chunks, sources = retrieve_context(query, top_k)
    context = "\n\n---\n\n".join(
        [f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)]
    )

    prompt = f"""You are an expert assistant for Indian Railways passenger rules and rights.
Your answers must be based ONLY on the provided context from official Indian Railways documents.
If the answer is not in the context, say: "This information is not covered in the available railway documents."

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, structured, and accurate answer based strictly on the context above.
Use bullet points where helpful. Be concise but complete."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text, chunks, sources

# ─────────────────────────────────────────────
#  SAMPLE QUESTIONS
# ─────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the refund rules if a train is cancelled?",
    "What ID proof is required during train travel?",
    "How do I file a complaint about railway services?",
    "Are senior citizens eligible for ticket concessions?",
    "What facilities must railways provide at stations?",
    "What happens if AC equipment fails during travel?",
]

st.markdown('<div class="section-label">Try a sample question</div>', unsafe_allow_html=True)
cols = st.columns(3)
selected_sample = None
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 3].button(q[:48] + ("…" if len(q) > 48 else ""), key=f"sq_{i}"):
        selected_sample = q

# ─────────────────────────────────────────────
#  QUERY INPUT
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Your question</div>', unsafe_allow_html=True)
default_val = selected_sample if selected_sample else ""
query = st.text_input(
    label="query_input",
    label_visibility="collapsed",
    placeholder="e.g. What are my rights if a train is delayed?",
    value=default_val,
)

ask_btn = st.button("🔍  Get Answer", use_container_width=False)

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
if ask_btn or (selected_sample and selected_sample != st.session_state.get("last_sample")):
    if selected_sample:
        st.session_state["last_sample"] = selected_sample
        query = selected_sample

    if not query.strip():
        st.warning("Please enter a question.")
    elif not api_key_input.strip():
        st.markdown(
            '<div class="info-badge">🔑 Please enter your <b>Anthropic API Key</b> in the sidebar to get AI-powered answers.</div>',
            unsafe_allow_html=True,
        )
    elif not data_loaded:
        st.error("Vector store not loaded. Please check the data files.")
    else:
        with st.spinner("Searching railway documents and generating answer…"):
            try:
                answer, chunks, sources = generate_answer(query, api_key_input, top_k)

                # ── Answer ──
                st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                # ── Sources ──
                unique_sources = list(dict.fromkeys(sources))
                chips = "".join(
                    f'<span class="source-chip">📄 {s}</span>' for s in unique_sources
                )
                st.markdown(
                    f'<div class="source-row"><b style="font-size:0.8rem;color:#666;margin-right:6px;">Sources:</b>{chips}</div>',
                    unsafe_allow_html=True,
                )

                # ── Retrieved context (expandable) ──
                with st.expander("📂 View retrieved document chunks"):
                    for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
                        st.markdown(
                            f'<div class="ctx-block"><b>Chunk {i} — {src}</b><br><br>{chunk[:600]}{"…" if len(chunk) > 600 else ""}</div>',
                            unsafe_allow_html=True,
                        )

            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key. Please check your Anthropic API key in the sidebar.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#aaa'>RailYatra Query Engine · Answers grounded in official Indian Railways documents · "
    "Not an official Indian Railways product.</small>",
    unsafe_allow_html=True,
)
