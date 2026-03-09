# 🚆 RailYatra Query Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Claude AI](https://img.shields.io/badge/Claude%20AI-Sonnet%204-orange?style=for-the-badge&logo=anthropic&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An end-to-end Retrieval-Augmented Generation (RAG) system for answering Indian Railways passenger rules and rights — powered by Claude AI and official government documents.**

[🚀 Run Locally](#-setup--installation) · [📓 Notebooks](#-pipeline-notebooks) · [🏗️ Architecture](#️-system-architecture) · [📸 UI Preview](#-ui-preview)

</div>

---

## 📌 Overview

**RailYatra Query Engine** answers passenger queries strictly based on official Indian Railways policy documents such as the **Citizen Charter**, **Refund Rules**, and **Passenger Rights** guidelines.

The system uses a **RAG (Retrieval-Augmented Generation)** pipeline to retrieve the most relevant rule sections from a FAISS vector store and generate grounded, hallucination-free answers using **Claude claude-sonnet-4-20250514** (Anthropic).

> ⚠️ This is **not** an official Indian Railways product. All answers are sourced from uploaded official documents for informational purposes only.

---

## 🎯 Problem Statement

Indian Railways passenger rights, refund rules, and service policies are spread across **lengthy, complex PDF documents**. Passengers struggle to locate accurate information quickly. This project enables **policy-aware, document-grounded question answering** — no hallucination, no guessing.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE (Notebooks)                │
│                                                                 │
│  PDF Documents  →  Text Extraction  →  Cleaning & Chunking     │
│       ↓                                                         │
│  Sentence Embeddings (all-MiniLM-L6-v2)                        │
│       ↓                                                         │
│  FAISS Vector Index  +  Metadata JSON                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓  (data/railway_faiss.index)
┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE (app.py)                   │
│                                                                 │
│  User Query  →  Query Embedding  →  FAISS Semantic Search      │
│                                           ↓                    │
│                              Top-K Relevant Chunks             │
│                                           ↓                    │
│                         Prompt Construction with Context       │
│                                           ↓                    │
│                      Claude claude-sonnet-4-20250514 (Anthropic API)  │
│                                           ↓                    │
│                         Grounded, Factual Answer               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📸 UI Preview

### How to Visualize the Interface Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the Streamlit app
streamlit run app.py

# Step 3: Open in browser (auto-opens, or navigate to)
# http://localhost:8501
```

### How to Deploy on Streamlit Cloud (Free, Public URL)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account (`MahaswetaProjects`)
3. Click **"New app"**
4. Select repository: `MahaswetaProjects/RailYatra-Query-Engine`
5. Branch: `main` · Main file: `app.py`
6. Click **"Advanced settings"** → Add secret:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
7. Click **Deploy** — your app gets a public URL like:
   `https://railyatra-query-engine.streamlit.app`

### App Interface Walkthrough

```
┌──────────────────────────────────────────────────────────────────┐
│  SIDEBAR                  │  MAIN AREA                           │
│  ─────────────────        │  ──────────────────────────────────  │
│  ⚙️ Configuration          │  🚆 RailYatra Query Engine           │
│                           │  Instant answers from official        │
│  [Anthropic API Key]      │  Indian Railways documents            │
│  ●●●●●●●●●●●●●●●          │                                      │
│                           │  ── Sample Questions ──              │
│  Top-K Chunks: [4]        │  [Refund rules?] [ID proof?]         │
│  ───────────              │  [File complaint?] [Senior concession]│
│  ℹ️ About                  │                                      │
│  RailYatra answers        │  ── Your Question ──                 │
│  passenger queries        │  ┌──────────────────────────────┐    │
│  from official docs       │  │ Type your question here...   │    │
│                           │  └──────────────────────────────┘    │
│                           │  [🔍 Get Answer]                     │
│                           │                                      │
│                           │  ── Answer ──                        │
│                           │  ┌─────────────────────────────┐    │
│                           │  │ • Clear, structured answer  │    │
│                           │  │   from official documents   │    │
│                           │  └─────────────────────────────┘    │
│                           │  Sources: [CitizenCharter.pdf]       │
│                           │  [📂 View retrieved chunks ▼]        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RailYatra-Query-Engine/
│
├── 📄 app.py                                    ← Main Streamlit application
├── 📄 requirements.txt                          ← Python dependencies
├── 📄 README.md                                 ← Project documentation
│
├── 📂 .streamlit/
│   └── config.toml                              ← UI theme (blue professional)
│
├── 📂 data/
│   ├── railway_faiss.index                      ← FAISS vector index (generated)
│   ├── railway_metadata.json                    ← Chunk text + source metadata
│   └── README.md                                ← How to regenerate data files
│
└── 📂 notebooks/
    ├── Dataset_Creation_from_PDFs.ipynb         ← Step 1: PDF → raw text
    ├── Text_Cleaning_and_Chunking.ipynb         ← Step 2: Clean + chunk text
    ├── Embeddings_and_Vector_Store.ipynb        ← Step 3: Build FAISS index
    └── RAG_Pipeline__Retrieval___Answering_.ipynb ← Step 4: Full pipeline demo
```

---

## 📓 Pipeline Notebooks

Run these **in order** to build the vector store from your PDFs:

| # | Notebook | Input | Output | Description |
|---|----------|-------|--------|-------------|
| 1 | `Dataset_Creation_from_PDFs.ipynb` | PDFs in `Data/PDFs/` | `railway_rules_raw.json` | Extracts text page-by-page from official railway PDFs |
| 2 | `Text_Cleaning_and_Chunking.ipynb` | `railway_rules_raw.json` | `railway_rules_chunks.json` | Cleans whitespace, splits into 350-word overlapping chunks |
| 3 | `Embeddings_and_Vector_Store.ipynb` | `railway_rules_chunks.json` | `railway_faiss.index` + `railway_metadata.json` | Generates embeddings and builds FAISS index |
| 4 | `RAG_Pipeline__Retrieval___Answering_.ipynb` | FAISS index + metadata | — | Full end-to-end pipeline demo + evaluation |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11+
- An [Anthropic API Key](https://console.anthropic.com) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/MahaswetaProjects/RailYatra-Query-Engine.git
cd RailYatra-Query-Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data files

Place your source PDFs in `Data/PDFs/`, then run the notebooks (1 → 2 → 3) to generate:
- `data/railway_faiss.index`
- `data/railway_metadata.json`

### 4. Set your API key (optional — can also enter in the UI sidebar)

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Launch the app

```bash
streamlit run app.py
```

Visit **http://localhost:8501** in your browser.

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| PDF Parsing | `pypdf` | Extract text from official railway PDFs |
| Text Processing | Python (`re`) | Clean and chunk text into segments |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Convert text to semantic vectors |
| Vector Store | `FAISS` (Facebook AI) | Fast similarity search over embeddings |
| LLM | Claude claude-sonnet-4-20250514 (Anthropic) | Generate grounded, structured answers |
| Frontend | `Streamlit` | Interactive web UI |

---

## ✨ Key Features

- ✅ **Document-grounded answers** — strictly from official Indian Railways rules
- ✅ **Claude AI** for high-quality, structured, multi-point answers
- ✅ **Semantic retrieval** via FAISS — finds relevant sections even with paraphrased queries
- ✅ **Source attribution** — shows which documents the answer came from
- ✅ **Retrieved context viewer** — expandable view of the raw chunks used
- ✅ **Sample questions** — one-click exploration of common passenger queries
- ✅ **Adjustable top-k** — control how many chunks are retrieved via sidebar
- ✅ **Professional UI** — government-official blue theme, clean typography

---

## 📋 Sample Questions to Try

| Category | Sample Question |
|----------|----------------|
| Refunds | What are the refund rules if a train is cancelled? |
| ID Proof | What ID proof is required during train travel? |
| Complaints | How do I file a complaint about railway services? |
| Concessions | Are senior citizens eligible for ticket concessions? |
| Facilities | What facilities must railways provide at stations? |
| AC Failure | What happens if AC equipment fails during travel? |

---

## 🔮 Future Enhancements

- [ ] Source-level inline citations in answers
- [ ] Automated RAG evaluation metrics (RAGAS framework)
- [ ] Multilingual query support (Hindi, Odia, Tamil, etc.)
- [ ] Integration with live Indian Railways API for train status
- [ ] Chat history and multi-turn conversation support
- [ ] Mobile-optimised responsive UI

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built with ❤️ using **Claude AI** · **FAISS** · **Streamlit**

</div>
