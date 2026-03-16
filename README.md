# FinanceGPT — AI-Powered Financial Document Intelligence

> Local RAG + Knowledge Graph system for analyzing SEC 10-K filings using Llama 3.2, ChromaDB, and Neo4j.

---

## What It Does

FinanceGPT lets you ask complex financial questions and get answers grounded in real SEC filings. It combines semantic search over document chunks with graph-based relationship traversal to answer questions that a simple keyword search or plain LLM cannot.

Example queries it can handle:
- "What are Tesla's top supply chain risks?"
- "Which tech companies mention semiconductor shortages?"
- "How have Microsoft's AI-related risks evolved over the last 3 years?"
- "What risks do energy companies have in common?"
- "How could climate regulations affect oil companies' capital expenditures?"

---

## Architecture

```
SEC EDGAR Filings (50+ S&P 500 companies)
            ↓
   Document Parsing & Chunking
            ↓
  ┌─────────────────────────────┐
  │  Vector Search (ChromaDB)   │   ←  Semantic similarity
  │  Knowledge Graph (Neo4j)    │   ←  Entity relationships
  └─────────────────────────────┘
            ↓
     Hybrid Retrieval Layer
            ↓
     Llama 3.2 (via Ollama)
            ↓
       Streamlit UI
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Llama 3.2 via Ollama | Local inference, no API cost |
| Vector DB | ChromaDB | Semantic search over document chunks |
| Graph DB | Neo4j | Entity relationships and multi-hop reasoning |
| NLP | spaCy `en_core_web_lg` | Named entity recognition |
| Orchestration | LangChain | RAG pipeline management |
| UI | Streamlit | Chat interface and graph visualization |
| Embeddings | `all-MiniLM-L6-v2` | Sentence-level embeddings |

---

## Dataset

- **Source:** SEC EDGAR (public domain, free)
- **Coverage:** 50 S&P 500 companies across 5 sectors
- **Filings:** 10-K annual reports, last 3 years (~145 filings)
- **Sections extracted:** Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements)

| Sector | Companies |
|--------|-----------|
| Technology | AAPL, MSFT, GOOGL, TSLA, NVDA, META, ORCL, CRM, INTC, AMD |
| Finance | JPM, BAC, WFC, GS, MS, C, BLK, AXP, USB, PNC |
| Healthcare | JNJ, PFE, UNH, CVS, ABBV, MRK, TMO, ABT, DHR, BMY |
| Energy | XOM, CVX, COP, SLB, EOG, PXD, MPC, PSX, VLO, OXY |
| Retail | WMT, AMZN, HD, TGT, COST, LOW, EBAY, ETSY, KR, DG |

---

## Knowledge Graph Stats

| Element | Count |
|---------|-------|
| Companies | 49 |
| Risk nodes | 1,874 |
| Metric nodes | 23 |
| Geography nodes | 676 |
| REPORTS_RISK relationships | 1,874 |
| HAS_METRIC relationships | 2,546 |
| OPERATES_IN relationships | 1,525 |

---

## Project Structure

```
financegpt/
├── data/
│   ├── sec_filings/        # Raw downloaded 10-K filings
│   ├── parsed/
│   │   ├── filings.json    # Parsed text sections
│   │   └── entities.json   # Extracted NLP entities
│   └── chromadb/           # Persisted vector database
├── download_filings.py     # SEC EDGAR downloader
├── parse_filings.py        # HTML parser and section extractor
├── extract_entities.py     # spaCy NER pipeline
├── populate_graph.py       # Neo4j graph population
├── build_vectordb.py       # ChromaDB embedding pipeline
├── rag_pipeline.py         # LangChain RAG + hybrid retrieval
├── app.py                  # Streamlit UI
├── .env                    # Secrets (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- [Neo4j Desktop](https://neo4j.com/download/) installed

### 1. Clone and install dependencies

```bash
git clone https://github.com/addetsi/FinanceGPT.git
cd FinanceGPT
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Pull the LLM

```bash
ollama pull llama3.2
```

### 3. Set up environment variables

Create a `.env` file:
```
NEO4J_PASSWORD=your_password
SEC_EMAIL=your@email.com
```

### 4. Set up Neo4j

- Open Neo4j Desktop and create a local database named `financegpt-db`
- Start the database
- Run the constraints from `schema.cypher` in Neo4j Browser

### 5. Run the pipeline

```bash
python download_filings.py   # Download SEC filings
python parse_filings.py      # Parse HTML to text
python extract_entities.py   # Extract NLP entities
python populate_graph.py     # Build knowledge graph
python build_vectordb.py     # Generate embeddings
```

### 6. Launch the app

```bash
streamlit run app.py
```

---

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| CPU | Intel i5 / AMD Ryzen 5 (4+ cores) |
| RAM | 16 GB (32 GB recommended) |
| Storage | 20 GB free |
| GPU | Not required |

Expected response time: 10–20 seconds per query on CPU.

---

## Known Limitations

- Llama 3.2 3B may struggle with very complex multi-hop reasoning — upgrading to 8B improves quality
- Entity extraction uses pattern matching; some false positives in organization names
- `COMPETES_WITH` relationships are sparse since companies rarely name competitors by ticker
- Financial statement numbers are not fully extracted (qualitative risk analysis is the primary focus)

---

## Roadmap

- [ ] Streamlit chat UI with graph visualization
- [ ] Azure Container Instances deployment
- [ ] Upgrade to Llama 3.2 8B for better reasoning
- [ ] Add earnings call transcripts as a second data source
- [ ] Fine-tune entity extraction for financial domain
- [ ] Add time-series risk trend analysis

---

## License

MIT License — data sourced from SEC EDGAR (public domain).