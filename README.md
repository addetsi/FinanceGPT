# FinanceGPT — AI-Powered Financial Document Intelligence

> A local RAG + Knowledge Graph system for analyzing SEC 10-K filings. Ask complex financial questions and get answers grounded in real filings from 49 S&P 500 companies — powered by Llama 3.2, ChromaDB, and Neo4j.

---

## Demo

![FinanceGPT Chat Interface](assets/screenshots/chat_ui.jpg)

---

## What It Does

FinanceGPT lets you ask complex financial questions and get answers grounded in real SEC filings. It combines semantic search over document chunks with graph-based relationship traversal to answer questions that a simple keyword search or plain LLM cannot handle.

### Example Queries

| Query Type | Example |
|------------|---------|
| Single company risk | "What are Apple's main supply chain risks?" |
| Multi-company comparison | "Compare Microsoft and Google's cybersecurity risks" |
| Sector analysis | "What risks do energy companies have in common?" |
| Geographic exposure | "Which tech companies report China-related risks?" |
| Regulatory focus | "What regulatory risks do banks face?" |
| Causal reasoning | "How could climate regulations affect oil companies?" |

---

## Screenshots

### Chat Interface with Past Conversations
![Chat UI](assets/screenshots/chat_ui.jpg)

### Multi-Company Comparison
![Multi Company Comparison](assets/screenshots/multi_company.jpg)

### Knowledge Graph Visualization
![Knowledge Graph](assets/screenshots/graph_viz.jpg)

### Neo4j Browser — Apple Risk Network
![Neo4j Browser](assets/screenshots/neo4j_browser.jpg)

---

## Architecture

```
SEC EDGAR Filings (49 S&P 500 companies, 145 filings)
                        ↓
          Document Parsing & Chunking
          (BeautifulSoup, section extraction)
                        ↓
         ┌──────────────────────────────┐
         │                              │
  Vector Search                  Knowledge Graph
  (ChromaDB)                     (Neo4j)
  - Sentence embeddings          - Companies, Risks
  - Semantic similarity          - Metrics, Geographies
  - Section metadata             - Relationships
         │                              │
         └──────────── Hybrid Retrieval ┘
                        ↓
              Llama 3.2 (via Ollama)
              Local inference, no API cost
                        ↓
              Streamlit Chat UI
              - Persistent chat history
              - Graph visualization
              - Source citations
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
| Embeddings | `all-MiniLM-L6-v2` | Sentence-level embeddings |
| UI | Streamlit | Chat interface with persistent history |
| Graph Viz | PyVis | Interactive knowledge graph display |

---

## Dataset

- **Source:** SEC EDGAR (public domain, free)
- **Coverage:** 49 S&P 500 companies across 5 sectors
- **Filings:** 10-K annual reports, 2022–2025 (~145 filings)
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

## Setup Instructions

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- [Neo4j Desktop](https://neo4j.com/download/) installed

### 1. Clone and install dependencies

```bash
git clone https://github.com/addetsi/FinanceGPT
cd financeGPT
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

Create a `.env` file in the project root:

```
NEO4J_PASSWORD=your_password
SEC_EMAIL=your@email.com
```

### 4. Set up Neo4j

- Open Neo4j Desktop and create a local database named `financegpt-db`
- Set password to match your `.env` file
- Start the database
- In Neo4j Browser run:

```cypher
CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE;
CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (r:Risk) REQUIRE r.risk_id IS UNIQUE;
CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector);
CREATE INDEX risk_category IF NOT EXISTS FOR (r:Risk) ON (r.category);
```

### 5. Run the pipeline (in order)

```bash
python download_filings.py      
filings
python parse_filings.py         
sections
python extract_entities.py      
python populate_graph.py       
graph
python build_vectordb.py        
embeddings
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

### Hybrid Retrieval

When you ask a question, FinanceGPT automatically detects the query type and routes it:

| Query Type | Strategy |
|------------|----------|
| Single company | Filtered vector search + targeted graph lookup |
| Multi-company | Cross-company vector search + parallel graph queries |
| Sector query | Sector-filtered vector + sector-wide graph traversal |
| General query | Broad semantic search + sampled graph context |

### Knowledge Graph

The graph enables relationship-based reasoning that pure vector search cannot do:

```cypher
-- Find all risks for a company by category
MATCH (c:Company {ticker: "AAPL"})-[:REPORTS_RISK]->(r:Risk)
RETURN c.ticker, r.description, r.category

-- Find companies sharing the same risk type
MATCH (c:Company)-[:REPORTS_RISK]->(r:Risk {category: "Geopolitical"})
RETURN c.ticker, r.description
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | Intel i7 / AMD Ryzen 7 |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 30 GB free |
| GPU | Not required | Not required |

Expected response time: 15–30 seconds per query on CPU.

---

## Known Limitations

- Llama 3.2 3B may struggle with very complex multi-hop reasoning — upgrading to 8B improves quality
- Entity extraction uses pattern matching; some false positives in organization names
- `COMPETES_WITH` relationships are sparse since companies rarely name competitors by ticker in filings
- Financial statement numbers are not fully extracted — qualitative risk analysis is the primary focus

---


This project demonstrates:

- **RAG systems** — combining retrieval with generation to ground LLM answers in real data
- **Knowledge graphs** — using Neo4j for relationship-based reasoning beyond what vector search can do
- **Local LLMs** — running Llama 3.2 via Ollama with no API costs and full data privacy
- **Finance domain expertise** — working with SEC filings, financial terminology, and risk analysis
- **Full-stack AI** — end-to-end pipeline from raw data to a production-ready UI

---
