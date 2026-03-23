from neo4j import GraphDatabase
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path="data/chromadb")
collection = chroma_client.get_collection("sec_filings")
neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password"))
)

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","TSLA":"Technology",
    "NVDA":"Technology","META":"Technology","ORCL":"Technology","CRM":"Technology",
    "INTC":"Technology","AMD":"Technology",
    "JPM":"Finance","BAC":"Finance","WFC":"Finance","GS":"Finance","MS":"Finance",
    "C":"Finance","BLK":"Finance","AXP":"Finance","USB":"Finance","PNC":"Finance",
    "JNJ":"Healthcare","PFE":"Healthcare","UNH":"Healthcare","CVS":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare",
    "DHR":"Healthcare","BMY":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy",
    "PXD":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy","OXY":"Energy",
    "WMT":"Retail","AMZN":"Retail","HD":"Retail","TGT":"Retail","COST":"Retail",
    "LOW":"Retail","EBAY":"Retail","ETSY":"Retail","KR":"Retail","DG":"Retail"
}

SECTOR_KEYWORDS = {
    "technology": "Technology", "tech": "Technology", "software": "Technology",
    "finance": "Finance", "bank": "Finance", "banking": "Finance", "financial": "Finance",
    "healthcare": "Healthcare", "health": "Healthcare", "pharma": "Healthcare",
    "energy": "Energy", "oil": "Energy", "gas": "Energy",
    "retail": "Retail", "consumer": "Retail"
}

def detect_sector(query):
    query_lower = query.lower()
    for keyword, sector in SECTOR_KEYWORDS.items():
        if keyword in query_lower:
            return sector
    return None

def get_tickers_by_sector(sector):
    return [t for t, s in SECTOR_MAP.items() if s == sector]

def broad_vector_search(query, n_results=8):
    """Search across all companies with no filter."""
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "ticker": meta["ticker"],
            "section": meta["section"],
            "filing_date": meta["filing_date"],
            "similarity": round(1 - dist, 3)
        })
    return chunks

def sector_graph_search(sector, keywords):
    """Search graph for all companies in a sector."""
    tickers = get_tickers_by_sector(sector)
    results = []

    with neo4j_driver.session() as session:
        for ticker in tickers:
            risk_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:REPORTS_RISK]->(r:Risk)
                WHERE ANY(kw IN $keywords WHERE toLower(r.description) CONTAINS kw)
                RETURN r.description AS risk, r.category AS category
                LIMIT 3
            """, ticker=ticker, keywords=keywords)

            risks = [f"{r['category']}: {r['risk']}" for r in risk_result]
            if risks:
                results.append({"ticker": ticker, "risks": risks})

    return results

def multi_company_graph_search(tickers, keywords):
    """Search graph across multiple specific companies."""
    results = []

    with neo4j_driver.session() as session:
        for ticker in tickers:
            risk_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:REPORTS_RISK]->(r:Risk)
                WHERE ANY(kw IN $keywords WHERE toLower(r.description) CONTAINS kw)
                RETURN r.description AS risk, r.category AS category
                LIMIT 5
            """, ticker=ticker, keywords=keywords)

            geo_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:OPERATES_IN]->(g:Geography)
                RETURN g.name AS geo LIMIT 5
            """, ticker=ticker)

            risks = [f"{r['category']}: {r['risk']}" for r in risk_result]
            geos = [r["geo"] for r in geo_result]

            results.append({
                "ticker": ticker,
                "risks": risks,
                "geographies": geos
            })

    return results

def smart_retrieve(query, tickers, keywords):
    """
    Route retrieval strategy based on query type:
    - Single company  → filtered vector + targeted graph
    - Multi company   → per-company vector + multi graph
    - Sector query    → broad vector + sector graph
    - General query   → broad vector + sample graph
    """
    sector = detect_sector(query)

    if len(tickers) == 1:
        # Single company
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"ticker": tickers[0]},
            include=["documents", "metadatas", "distances"]
        )
        chunks = [{
            "text": doc, "ticker": meta["ticker"],
            "section": meta["section"], "filing_date": meta["filing_date"],
            "similarity": round(1 - dist, 3)
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]
        graph_results = multi_company_graph_search(tickers, keywords)
        strategy = "single-company"

    elif len(tickers) > 1:
        # Multi company comparison
        chunks = broad_vector_search(query, n_results=6)
        chunks = [c for c in chunks if c["ticker"] in tickers]
        graph_results = multi_company_graph_search(tickers, keywords)
        strategy = "multi-company"

    elif sector:
        # Sector-wide query
        chunks = broad_vector_search(query, n_results=8)
        sector_tickers = get_tickers_by_sector(sector)
        chunks = [c for c in chunks if c["ticker"] in sector_tickers]
        graph_results = sector_graph_search(sector, keywords)
        strategy = f"sector-{sector}"

    else:
        # General broad query
        chunks = broad_vector_search(query, n_results=8)
        sample_tickers = list(set(c["ticker"] for c in chunks))[:5]
        graph_results = multi_company_graph_search(sample_tickers, keywords)
        strategy = "broad"

    return chunks, graph_results, strategy