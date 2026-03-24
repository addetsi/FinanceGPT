from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from hybrid_retrieval import smart_retrieve

load_dotenv()

#initialise components 

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model="llama3.2", temperature=0.1, base_url=OLLAMA_URL)

chroma_client = PersistentClient(path="data/chromadb")
collection = chroma_client.get_collection("sec_filings")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password"))
)

#prompt template 

PROMPT_TEMPLATE = """You are a financial analyst assistant specializing in SEC filings analysis.
Answer the question using ONLY the context provided below.
Always refer to companies by their full name, not just their ticker symbol.
Always cite which company and filing section your answer comes from.
If the context does not contain enough information, say so clearly.

TICKER TO NAME REFERENCE:
AAPL=Apple, MSFT=Microsoft, GOOGL=Google/Alphabet, TSLA=Tesla, NVDA=NVIDIA,
META=Meta/Facebook, ORCL=Oracle, CRM=Salesforce, INTC=Intel, AMD=AMD,
JPM=JPMorgan Chase, BAC=Bank of America, WFC=Wells Fargo, GS=Goldman Sachs,
MS=Morgan Stanley, C=Citigroup, BLK=BlackRock, AXP=American Express,
JNJ=Johnson & Johnson, PFE=Pfizer, UNH=UnitedHealth, CVS=CVS Health,
ABBV=AbbVie, MRK=Merck, XOM=ExxonMobil, CVX=Chevron, COP=ConocoPhillips,
WMT=Walmart, AMZN=Amazon, HD=Home Depot, TGT=Target, COST=Costco

CONTEXT FROM SEC FILINGS:
{vector_context}

KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context}

QUESTION: {question}

ANSWER (use company names not tickers, be specific, use bullet points. Do NOT include source citations inline — sources are shown separately):"""
prompt = PromptTemplate(
    input_variables=["vector_context", "graph_context", "question"],
    template=PROMPT_TEMPLATE
)

#vector retrieval 

def vector_search(query, n_results=5, ticker_filter=None):
    query_embedding = embedder.encode(query).tolist()

    where_filter = {"ticker": ticker_filter} if ticker_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
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

#graph retrieval

def graph_search(tickers, topic_keywords):
    results = []

    with neo4j_driver.session() as session:
        for ticker in tickers:

            #get risks for this company
            risk_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:REPORTS_RISK]->(r:Risk)
                WHERE ANY(kw IN $keywords WHERE toLower(r.description) CONTAINS kw)
                RETURN r.description AS risk, r.category AS category
                LIMIT 5
            """, ticker=ticker, keywords=topic_keywords)

            risks = [f"{r['category']}: {r['risk']}" for r in risk_result]

            #get geographies
            geo_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:OPERATES_IN]->(g:Geography)
                RETURN g.name AS geo
                LIMIT 5
            """, ticker=ticker)

            geos = [r["geo"] for r in geo_result]

            #get metrics
            metric_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_METRIC]->(m:Metric)
                RETURN DISTINCT m.metric_name AS metric
                LIMIT 5
            """, ticker=ticker)

            metrics = [r["metric"] for r in metric_result]

            if risks or geos or metrics:
                results.append({
                    "ticker": ticker,
                    "risks": risks,
                    "geographies": geos,
                    "metrics": metrics
                })

    return results

#entity extraction from query

KNOWN_TICKERS = {
    "AAPL","MSFT","GOOGL","TSLA","NVDA","META","ORCL","CRM","INTC","AMD",
    "JPM","BAC","WFC","GS","MS","C","BLK","AXP","USB","PNC",
    "JNJ","PFE","UNH","CVS","ABBV","MRK","TMO","ABT","DHR","BMY",
    "XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX","VLO","OXY",
    "WMT","AMZN","HD","TGT","COST","LOW","EBAY","ETSY","KR","DG"
}

COMPANY_NAME_MAP = {
    #technology
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro": "AMD",

    #finance
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "chase": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "goldman": "GS",
    "morgan stanley": "MS",
    "citigroup": "C",
    "citi": "C",
    "blackrock": "BLK",
    "american express": "AXP",
    "amex": "AXP",
    "us bancorp": "USB",
    "pnc": "PNC",

    #healthcare
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "cvs": "CVS",
    "abbvie": "ABBV",
    "merck": "MRK",
    "thermo fisher": "TMO",
    "abbott": "ABT",
    "danaher": "DHR",
    "bristol myers": "BMY",
    "bristol-myers": "BMY",

    #energy
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "conoco": "COP",
    "schlumberger": "SLB",
    "eog resources": "EOG",
    "pioneer": "PXD",
    "marathon": "MPC",
    "phillips 66": "PSX",
    "valero": "VLO",
    "occidental": "OXY",

    #retail
    "walmart": "WMT",
    "amazon": "AMZN",
    "home depot": "HD",
    "target": "TGT",
    "costco": "COST",
    "lowe": "LOW",
    "lowes": "LOW",
    "ebay": "EBAY",
    "etsy": "ETSY",
    "kroger": "KR",
    "dollar general": "DG",
}

def extract_tickers_from_query(query):
    query_lower = query.lower()
    found = set()

    #match ticker symbols directly
    for word in query.upper().split():
        clean = word.strip("?.,!")
        if clean in KNOWN_TICKERS:
            found.add(clean)

    #match company names
    for name, ticker in COMPANY_NAME_MAP.items():
        if name in query_lower:
            found.add(ticker)

    return list(found)

def extract_keywords_from_query(query):
    stop_words = {"what","are","the","how","do","does","did","which","company",
                  "companies","their","its","in","of","a","an","and","or","for",
                  "to","is","was","were","have","has","had","with","about","tell",
                  "me","us","show","give","find","list","compare","between"}
    words = query.lower().split()
    return [w.strip("?.,!") for w in words if w not in stop_words and len(w) > 3]

#format context for LLM 

def format_vector_context(chunks):
    if not chunks:
        return "No relevant document chunks found."
    lines = []
    for c in chunks:
        lines.append(
            f"[{c['ticker']} | {c['section']} | {c['filing_date']} | "
            f"similarity: {c['similarity']}]\n{c['text'][:500]}"
        )
    return "\n\n---\n\n".join(lines)

def format_graph_context(graph_results):
    if not graph_results:
        return "No graph relationships found."
    lines = []
    for r in graph_results:
        lines.append(
            f"{r['ticker']}:\n"
            f"  Risks: {', '.join(r['risks'][:3]) or 'none'}\n"
            f"  Geographies: {', '.join(r['geographies'][:3]) or 'none'}\n"
            f"  Metrics: {', '.join(r.get('metrics', [])[:3]) or 'none'}"
        )
    return "\n\n".join(lines)

def query_financegpt(question, verbose=False):
    print(f"\nQuery: {question}")
    print("-" * 60)

    tickers = extract_tickers_from_query(question)
    keywords = extract_keywords_from_query(question)

    if verbose:
        print(f"Detected tickers  : {tickers}")
        print(f"Detected keywords : {keywords}")

    #use smart hybrid retrieval
    vector_chunks, graph_results, strategy = smart_retrieve(question, tickers, keywords)

    if verbose:
        print(f"Retrieval strategy: {strategy}")
        print(f"Vector chunks     : {len(vector_chunks)}")
        print(f"Graph results     : {len(graph_results)}")
        print("\nGenerating answer...")

    vector_context = format_vector_context(vector_chunks)
    graph_context = format_graph_context(graph_results)

    final_prompt = prompt.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question
    )

    answer = llm.invoke(final_prompt)
    print(f"\nAnswer:\n{answer}")
    return answer, vector_chunks, graph_results

#test queries 

if __name__ == "__main__":
    test_questions = [
        "What are Apple's main supply chain risks?",
        "What cybersecurity risks does Microsoft report?",
        "Which companies mention climate risk in their filings?",
    ]

    for question in test_questions:
        query_financegpt(question, verbose=True)
        print("\n" + "=" * 60 + "\n")