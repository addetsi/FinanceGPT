import json
import hashlib
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))

driver = GraphDatabase.driver(URI, auth=AUTH)

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

RISK_CATEGORIES = {
    "cybersecurity":"Operational", "cyber":"Operational", "breach":"Operational",
    "ransomware":"Operational", "competition":"Market", "war":"Geopolitical",
    "regulatory":"Regulatory", "compliance":"Regulatory", "litigation":"Regulatory",
    "climate":"Environmental", "environmental":"Environmental", "carbon":"Environmental",
    "inflation":"Market", "rate":"Market", "currency":"Market", "forex":"Market",
    "supply":"Operational", "chain":"Operational", "tariff":"Geopolitical",
    "sanction":"Geopolitical", "geopolit":"Geopolitical"
}


def make_risk_id(ticker, risk_text, filing_date):
    key = f"{ticker}_{risk_text}_{filing_date}"
    
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_risk_category(risk_text):
    for keyword, category in RISK_CATEGORIES.items():
        if keyword in risk_text.lower():
            return category 
    
    return "General"

def populate(tx, filing, entities):
    ticker = filing["ticker"]
    filing_date = filing["filing_date"]
    sector = SECTOR_MAP.get(ticker, "Unknown")

    #create company node
    tx.run("""
        MERGE (c:Company {ticker: $ticker})
        SET c.sector = $sector 
    """, ticker=ticker, sector=sector)

    for section_name, ents in entities.items():
        #risk nodes

        for risk_text in ents.get("risks", []):
            risk_id = make_risk_id (ticker, risk_text, filing_date)
            category = get_risk_category(risk_text)

            tx.run("""
                MERGE (r:Risk {risk_id: $risk_id}) 
                SET r.description = $description,
                    r.category = $category,
                    r.filing_date = $filing_date
                WITH r
                MATCH (c:Company {ticker: $ticker})
                MERGE (c)-[:REPORTS_RISK]->(r)
            """, risk_id=risk_id, description=risk_text,
                category=category, filing_date=filing_date, ticker=ticker)
            
            #metric nodes 
            for metric_text in ents.get("metrics", []):
                tx.run("""
                MERGE (m:Metric {metric_name: $metric_name})
                WITH m
                MATCH (c:Company {ticker: $ticker})
                MERGE (c)-[:HAS_METRIC {section: $section, filing_date: $filing_date}]->(m)
            """, metric_name=metric_text, ticker=ticker,
                 section=section_name, filing_date=filing_date)
                
            #geography nodes
            for loc in ents.get("locations", []):
                if len(loc) > 2: #skip abbreviations 
                    tx.run("""
                    MERGE (g:Geography {name: $name})
                    WITH g
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:OPERATES_IN]->(g)
                """, name=loc, ticker=ticker)
                    
        #competitor relationships (companies mentioning other S&P500 tickers)
        known_tickers = set(SECTOR_MAP.keys())
        for org in ents.get("organizations", []):
            org_upper = org.upper().strip()
            if org_upper in known_tickers and org_upper != ticker:
                tx.run("""
                    MATCH (c1:Company {ticker: $ticker})
                    MATCH (c2:Company {ticker: $org})
                    MERGE (c1)-[:COMPETES_WITH]->(c2)
                """, ticker=ticker, org=org_upper)

def main():
    with open("data/parsed/entities.json", "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    with open("data/parsed/filings.json", "r", encoding="utf-8") as f:
        all_filings = json.load(f)


    print(f"Populating graph with {len(all_entities)} filings...\n")

    with driver.session() as session:
        for i, entity_doc in enumerate(all_entities):
            ticker = entity_doc["ticker"]
            filing_date = entity_doc["filings_date"]
            print(f"[{i+1}/{len(all_entities)}] {ticker} - {filing_date}")

            session.execute_write(
                populate,
                {"ticker": ticker, "filing_date": filing_date},
                entity_doc['entities']
            )
        
        print("\n Graph Pupulated...\n")


    with driver.session() as session:
            counts = {
            "Companies":     session.run("MATCH (c:Company) RETURN count(c) AS n").single()["n"],
            "Risks":         session.run("MATCH (r:Risk) RETURN count(r) AS n").single()["n"],
            "Metrics":       session.run("MATCH (m:Metric) RETURN count(m) AS n").single()["n"],
            "Geographies":   session.run("MATCH (g:Geography) RETURN count(g) AS n").single()["n"],
            "REPORTS_RISK":  session.run("MATCH ()-[r:REPORTS_RISK]->() RETURN count(r) AS n").single()["n"],
            "HAS_METRIC":    session.run("MATCH ()-[r:HAS_METRIC]->() RETURN count(r) AS n").single()["n"],
            "OPERATES_IN":   session.run("MATCH ()-[r:OPERATES_IN]->() RETURN count(r) AS n").single()["n"],
            "COMPETES_WITH": session.run("MATCH ()-[r:COMPETES_WITH]->() RETURN count(r) AS n").single()["n"],
                }

    print("--- Graph Summary ---")
    for label, count in counts.items():
        print(f"{label}: {count}")

    driver.close()

if __name__ == "__main__":
    main()


