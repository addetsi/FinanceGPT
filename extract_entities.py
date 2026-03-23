import json 
import spacy
from spacy.matcher import Matcher
import re 
import logging

logger = logging.getLogger(__name__)

#load spacy model
nlp = spacy.load("en_core_web_lg")

def build_matcher(nlp):
    matcher = Matcher(nlp.vocab)

    #financial metrics patterns 
    metric_patterns = [
        [{"LOWER": {"IN": ["revenue", "sales", "income", "ebitda", "margin",
                           "earnings", "profit", "loss", "cashflow", "debt",
                           "capex", "eps", "roe", "roa"]}}],
        [{"LOWER": "net"}, {"LOWER": {"IN": ["income", "loss", "revenue", "sales"]}}],
        [{"LOWER": "gross"}, {"LOWER": {"IN": ["margin", "profit", "loss"]}}],
        [{"LOWER": "operating"}, {"LOWER": {"IN": ["income", "loss", "margin", "expenses"]}}],
    ]

    matcher.add("METRIC", metric_patterns)

    #risk category patterns 
    risk_patterns = [
        [{"LOWER": {"IN": ["cybersecurity", "cyber", "ransomware", "breach", "hack"]}}],
        [{"LOWER": {"IN": ["regulatory", "compliance", "litigation", "lawsuit"]}}],
        [{"LOWER": {"IN": ["supply", "chain"]}, "OP": "+"}, {"LOWER": {"IN": ["risk", "disruption", "shortage"]}}],
        [{"LOWER": {"IN": ["inflation", "interest", "rate", "currency", "forex"]}}],
        [{"LOWER": {"IN": ["climate", "environmental", "esg", "carbon"]}}],
        [{"LOWER": "competition"}, {"LOWER": {"IN": ["risk", "pressure", "threat"]}, "OP": "?"}],
        [{"LOWER": {"IN": ["geopolit", "sanction", "tariff", "trade", "war"]}}],
    ]

    matcher.add("RISK", risk_patterns)

    return matcher


def extract_entities(text, nlp, mathcer):
    """Extract all entities from a text chunk"""

    #limit text length for spacy processing 
    text = text[:50000]
    doc = nlp(text)

    entities = {
        "organizations": [],
        "money": [],
        "dates": [],
        "locations": [],
        "metrics": [],
        "risks": []
    }

    #standard NER entities 
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["organizations"].append(ent.text.strip())
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text.strip())
        elif ent.label_ == "DATE":
            entities['dates'].append(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            entities["locations"].append(ent.text.strip())
        
    #custom matcher entities 
    matches = mathcer(doc)

    for match_id, start, end in matches:
        span_text = doc[start:end].text.strip().lower()
        label = nlp.vocab.strings[match_id]
        if label == "METRIC":
            entities["metrics"].append(span_text)
        elif label == "RISK":
            entities["risks"].append(span_text)

    #duplicates 
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities


def process_filings(input_path="data/parsed/filings.json",
                     output_path="data/parsed/entities.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        filings = json.load(f)

    matcher = build_matcher(nlp)

    results = []

    for i, filing in enumerate(filings):
        ticker = filing['ticker']
        date = filing['filing_date']
        print(f"[{i+1}/{len(filings)}] Extracted entities: {ticker}")

        filing_entities = {}

        for section_name, text in filing["sections"].items():
            if not text or len(text) < 100:
                continue
            filing_entities[section_name] = extract_entities(text, nlp, matcher)


        results.append({
            "ticker": ticker,
            "filings_date": date,
            "entities": filing_entities
        })

        #preview 
        if i == 0:
            print("\n--sample of entities from fillings---")
            for section, ents in filing_entities.items():
                print(f"\n  {section}:")
                print(f"    Organizations : {ents['organizations'][:5]}")
                print(f"    Money values  : {ents['money'][:5]}")
                print(f"    Metrics       : {ents['metrics'][:5]}")
                print(f"    Risks         : {ents['risks'][:5]}")
                print(f"    Locations     : {ents['locations'][:5]}")
            print("-" * 40)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Entities saved to {output_path}")

    return results


if __name__ == "__main__":
    logger.info("Starting sec filings extraction")
    process_filings()

                  
