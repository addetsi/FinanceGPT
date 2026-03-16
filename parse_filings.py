import os
import json
import re
from bs4 import BeautifulSoup

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def extract_section(text, start_patterns, end_patterns):
    text_lower = text.lower()
    
    start_idx = -1
    for pattern in start_patterns:
        idx = text_lower.find(pattern.lower())
        if idx != -1:
            start_idx = idx
            break
    
    if start_idx == -1:
        return ""
    
    end_idx = len(text)
    for pattern in end_patterns:
        idx = text_lower.find(pattern.lower(), start_idx + 100)
        if idx != -1:
            end_idx = min(end_idx, idx)
            break
    
    return text[start_idx:end_idx]

def extract_html_from_submission(filepath):
    """Extract the main HTML document embedded inside full-submission.txt"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    
    doc_pattern = re.compile(
        r'<DOCUMENT>(.*?)</DOCUMENT>', 
        re.DOTALL | re.IGNORECASE
    )
    
    for match in doc_pattern.finditer(content):
        doc = match.group(1)
        # Check if this is the main 10-K document
        type_match = re.search(r'<TYPE>([^\n]+)', doc, re.IGNORECASE)
        if type_match and '10-K' in type_match.group(1).upper():
            # Extract the actual HTML/text content
            text_match = re.search(r'<TEXT>(.*?)(?:</TEXT>|$)', doc, re.DOTALL | re.IGNORECASE)
            if text_match:
                return text_match.group(1)
    
    #return full content if no document tags found
    return content

def parse_filing(filepath):
    """Parse a single 10-K filing and extract key sections."""
    raw = extract_html_from_submission(filepath)
    
    # Parse as HTML if it contains HTML tags
    if '<html' in raw.lower() or '<body' in raw.lower():
        soup = BeautifulSoup(raw, 'lxml')
        for table in soup.find_all('table'):
            table.decompose()
        text = soup.get_text(separator=' ')
    else:
        text = raw
    
    text = clean_text(text)
    
    sections = {
        "risk_factors": extract_section(
            text,
            start_patterns=["item 1a", "risk factors"],
            end_patterns=["item 1b", "item 2", "unresolved staff comments"]
        ),
        "mda": extract_section(
            text,
            start_patterns=["item 7", "management s discussion", "management's discussion"],
            end_patterns=["item 7a", "item 8", "quantitative and qualitative"]
        ),
        "financial_statements": extract_section(
            text,
            start_patterns=["item 8", "financial statements"],
            end_patterns=["item 9", "changes in and disagreements"]
        )
    }
    
    return sections

def process_all_filings(base_dir="data/sec_filings/sec-edgar-filings"):
    results = []
    
    for ticker in os.listdir(base_dir):
        ticker_path = os.path.join(base_dir, ticker)
        if not os.path.isdir(ticker_path):
            continue
            
        filing_type_path = os.path.join(ticker_path, "10-K")
        if not os.path.exists(filing_type_path):
            continue
        
        for filing_date in os.listdir(filing_type_path):
            filing_path = os.path.join(filing_type_path, filing_date)
            if not os.path.isdir(filing_path):
                continue
            
            # Look for full-submission.txt
            submission_file = os.path.join(filing_path, "full-submission.txt")
            if not os.path.exists(submission_file):
                continue
            
            print(f"Parsing {ticker} - {filing_date}...")
            
            try:
                sections = parse_filing(submission_file)
                
                if any(len(v) > 500 for v in sections.values()):
                    results.append({
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "filepath": submission_file,
                        "sections": sections
                    })
                    print(f"  Done {ticker} - Risk factors: {len(sections['risk_factors'])} chars")
                else:
                    print(f"  Failed  {ticker} - Sections too short")
                    
            except Exception as e:
                print(f"  Failed {ticker}: {e}")
    
    os.makedirs("data/parsed", exist_ok=True)
    output_path = "data/parsed/filings.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Done! Parsed {len(results)} filings saved to {output_path}")
    return results

if __name__ == "__main__":
    results = process_all_filings()
    
    print("\n--- Quality Check ---")
    for r in results[:3]:
        print(f"\n{r['ticker']} ({r['filing_date']}):")
        for section, text in r['sections'].items():
            print(f"  {section}: {len(text)} characters")
        print(f"  Risk factors preview: {r['sections']['risk_factors'][:200]}...")