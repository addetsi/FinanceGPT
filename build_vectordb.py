import json 
import os 
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

#initialize ChromaDB (persists to disk)
client = PersistentClient(path="data/chromadb")
collection = client.get_or_create_collection(
    name= "sec_filings",
    metadata={"hnsw:space": "cosine"}
)

#load embedding model 
print("loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model Loaded..")

def chunk_text(text, chunk_size=800, overlap=100):
    """split text into overlapping chunks by word count"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk) > 100:
            chunks.append(chunk)
        
        start += chunk_size - overlap

    return chunks


def build_vectordb(filing_path="data/parsed/filings.json"):
    with open(filing_path, 'r', encoding="utf-8") as f:
        filings = json.load(f)

    total_chunks  = 0
    skipped = 0

    for i, filing in enumerate(filings):
        ticker = filing["ticker"]
        filing_date = filing["filing_date"]
        print(f"[{i+1/len(filings)}] Embeding {ticker} - {filing_date}")

        for section_name, text in filing["sections"].items():
            if not text or len(text) < 200:
                continue 

            chunks = chunk_text(text)

            for j, chunk in enumerate(chunks):
                doc_id = f"{ticker}_{filing_date}_{section_name}_{j}"

                existing = collection.get(ids=[doc_id])
                if existing["ids"]:
                    skipped +=1
                    continue

                embedding = embedder.encode(chunk).tolist()

                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "section": section_name,
                        "chunk_index": j
                    }]
                )

                total_chunks += 1

        print(f"  {ticker} - chunks added so far: {total_chunks}")

print("\n--- Semantic Search Test ---")
query = "supply chain risks and semiconductor shortages"
query_embedding = embedder.encode(query).tolist()

results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

for k, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\nResult {k+1}: {meta['ticker']} | {meta['section']} | {meta['filing_date']}")
        print(f"  Similarity : {1 - dist:.3f}")
        print(f"  Preview    : {doc[:150]}...")

if __name__ == "__main__":
    build_vectordb()