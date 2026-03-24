import streamlit as st
import sqlite3
import json
import uuid
from datetime import datetime
from rag_pipeline import (
    extract_tickers_from_query,
    extract_keywords_from_query,
    format_vector_context,
    format_graph_context,
    prompt,
    llm
)
from hybrid_retrieval import smart_retrieve
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from pyvis.network import Network
import tempfile

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password"))
)

#database setup 

DB_PATH = "data/chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            graph_html TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_conversation(conv_id, title):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (conv_id, title, now, now))
    conn.execute("""
        UPDATE conversations SET updated_at = ? WHERE id = ?
    """, (now, conv_id))
    conn.commit()
    conn.close()

def save_message(conv_id, role, content, sources=None, graph_html=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO messages (conversation_id, role, content, sources, graph_html, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        conv_id, role, content,
        json.dumps(sources) if sources else None,
        graph_html,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def load_conversations():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, title, updated_at FROM conversations
        ORDER BY updated_at DESC
    """).fetchall()
    conn.close()
    return rows

def load_messages(conv_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT role, content, sources, graph_html
        FROM messages WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, (conv_id,)).fetchall()
    conn.close()
    return [{
        "role": r[0],
        "content": r[1],
        "sources": json.loads(r[2]) if r[2] else None,
        "graph_html": r[3]
    } for r in rows]

def delete_conversation(conv_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

def generate_title(question):
    words = question.strip().split()
    title = " ".join(words[:6])
    return title + ("..." if len(words) > 6 else "")

#graph visualisation 

def build_graph(tickers, keywords):
    net = Network(height="400px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.set_options("""
    {
        "physics": {"stabilization": {"iterations": 100}},
        "nodes": {"font": {"size": 12}},
        "edges": {"smooth": {"type": "dynamic"}}
    }
    """)
    added_nodes = set()

    with neo4j_driver.session() as session:
        for ticker in tickers:
            if ticker not in added_nodes:
                net.add_node(ticker, label=ticker, color="#4a90d9",
                           size=25, title=f"Company: {ticker}")
                added_nodes.add(ticker)

            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:REPORTS_RISK]->(r:Risk)
                WHERE ANY(kw IN $keywords WHERE toLower(r.description) CONTAINS kw)
                RETURN r.description AS risk, r.category AS category
                LIMIT 6
            """, ticker=ticker, keywords=keywords)

            for record in result:
                risk_label = record["risk"][:20]
                risk_id = f"{ticker}_{risk_label}"
                color = {
                    "Market": "#e74c3c",
                    "Operational": "#e67e22",
                    "Regulatory": "#9b59b6",
                    "Environmental": "#2ecc71",
                    "Geopolitical": "#e91e63"
                }.get(record["category"], "#95a5a6")

                if risk_id not in added_nodes:
                    net.add_node(risk_id, label=risk_label, color=color,
                               size=15, title=f"{record['category']}: {record['risk']}")
                    added_nodes.add(risk_id)
                net.add_edge(ticker, risk_id, title="REPORTS_RISK")

            geo_result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:OPERATES_IN]->(g:Geography)
                RETURN g.name AS geo LIMIT 4
            """, ticker=ticker)

            for record in geo_result:
                geo_id = f"geo_{record['geo']}"
                if geo_id not in added_nodes:
                    net.add_node(geo_id, label=record["geo"], color="#1abc9c",
                               size=10, title=f"Geography: {record['geo']}")
                    added_nodes.add(geo_id)
                net.add_edge(ticker, geo_id, title="OPERATES_IN")

    return net

#page config 

st.set_page_config(
    page_title="FinanceGPT",
    page_icon="📈",
    layout="wide"
)

init_db()

#session state 

if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

#sidebar 

with st.sidebar:
    st.title("📈 FinanceGPT")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    #past conversations
    conversations = load_conversations()
    if conversations:
        st.subheader("Past conversations")
        for conv_id, title, updated_at in conversations:
            date_str = updated_at[:10]
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"💬 {title}\n_{date_str}_",
                    key=f"conv_{conv_id}",
                    use_container_width=True
                ):
                    st.session_state.current_conv_id = conv_id
                    st.session_state.messages = load_messages(conv_id)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{conv_id}"):
                    delete_conversation(conv_id)
                    if st.session_state.current_conv_id == conv_id:
                        st.session_state.current_conv_id = None
                        st.session_state.messages = []
                    st.rerun()

    st.divider()

    #example queries
    st.subheader("Example queries")
    example_queries = [
        "What are Apple's main supply chain risks?",
        "Compare Microsoft and Google's cybersecurity risks",
        "Which energy companies mention climate risk?",
        "What geopolitical risks does Tesla report?",
        "Which companies are exposed to semiconductor shortages?",
        "What regulatory risks do banks face?",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_query = q
            st.rerun()

    st.divider()
    st.caption("49 companies · 145 filings · 2022–2025")

    show_sources = st.toggle("Show sources", value=True)
    show_graph = st.toggle("Show graph", value=True)

#main chat area 

if not st.session_state.current_conv_id and not st.session_state.messages:
    st.markdown("## Welcome to FinanceGPT")
    st.markdown(
        "Ask any question about SEC 10-K filings from 49 S&P 500 companies. "
        "Powered by Llama 3.2, ChromaDB, and Neo4j knowledge graph."
    )
    st.info("👈 Select a past conversation or start a new one using the sidebar.")

#render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if show_sources and message.get("sources"):
                with st.expander("📄 Source chunks"):
                    for chunk in message["sources"]:
                        st.caption(
                            f"**{chunk['ticker']}** | {chunk['section']} | "
                            f"{chunk['filing_date']} | similarity: {chunk['similarity']}"
                        )
                        st.text(chunk["text"][:300] + "...")
                        st.divider()

            if show_graph and message.get("graph_html"):
                with st.expander("🕸️ Knowledge graph"):
                    st.components.v1.html(message["graph_html"], height=420)

#handle input 

user_input = st.chat_input("Ask about SEC filings...")

if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    #create new conversation if needed
    if not st.session_state.current_conv_id:
        st.session_state.current_conv_id = str(uuid.uuid4())
        title = generate_title(user_input)
        save_conversation(st.session_state.current_conv_id, title)

    #save and display user message
    save_message(st.session_state.current_conv_id, "user", user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    #generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching filings and graph..."):
            tickers = extract_tickers_from_query(user_input)
            keywords = extract_keywords_from_query(user_input)
            vector_chunks, graph_results, strategy = smart_retrieve(
                user_input, tickers, keywords
            )

            vector_context = format_vector_context(vector_chunks)
            graph_context = format_graph_context(graph_results)
            final_prompt = prompt.format(
                vector_context=vector_context,
                graph_context=graph_context,
                question=user_input
            )

            answer_placeholder = st.empty()
            answer = ""
            for chunk in llm.stream(final_prompt):
                answer += chunk
                answer_placeholder.markdown(answer + "|")
            answer_placeholder.markdown(answer)

            #build graph
            graph_html = None
            graph_tickers = tickers if tickers else list(
                set(c["ticker"] for c in vector_chunks)
            )[:4]

            if graph_tickers and show_graph:
                try:
                    net = build_graph(graph_tickers, keywords)
                    with tempfile.NamedTemporaryFile(
                        suffix=".html", delete=False, mode="w"
                    ) as f:
                        net.save_graph(f.name)
                        graph_html = open(f.name).read()
                    with st.expander("🕸️ Knowledge graph"):
                        st.components.v1.html(graph_html, height=420)
                except Exception as e:
                    st.caption(f"Graph unavailable: {e}")

            if show_sources and vector_chunks:
                with st.expander("📄 Source chunks"):
                    for chunk in vector_chunks:
                        st.caption(
                            f"**{chunk['ticker']}** | {chunk['section']} | "
                            f"{chunk['filing_date']} | similarity: {chunk['similarity']}"
                        )
                        st.text(chunk["text"][:300] + "...")
                        st.divider()

    #save assistant message
    save_message(
        st.session_state.current_conv_id,
        "assistant", answer,
        sources=vector_chunks,
        graph_html=graph_html
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": vector_chunks,
        "graph_html": graph_html
    })

    #update conversation timestamp
    save_conversation(st.session_state.current_conv_id, generate_title(
        st.session_state.messages[0]["content"]
    ))

    st.rerun()