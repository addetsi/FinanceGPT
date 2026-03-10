from neo4j import GraphDatabase
from dotenv import load_dotenv 
import os

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    result = session.run("MATCH (c:Company) RETURN c.ticker AS ticker, c.name AS name")
    for record in result:
        print(f"Found company: {record['ticker']} - {record['name']}")

driver.close()
print("Neo4j connection successful!")
