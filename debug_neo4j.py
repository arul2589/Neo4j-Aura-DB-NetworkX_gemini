import os
from langchain_neo4j import Neo4jGraph


os.environ["NEO4J_URI"] = "###############"
os.environ["NEO4J_USERNAME"] = "###############"
os.environ["NEO4J_PASSWORD"] = "###############"

graph = Neo4jGraph(database="###############")

# Let's print out all nodes and relationships stored in AuraDB
print("--- All Nodes ---")
print(graph.query("MATCH (n) RETURN labels(n) AS label, n.id AS id"))

print("\n--- All Relationships ---")
print(graph.query("MATCH (n)-[r]->(m) RETURN n.id AS source, type(r) AS rel, m.id AS target"))