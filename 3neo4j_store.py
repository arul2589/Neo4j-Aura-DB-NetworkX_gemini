import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# 1. Set your credentials (using your specific Aura instance details)
os.environ["OPENAI_API_KEY"] = "###############"
os.environ["NEO4J_URI"] = "###############"
os.environ["NEO4J_USERNAME"] = "###############"
os.environ["NEO4J_PASSWORD"] = "###############"

# 2. Initialize the Neo4j Graph connection, specifying your database name explicitly
print("Connecting to Neo4j AuraDB...")
graph = Neo4jGraph(database="###############")

# 3. Re-run our text extraction
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
The company is headquartered in San Francisco, California. 
Emil Eifrem, Johan Svensson, and Peter Neubauer founded Neo4j in 2000.
"""
docs = [Document(page_content=sample_text)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(docs)

# 4. Push the graph documents into Neo4j AuraDB
print("Saving graph documents to Neo4j AuraDB...")
graph.add_graph_documents(graph_documents)

print("Success! Data has been written to your Neo4j AuraDB instance.")

# 5. Run a quick verification query
result = graph.query("MATCH (n) RETURN n.id AS node, labels(n) AS type LIMIT 5")
print("\n--- Verification Query Results from AuraDB ---")
for record in result:
    print(record)