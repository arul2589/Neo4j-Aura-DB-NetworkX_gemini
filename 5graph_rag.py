import os
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import NetworkxEntityGraph

# 1. Ensure OpenAI API key is set
os.environ["OPENAI_API_KEY"] = "###############"

# 2. Recreate our graph (Steps 1-4 combined into a quick setup)
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
The company is headquartered in San Francisco, California. 
Emil Eifrem, Johan Svensson, and Peter Neubauer founded Neo4j in 2000.
"""
docs = [Document(page_content=sample_text)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(docs)

# Populate NetworkX graph
net_graph = NetworkxEntityGraph()
for graph_doc in graph_documents:
    for rel in graph_doc.relationships:
        net_graph._graph.add_edge(
            rel.source.id, 
            rel.target.id, 
            relation=rel.type
        )

# 3. Define a user question
user_query = "Who founded Neo4j and in what year?"
print(f"User Question: {user_query}\n")

# 4. Graph Retrieval: Extract relevant subgraphs or neighbors based on the query keywords
# For a simple approach, let's find matching nodes and pull their direct relationships (1-hop neighborhood)
relevant_context = []
query_keywords = ["Neo4j", "Emil Eifrem", "Johan Svensson", "Peter Neubauer", "2000"] # In real apps, an LLM or keyword extractor finds these

for node in net_graph._graph.nodes():
    # If any keyword matches a node name (case-insensitive check)
    if any(keyword.lower() in node.lower() for keyword in ["Neo4j", "Emil"]):
        # Get outgoing and incoming neighbors
        neighbors = list(net_graph._graph.neighbors(node))
        for neighbor in neighbors:
            edge_data = net_graph._graph.get_edge_data(node, neighbor)
            relation = edge_data.get("relation") if edge_data else "RELATED_TO"
            relevant_context.append(f"({node}) -[:{relation}]-> ({neighbor})")

# Format context nicely for the LLM
context_string = "\n".join(relevant_context)
print("--- Retrieved Graph Context ---")
print(context_string)
print("-" * 30)

# 5. Generation: Pass the graph context and question to the LLM
prompt = f"""
You are an AI assistant. Answer the user's question using ONLY the provided Knowledge Graph context below.

Knowledge Graph Context:
{context_string}

User Question: {user_query}
Answer:
"""

response = llm.invoke(prompt)
print("\n--- LLM Response (Graph RAG) ---")
print(response.content)