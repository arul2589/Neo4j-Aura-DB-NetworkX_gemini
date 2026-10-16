import os
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import NetworkxEntityGraph

# 1. Ensure your OpenAI API key is set
os.environ["OPENAI_API_KEY"] = "############"

# 2. Re-run our text extraction
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
The company is headquartered in San Francisco, California. 
Emil Eifrem, Johan Svensson, and Peter Neubauer founded Neo4j in 2000.
"""
docs = [Document(page_content=sample_text)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(docs)

# 3. Initialize the NetworkX Entity Graph
print("Initializing local NetworkX graph...")
net_graph = NetworkxEntityGraph()

# 4. Add the graph documents into NetworkX
# Note: NetworkX graph store in LangChain typically expects nodes and edges. 
# Let's add them directly using NetworkX's underlying graph or LangChain's wrapper.
print("Adding graph documents to NetworkX...")

# We can populate the NetworkxEntityGraph or use NetworkX directly:
for graph_doc in graph_documents:
    for rel in graph_doc.relationships:
        # Add edges and relationships to the underlying NetworkX graph
        net_graph._graph.add_edge(
            rel.source.id, 
            rel.target.id, 
            relation=rel.type
        )

print("Success! Graph stored locally in NetworkX memory.")

# 5. Inspect the NetworkX graph locally
print("\n--- Local Graph Statistics ---")
print(f"Total Nodes: {net_graph._graph.number_of_nodes()}")
print(f"Total Edges: {net_graph._graph.number_of_edges()}")

print("\n--- Nodes in NetworkX ---")
print(list(net_graph._graph.nodes(data=True)))

print("\n--- Edges (Relationships) in NetworkX ---")
for u, v, data in net_graph._graph.edges(data=True):
    print(f"({u}) -[:{data.get('relation')}> ({v})")