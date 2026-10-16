import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

# 0. Set your OpenAI API key (or use os.environ)
os.environ["OPENAI_API_KEY"] = "################"

# 1. Initialize the LLM (temperature=0 ensures deterministic, factual extractions)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Initialize LangChain's Graph Transformer
llm_transformer = LLMGraphTransformer(llm=llm)

# 3. Take the text we loaded in Step 1
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
The company is headquartered in San Francisco, California. 
Emil Eifrem, Johan Svensson, and Peter Neubauer founded Neo4j in 2000.
"""
docs = [Document(page_content=sample_text)]

# 4. Extract nodes (entities) and relationships automatically using the LLM
print("Extracting entities and relationships... Please wait...")
graph_documents = llm_transformer.convert_to_graph_documents(docs)

# 5. Inspect the extracted graph structure
print(f"\nExtracted {len(graph_documents)} graph document(s).\n")

for i, graph_doc in enumerate(graph_documents):
    print(f"--- Nodes (Entities) --- [Total: {len(graph_doc.nodes)}]")
    for node in graph_doc.nodes:
        print(f"Node: {node.id} | Type: {node.type}")
        
    print(f"\n--- Relationships --- [Total: {len(graph_doc.relationships)}]")
    for rel in graph_doc.relationships:
        print(f"({rel.source.id}) -[:{rel.type}]-> ({rel.target.id})")

