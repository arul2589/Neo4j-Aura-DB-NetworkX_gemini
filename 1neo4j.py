from langchain_community.document_loaders import TextLoader

# 1. Create a sample text file for demonstration
sample_text = """
Neo4j is a graph database management system developed by Neo4j, Inc. 
The company is headquartered in San Francisco, California. 
Emil Eifrem, Johan Svensson, and Peter Neubauer founded Neo4j in 2000.
"""

file_path = "sample_knowledge.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(sample_text.strip())

# 2. Use LangChain's TextLoader to load the document
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 3. Inspect the loaded document
print(f"Loaded {len(docs)} document(s).\n")
print("--- Document Content ---")
print(docs[0].page_content)
print("\n--- Metadata ---")
print(docs[0].metadata)