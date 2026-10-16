import os
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# 1. Set your credentials
os.environ["OPENAI_API_KEY"] = "###############"
os.environ["NEO4J_URI"] = "###############"
os.environ["NEO4J_USERNAME"] = "###############"
os.environ["NEO4J_PASSWORD"] = "###############"
# 2. Connect to Neo4j AuraDB
print("Connecting to Neo4j AuraDB for RAG...")
graph = Neo4jGraph(database="###############")
graph.refresh_schema()

# 3. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Define a custom Cypher generation prompt with case-insensitivity instructions
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
- Use only the provided nodes, relationships, and properties in the schema.
- Always use case-insensitive matching for string property values (e.g., use `toLower(n.id) = toLower('neo4j')` or `toLower(n.id) CONTAINS 'neo4j'`) to avoid casing mismatches.
- Return only the executable Cypher query.

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# 5. Create the Graph Cypher QA Chain with our custom prompt
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True 
)

# 6. Ask the question again!
user_query = "Who founded Neo4j and who are all the founders?"
print(f"\nUser Question: {user_query}\n")

response = chain.invoke({"query": user_query})

print("\n--- Final Graph RAG Answer ---")
print(response["result"])