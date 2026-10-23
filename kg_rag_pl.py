import os
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument

# Load environment variables
load_dotenv()

# ===========================================================================
# STEP 1: DOCUMENT LOADER & CHUNKING
# ===========================================================================
print("--- Running Step 1: Loading & Chunking ---")
sample_data = """
GraphRAG combines Knowledge Graphs with Retrieval-Augmented Generation. 
LangChain provides tools to extract entities like companies, technologies, and concepts.
Neo4j is a popular graph database, developed by Neo4j Inc. It stores nodes and relationships.
NetworkX is a Python library for studying graphs and network structures.
"""

with open("sample_document.txt", "w") as f:
    f.write(sample_data.strip())

loader = TextLoader("sample_document.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)
print(f"Total Chunks Created: {len(chunks)}")

# ===========================================================================
# STEP 2 & 3: ENTITY EXTRACTION & RELATIONSHIP MAPPING SCHEMAS
# ===========================================================================
class EntitySchema(BaseModel):
    name: str = Field(description="Entity name, e.g., 'Neo4j'")
    type: str = Field(description="Entity category, e.g., 'Technology'")

class RelationshipSchema(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(description="Relationship type in UPPERCASE_SNAKE_CASE")
    target: str = Field(description="Target entity name")

class KnowledgeGraph(BaseModel):
    entities: list[EntitySchema]
    relationships: list[RelationshipSchema]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(KnowledgeGraph)

# ===========================================================================
# STEP 4A: GRAPH STORAGE (NEO4J AURADB)
# ===========================================================================
print("\n--- Running Step 4A: Storing into Neo4j ---")
neo4j_graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

graph_documents = []

for chunk in chunks:
    extracted_data = structured_llm.invoke(
        f"Extract entities and relationships from:\n\n{chunk.page_content}"
    )
    
    nodes_map = {e.name: Node(id=e.name, type=e.type) for e in extracted_data.entities}
    
    relationships_list = []
    for r in extracted_data.relationships:
        if r.source in nodes_map and r.target in nodes_map:
            rel = Relationship(
                source=nodes_map[r.source],
                target=nodes_map[r.target],
                type=r.relation
            )
            relationships_list.append(rel)
            
    doc = Document(page_content=chunk.page_content)
    graph_doc = GraphDocument(
        nodes=list(nodes_map.values()),
        relationships=relationships_list,
        source=doc
    )
    graph_documents.append(graph_doc)

neo4j_graph.add_graph_documents(graph_documents)
print("Successfully stored graph documents in Neo4j!")

# ===========================================================================
# STEP 4B: GRAPH STORAGE (NETWORKX)
# ===========================================================================
print("\n--- Running Step 4B: Storing into NetworkX ---")
nx_graph = nx.DiGraph()

for chunk in chunks:
    extracted_data = structured_llm.invoke(
        f"Extract entities and relationships from:\n\n{chunk.page_content}"
    )
    for e in extracted_data.entities:
        nx_graph.add_node(e.name, type=e.type)
    for r in extracted_data.relationships:
        nx_graph.add_edge(r.source, r.target, relation=r.relation)

print(f"NetworkX Graph: {nx_graph.number_of_nodes()} Nodes, {nx_graph.number_of_edges()} Edges")

# Save visualization image
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(nx_graph, k=0.6)
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=9, font_weight='bold', arrowsize=15)
edge_labels = nx.get_edge_attributes(nx_graph, 'relation')
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=8)
plt.title("Knowledge Graph (NetworkX)")
plt.savefig("knowledge_graph.png")
print("Saved local graph visualization as 'knowledge_graph.png'")

# ===========================================================================
# STEP 5: GRAPH RETRIEVAL / GraphRAG
# ===========================================================================
print("\n=== STEP 5: GRAPH QUERY / GraphRAG RESULTS ===")

neo4j_graph.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=neo4j_graph,
    verbose=True,
    allow_dangerous_requests=True
)

questions = [
    "What does GraphRAG combine?",
    "Who developed Neo4j and what does it store?",
    "What is NetworkX used for?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    response = chain.invoke({"query": q})
    print(f"Answer: {response['result']}")