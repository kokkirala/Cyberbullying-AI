import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "rag/chroma_db"

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_collection(
    name="cyberbullying_knowledge",
    embedding_function=embedding_function
)

query = "What should I do if someone is cyberbullying me?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nRAG SEARCH RESULTS:\n")

for i, document in enumerate(results["documents"][0]):
    print(f"Result {i + 1}:")
    print(document)
    print("-" * 60)