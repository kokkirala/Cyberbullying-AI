import os
import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "rag/chroma_db"

os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="cyberbullying_knowledge",
    embedding_function=embedding_function
)

documents = []
ids = []

for filename in os.listdir(KNOWLEDGE_DIR):

    if filename.endswith(".txt"):

        filepath = os.path.join(
            KNOWLEDGE_DIR,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        chunks = [
            text[i:i + 500]
            for i in range(0, len(text), 500)
        ]

        for number, chunk in enumerate(chunks):

            documents.append(chunk)
            ids.append(f"{filename}_{number}")

collection.upsert(
    documents=documents,
    ids=ids
)

print("RAG knowledge base created successfully.")
print("Documents/chunks:", collection.count())