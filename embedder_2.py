"""
Build a LangChain-compatible FAISS vector store from the chunk metadata.
Run this once to generate the index that llm.py expects.
"""
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load chunk metadata (already created by embedder.py)
with open("data/chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert chunks to LangChain Documents
documents = []
for chunk in chunks:
    doc = Document(
        page_content=chunk["text"],
        metadata={
            "id": chunk["id"],
            "question": chunk["question"],
            "answer": chunk["answer"],
            "product": chunk.get("product", "Unknown"),
            "sheet": chunk.get("sheet", "Unknown"),
        },
    )
    documents.append(doc)

print(f"Loaded {len(documents)} documents.")

# Create embeddings and build the FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Save in LangChain's format (creates index.faiss + index.pkl in data/)
vectorstore.save_local("data/vectorstore")
print("LangChain FAISS vector store saved to data/vectorstore/")
