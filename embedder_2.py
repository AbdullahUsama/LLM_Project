"""
Build a LangChain-compatible FAISS vector store from the chunk metadata.
Run this once to generate the index that llm.py expects.
"""
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.data_pipeline import append_langchain_vectorstore

summary = append_langchain_vectorstore()
print(f"Append summary: {summary}")
