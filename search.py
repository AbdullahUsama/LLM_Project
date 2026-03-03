import json
import faiss
from sentence_transformers import SentenceTransformer

# Load once
index = faiss.read_index('data/faiss_index.bin')
with open('data/chunk_metadata.json', 'r') as f:
    chunks = json.load(f)
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query, k=3):
    emb = model.encode([query])
    D, I = index.search(emb, k)
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        print(f"\n[{i+1}] Score: {dist:.2f}")
        print(f"Q: {chunks[idx]['question']}")
        print(f"A: {chunks[idx]['answer'][:200]}...")

# Interactive loop
while True:
    q = input("\nQuery (or 'quit'): ")
    if q.lower() == 'quit':
        break
    search(q)