import re 
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_and_clean_file(filepath):

    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    cleaned_data = [
        item for item in data
        if item.get('question', '').strip() and item.get('answer', '').strip()
    ]

    print(f"total data entries: {len(data)}, cleaned data entries: {len(cleaned_data)}")
    return cleaned_data


def normalize_text(text):
    # Remove pipe-delimited table formatting
    text = re.sub(r'\s*\|\s*', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading bullets/numbers
    text = re.sub(r'^[•◦\-\d\.]+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def create_chunks(data):
    chunks = []

    for idx, item in enumerate(data):
        question = normalize_text(item['question'])
        answer = normalize_text(item['answer'])

        chunk = {
            'id': idx,
            'text': f"Question: {question}\nAnswer: {answer}",  # Combined for embedding
            'question': question,  # Store separately for display
            'answer': answer,
            'product': item.get('product', 'Unknown'),
            'sheet': item.get('sheet', 'Unknown')
        }
        chunks.append(chunk)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    
    print("Loading the embedding model...")
    model = SentenceTransformer(model_name)

    texts = [chunk['text'] for chunk in chunks]

    print(f"Embeding {len(texts)} chunks...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"embeddings shape: {embeddings.shape}")
    return embeddings, model

def create_faiss_index(embeddings, chunks, index_path = 'data/faiss_index.bin', metadata_path = 'data/chunk_metadata.json'):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Save the chunk metadata for later retrieval
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Chunk metadata saved to {metadata_path}")

    return index



def main():
    data = load_and_clean_file('data/all_qa_pairs.json')
    chunks = create_chunks(data)
    embeddings, model = embed_chunks(chunks)
    index = create_faiss_index(embeddings, chunks)

    print(f"total chunks: {len(chunks)}, embedding dimension: {embeddings.shape[1]}, embeddings shape: {embeddings.shape}, index size: {index.ntotal}")
    
if __name__ == "__main__":
    main()