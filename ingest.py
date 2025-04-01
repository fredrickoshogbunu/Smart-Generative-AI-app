# ingestion/ingest.py
import os
import glob
import json
from transformers import AutoTokenizer, AutoModel  
import torch
import faiss 
import numpy as np
from kafka import KafkaConsumer 

# Directories containing source files
DATA_DIRS = {
    "kafka": "../kafka",
    "react": "../react",
    "spark": "../spark"
}

def read_files():
    documents = []
    # Traverse each folder recursively for .py files (can be extended to other types)
    for folder, path in DATA_DIRS.items():
        for filepath in glob.glob(os.path.join(path, "**/*.py"), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                documents.append({
                    "folder": folder,
                    "filepath": filepath,
                    "content": content
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return documents

def generate_embeddings(documents):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc["content"], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            model_output = model(**inputs)
        # Mean pooling for a single embedding vector
        embedding = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def kafka_consumer_simulation():
    # This is a simulated Kafka consumer to listen for file updates.
    consumer = KafkaConsumer('file_updates', bootstrap_servers=['localhost:9092'])
    for message in consumer:
        print("Received update:", message.value)
        # Upon receiving a message, re-run the ingestion pipeline as needed.
        # (For production, implement proper update logic here.)

if __name__ == "__main__":
    docs = read_files()
    if not docs:
        print("No documents found. Please ensure the source folders are correctly set.")
        exit(1)
    embeddings = generate_embeddings(docs)
    index = create_faiss_index(embeddings)
    # Save the FAISS index and documents mapping to disk for later retrieval.
    faiss.write_index(index, "vector.index")
    with open("documents.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("Indexing complete.")
