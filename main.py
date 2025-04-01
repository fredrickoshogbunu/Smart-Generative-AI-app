from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import logging
from typing import List
import torch.nn.functional as F
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for all files
DOCUMENTS_PATH = os.path.join(SCRIPT_DIR, "documents.json")
INDEX_PATH = os.path.join(SCRIPT_DIR, "vector.index")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
index = None
documents = None
tokenizer = None
model = None
llm = None

def load_resources():
    """Load FAISS index, documents, tokenizer, model, and LLM."""
    global index, documents, tokenizer, model, llm

    # Create documents directory if it doesn't exist
    os.makedirs(os.path.dirname(DOCUMENTS_PATH), exist_ok=True)

    # Initialize documents if they don't exist
    if not os.path.exists(DOCUMENTS_PATH):
        with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Load or create FAISS index
    try:
        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
            logger.info(f"FAISS index loaded successfully with {index.ntotal} vectors")
        else:
            logger.info("Creating new FAISS index...")
            dimension = 384
            index = faiss.IndexFlatL2(dimension)
            faiss.write_index(index, INDEX_PATH)
            logger.info("Created new empty FAISS index")
    except Exception as e:
        logger.error(f"Error with FAISS index: {e}")
        raise

    # Load documents
    try:
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents.")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        documents = []

    # Load models
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Models loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    # Load LLM
    try:
        llm = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
        logger.info("LLM loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        llm = None

@app.on_event("startup")
def startup_event():
    """Run on application startup."""
    logger.info("\n=== Starting up ===")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir()}")
    load_resources()
    logger.info("=== Startup complete ===\n")

@app.get("/debug/paths")
def debug_paths():
    """Endpoint to check file paths."""
    return {
        "script_directory": SCRIPT_DIR,
        "documents_path": DOCUMENTS_PATH,
        "documents_exists": os.path.exists(DOCUMENTS_PATH),
        "index_path": INDEX_PATH,
        "index_exists": os.path.exists(INDEX_PATH),
        "working_directory": os.getcwd(),
        "directory_contents": os.listdir(SCRIPT_DIR)
    }

class QueryRequest(BaseModel):
    query: str

def get_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for the query string."""
    if not tokenizer or not model:
        raise ValueError("Embedding model not loaded")

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs)

    embedding = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not index or index.ntotal == 0:
        # Try to initialize if not already done
        try:
            await initialize_documents()
        except Exception as e:
            logger.error(f"Auto-initialization failed: {e}")
            return {
                "response": "The system needs to be initialized. Please call /initialize endpoint first."
            }
    
    try:
        if not request.query:
            raise HTTPException(
                status_code=400,
                detail={"error": "Query cannot be empty"}
            )

        # Check if index is initialized
        if index is None or index.ntotal == 0:
            return {
                "response": "The system is not yet initialized with any documents. Please add some documents first."
            }

        # Get query embedding
        query_embedding = get_query_embedding(request.query)
        query_embedding = np.float32(query_embedding)  # Ensure correct data type
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Perform search with error handling
        try:
            k = min(3, max(1, index.ntotal - 1))  # Ensure k is valid
            distances, indices = index.search(query_embedding, k)
            
            if indices.size == 0 or (indices[0] == 0).all():
                return {"response": "No relevant documents found in the index."}
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"error": "Vector search failed", "details": str(e)}
            )

        # Retrieve documents
        retrieved_docs = []
        for i in indices[0]:
            if 0 <= i < len(documents):
                retrieved_docs.append(documents[i])

        if not retrieved_docs:
            return {"response": "No relevant documents found"}

        # Generate response with prompt truncation
        context = "\n---\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Query: {request.query}\n\nContext:\n{context}\n\nAnswer:"
        
        # Truncate the prompt to the maximum allowed tokens (e.g. 1024)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        max_allowed_tokens = 1024
        if len(prompt_tokens) > max_allowed_tokens:
            prompt_tokens = prompt_tokens[:max_allowed_tokens]
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            logger.info(f"Prompt truncated to {max_allowed_tokens} tokens.")
        
        try:
            if llm:
                generation = llm(prompt, max_new_tokens=100, do_sample=True, top_p=0.95)
                response = generation[0]["generated_text"]
            else:
                response = "LLM service unavailable"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = f"Could not generate full response. Context:\n{context}"

        return {"response": response}

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Query processing failed", "details": str(e)}
        )

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Generative AI API is running"}

# Add a new endpoint to check system status
@app.get("/status")
async def check_status():
    return {
        "index_initialized": index is not None,
        "index_size": index.ntotal if index is not None else 0,
        "documents_loaded": len(documents) if documents is not None else 0,
        "models_loaded": tokenizer is not None and model is not None
    }

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a given text."""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

@app.post("/index")
async def index_documents():
    global index, documents
    
    try:
        if not documents:
            return {"message": "No documents to index"}
            
        # Initialize a new index
        dimension = 384  # dimension for sentence-transformers/all-MiniLM-L6-v2
        new_index = faiss.IndexFlatL2(dimension)
        
        # Process each document
        all_embeddings = []
        valid_documents = []
        
        for doc in documents:
            if not doc.get("content"):
                logger.warning(f"Skipping document {doc.get('id', 'unknown')} - no content")
                continue
                
            try:
                embedding = generate_embedding(doc["content"])
                all_embeddings.append(embedding)
                valid_documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue
        
        if all_embeddings:
            # Convert list of embeddings to numpy array
            all_embeddings = np.vstack(all_embeddings)
            # Add to index
            new_index.add(all_embeddings)
            
            # Update global index and save
            index = new_index
            faiss.write_index(index, INDEX_PATH)
            
            # Update documents with only valid ones
            documents = valid_documents
            with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
                json.dump(documents, f)
            
            return {
                "message": f"Successfully indexed {new_index.ntotal} documents",
                "index_size": new_index.ntotal
            }
        else:
            return {"message": "No valid documents to index"}
            
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Indexing failed", "details": str(e)}
        )

@app.get("/documents")
async def get_documents():
    return {"document_count": len(documents) if documents else 0,
            "documents": [{"id": doc.get("id"), 
                         "content_length": len(doc.get("content", "")),
                         "content_preview": doc.get("content", "")[:100] + "..."
                        } for doc in documents]}

def clean_markdown_content(content: str) -> str:
    """Clean markdown content by removing frontmatter and normalizing text."""
    # Remove YAML frontmatter
    if content.startswith('---'):
        try:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                content = parts[2]
        except Exception as e:
            logger.error(f"Error removing frontmatter: {e}")
    content = content.strip()
    content = content.replace('\r\n', '\n')
    return content

def read_folder_contents(folder_path: str) -> list[dict]:
    """Read all files from a folder and return list of document chunks."""
    documents = []
    max_chunk_size = 4000  # Reduced chunk size for better processing
    
    try:
        files = glob.glob(f"{folder_path}/**/*.*", recursive=True)
        for file_path in files:
            if Path(file_path).is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        content = clean_markdown_content(content)
                        content = f"Source: {file_path}\n\n{content}"
                        chunks = [content[i:i + max_chunk_size] 
                                  for i in range(0, len(content), max_chunk_size)]
                        for i, chunk in enumerate(chunks):
                            doc = {
                                "id": f"{Path(file_path).stem}_{i}",
                                "content": chunk,
                                "source": str(file_path),
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                            documents.append(doc)
                        logger.info(f"Processed {file_path} into {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {e}")
    
    return documents

@app.post("/initialize")
async def initialize_documents():
    global documents, index
    
    try:
        folders = ['kafka', 'react', 'spark']
        new_documents = []
        
        for folder in folders:
            folder_path = os.path.join(SCRIPT_DIR, folder)
            logger.info(f"Processing folder: {folder_path}")
            if os.path.exists(folder_path):
                folder_docs = read_folder_contents(folder_path)
                new_documents.extend(folder_docs)
                logger.info(f"Added {len(folder_docs)} documents from {folder}")
            else:
                logger.warning(f"Folder not found: {folder_path}")
        
        if not new_documents:
            raise HTTPException(
                status_code=404,
                detail={"error": "No documents found in specified folders"}
            )
        
        documents = new_documents
        os.makedirs(os.path.dirname(DOCUMENTS_PATH), exist_ok=True)
        with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        dimension = 384
        new_index = faiss.IndexFlatL2(dimension)
        
        all_embeddings = []
        for doc in documents:
            try:
                embedding = generate_embedding(doc["content"])
                all_embeddings.append(embedding.squeeze())
            except Exception as e:
                logger.error(f"Error generating embedding for {doc['id']}: {e}")
                continue
        
        if not all_embeddings:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to generate any embeddings"}
            )
        
        all_embeddings = np.vstack(all_embeddings)
        new_index.add(all_embeddings)
        index = new_index
        faiss.write_index(index, INDEX_PATH)
        
        return {
            "message": f"Successfully initialized with {len(documents)} documents",
            "index_size": index.ntotal,
            "folders_processed": folders,
            "total_chunks": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Initialization failed", "details": str(e)}
        )

@app.get("/debug/folders")
async def debug_folders():
    folder_structure = {}
    for folder in ['kafka', 'react', 'spark']:
        folder_path = os.path.join(SCRIPT_DIR, folder)
        folder_structure[folder] = {
            "path": folder_path,
            "exists": os.path.exists(folder_path),
            "is_dir": os.path.isdir(folder_path) if os.path.exists(folder_path) else False,
            "files": glob.glob(f"{folder_path}/**/*.*", recursive=True) if os.path.exists(folder_path) else []
        }
    return folder_structure

@app.get("/debug/file/{folder}/{filename}")
async def debug_file(folder: str, filename: str):
    file_path = os.path.join(SCRIPT_DIR, folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return {
                "path": file_path,
                "content_length": len(content),
                "first_100_chars": content[:100],
                "exists": True
            }
    except Exception as e:
        return {
            "path": file_path,
            "error": str(e),
            "exists": os.path.exists(file_path)
        }

@app.get("/debug/content/{folder}/{filename}")
async def debug_content(folder: str, filename: str):
    file_path = os.path.join(SCRIPT_DIR, folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            cleaned = clean_markdown_content(content)
            return {
                "original_length": len(content),
                "cleaned_length": len(cleaned),
                "first_100_chars_original": content[:100],
                "first_100_chars_cleaned": cleaned[:100],
                "num_chunks": len(cleaned) // 4000 + 1
            }
    except Exception as e:
        return {"error": str(e)}
