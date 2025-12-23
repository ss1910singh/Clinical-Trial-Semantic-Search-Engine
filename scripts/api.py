import os
import chromadb
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn

DB_PATH = "./vector_db"
COLLECTION_NAME = "clinical_trials"
VECTOR_MODEL = 'all-MiniLM-L6-v2'


try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("   --- Vector Database loaded successfully ---")

    model = SentenceTransformer(VECTOR_MODEL)
    print("   --- Sentence Transformer model loaded successfully ---")
except Exception as e:
    print(f"Could not load models or database. {e}")
    print(f"   Looking for database at: {os.path.abspath(DB_PATH)}")
    exit()

app = FastAPI(
    title="Clinical Trial Semantic Search API",
    description="An API to find relevant clinical trials based on semantic meaning.",
    version="1.0.0"
)

class SearchQuery(BaseModel):
    query_text: str
    top_k: int = 5

@app.get("/")
def read_root():
    return {"message": "Clinical Trial Semantic Search API is running."}

@app.post("/search/")
def search_trials(query: SearchQuery):
    try:
        query_embedding = model.encode(query.query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.top_k
        )
        
        hits = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted_results = [
            {
                "nct_id": hit.get('nct_id'),
                "criterion_text": hit.get('text'),
                "similarity_score": 1 - distance
            }
            for hit, distance in zip(hits, distances)
        ]

        return {
            "query": query.query_text,
            "top_k_results": formatted_results
        }
    except Exception as e:
        return {"error": f"An error occurred during search: {str(e)}"}

if __name__ == "__main__":
    print("\n--- Starting FastAPI Server ---")
    print(f"Access the API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)