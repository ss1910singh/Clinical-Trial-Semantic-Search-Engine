import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DATA_FOLDER = os.path.join("./data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_FOLDER, "train_structured_knowledge.jsonl") 
DB_PATH = "vector_db"
COLLECTION_NAME = "clinical_trials"
VECTOR_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 512

def vectorize_and_index():
    if not os.path.exists(INPUT_FILE):
        print(f"Structured knowledge file not found at '{INPUT_FILE}'.")
        return

    print(f"Loading Sentence-Transformer model: '{VECTOR_MODEL}'...")
    model = SentenceTransformer(VECTOR_MODEL)
    print("   --- Model Loaded Successfully ---")
    print(f"-> Initializing vector database at '{DB_PATH}'...")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   --- Database collection '{COLLECTION_NAME}' ready ---")

    print(f"Starting to process and index documents from '{INPUT_FILE}'...")
    
    texts_batch = []
    metadata_batch = []
    ids_batch = []
    doc_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading and Indexing Documents"):
            try:
                record = json.loads(line)
                text = record.get("text")
                nct_id = record.get("nct_id", f"doc_{doc_count}")
                
                if not text:
                    continue

                texts_batch.append(text)
                metadata_batch.append({"nct_id": nct_id, "text": text})
                ids_batch.append(str(doc_count))
                doc_count += 1

                if len(texts_batch) >= BATCH_SIZE:
                    embeddings = model.encode(texts_batch)
                    collection.add(
                        embeddings=embeddings.tolist(),
                        documents=texts_batch,
                        metadatas=metadata_batch,
                        ids=ids_batch
                    )
                    
                    texts_batch, metadata_batch, ids_batch = [], [], []

            except Exception as e:
                print(f"Error processing line: {e}")

    if texts_batch:
        embeddings = model.encode(texts_batch)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts_batch,
            metadatas=metadata_batch,
            ids=ids_batch
        )

    print(f"   Successfully indexed {doc_count} documents into the '{COLLECTION_NAME}' database.")
    print(f"   Your vector database is saved in the '{DB_PATH}' folder.")

if __name__ == "__main__":
    vectorize_and_index()