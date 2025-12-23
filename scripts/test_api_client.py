import requests
import json

API_URL = "http://127.0.0.1:8000/search/"

TEST_QUERY = "patients with non-small cell lung cancer"

TOP_K = 5

def test_api():
    print(f"--- Testing Semantic Search API ---")
    print(f"-> Sending query to: {API_URL}")
    print(f"-> Query: '{TEST_QUERY}'")
    
    payload = {
        "query_text": TEST_QUERY,
        "top_k": TOP_K
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            results = response.json()
            print("\n--- Search Successful! ---")
            print("-> Top 5 matching criteria:")
            
            for i, hit in enumerate(results.get("top_k_results", [])):
                print(f"\n--- Result #{i+1} ---")
                print(f"  NCT ID:   {hit.get('nct_id')}")
                print(f"  Similarity: {hit.get('similarity_score', 0.0):.4f}")
                print(f"  Text:     \"{hit.get('criterion_text')}\"")
        
        else:
            print(f"\nAPI returned a non-200 status code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nCould not connect to the API.")
        print(f"   Please ensure the API server is running ('python scripts/09_api.py').")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api()