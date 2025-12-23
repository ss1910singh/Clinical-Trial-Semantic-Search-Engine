import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = os.path.join("./models", "clinical-ner-model")

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Trained model not found at '{MODEL_PATH}'.")
        return

    print(f"Loading your custom-trained model from '{MODEL_PATH}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        
        ner_pipeline = pipeline(
            "token-classification", 
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1 
        )
        print("\n--- Model Loaded Successfully. Ready for Inference ---")
    except Exception as e:
        print(f"Error loading the model pipeline: {e}")
        return
    
    test_sentences = [
        "Patient must have a platelet count > 100,000/μL.",
        "A history of non-small cell lung cancer is required.",
        "Subject must be male and over 18 years of age.",
        "No prior chemotherapy within the last 6 months.",
        "ECOG performance status must be 0 or 1.",
        "Exclusion of patients with uncontrolled diabetes.",
        "Must provide a sample from a recent biopsy.",
        "Serum creatinine less than 1.5x the upper limit of normal."
    ]

    for i, sentence in enumerate(test_sentences):
        print(f"\n--- Analyzing Sentence #{i+1} ---")
        print(f"Text: '{sentence}'")
        
        try:
            entities = ner_pipeline(sentence)
            if not entities:
                print("  -> No entities found.")
            else:
                print("  -> Found Entities:")
                for entity in entities:
                    print(f"    - Text:  '{entity['word']}'")
                    print(f"      Label: {entity['entity_group']}")
                    print(f"      Score: {entity['score']:.4f}")
        except Exception as e:
            print(f"  An error occurred during prediction: {e}")

if __name__ == "__main__":
    test_model()