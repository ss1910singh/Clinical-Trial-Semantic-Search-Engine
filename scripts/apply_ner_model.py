import os
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

PROCESSED_DATA_FOLDER = os.path.join("./data", "processed")
MODEL_PATH = os.path.join("./models", "clinical-ner-model")
INPUT_FILES = {
    "train": os.path.join(PROCESSED_DATA_FOLDER, "train.csv"),
    "test": os.path.join(PROCESSED_DATA_FOLDER, "test.csv")
}
OUTPUT_FILES = {
    "train": os.path.join(PROCESSED_DATA_FOLDER, "train_with_entities.jsonl"),
    "test": os.path.join(PROCESSED_DATA_FOLDER, "test_with_entities.jsonl")
}
BATCH_SIZE = 64 

def apply_ner_model_robust():

    NUM_THREADS = 8
    torch.set_num_threads(NUM_THREADS)
    print(f"-> Set PyTorch to use {NUM_THREADS} CPU threads for processing.")

    if not os.path.exists(MODEL_PATH):
        print(f"Trained model not found at '{MODEL_PATH}'.")
        return
    
    print(f"Loading tokenizer and model from '{MODEL_PATH}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        device = "cpu"
        model.to(device)
        model.eval() 
        print(f"   --- Model Loaded Successfully to CPU ---")
    except Exception as e:
        print(f"Error loading the model or tokenizer: {e}")
        return

    for split, input_file in INPUT_FILES.items():
        output_file = OUTPUT_FILES[split]
        
        if not os.path.exists(input_file):
            print(f"Input file not found for '{split}' split at '{input_file}'. Skipping.")
            continue
            
        print(f"\nLoading full '{split}' dataset from '{input_file}'...")
        df = pd.read_csv(input_file)
        texts = [str(text) for text in df['criterion_text'].dropna().tolist()]
        print(f"Found {len(texts)} criteria to process for the '{split}' split.")

        print(f"Applying NER model to all '{split}' criteria...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Processing {split} batches"):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                try:
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs.to(device)

                    with torch.no_grad():
                        logits = model(**inputs).logits
                    
                    predictions = torch.argmax(logits, dim=2)
                    
                    for j in range(len(batch_texts)):
                        original_text = batch_texts[j]
                        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][j])
                        predicted_labels = [model.config.id2label[p.item()] for p in predictions[j]]

                        entities = []
                        current_entity_words = []
                        current_entity_label = None

                        for token, label in zip(tokens, predicted_labels):
                            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                                continue

                            word = token.replace("##", "") if token.startswith("##") else " " + token
                            word = word.strip()

                            if label.startswith("B-"):
                                if current_entity_words:
                                    entities.append({"entity": "".join(current_entity_words).strip(), "label": current_entity_label})
                                current_entity_words = [word]
                                current_entity_label = label[2:]
                            elif label.startswith("I-") and current_entity_label == label[2:]:
                                current_entity_words.append(word)
                            else: 
                                if current_entity_words:
                                    entities.append({"entity": "".join(current_entity_words).strip(), "label": current_entity_label})
                                current_entity_words = []
                                current_entity_label = None
                        
                        if current_entity_words:
                            entities.append({"entity": "".join(current_entity_words).strip(), "label": current_entity_label})

                        output_record = {"text": original_text, "entities": entities}
                        f.write(json.dumps(output_record) + '\n')

                except Exception as e:
                    print(f"  - An unexpected error occurred in batch starting at index {i}: {e}")

        print(f"\nNER application for '{split}' complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    apply_ner_model_robust()