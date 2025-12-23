import os
import json
import spacy
from tqdm import tqdm

PROCESSED_DATA_FOLDER = os.path.join("./data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_FOLDER, "train_with_entities.jsonl") 
OUTPUT_FILE = os.path.join(PROCESSED_DATA_FOLDER, "train_structured_knowledge.jsonl")

NORMALIZATION_MAP = {
    "non-small cell lung cancer": "UMLS:C0007131",
    "nsclc": "UMLS:C0007131",
    "diabetes": "UMLS:C0011849",
    "diabetes mellitus": "UMLS:C0011849",
    "uncontrolled diabetes": "UMLS:C0011860",
    "hypertension": "UMLS:C0020538",
    "chemotherapy": "UMLS:C0013217",
    "biopsy": "UMLS:C0005558",
    "platelet count": "UMLS:C0032128",
    "serum creatinine": "UMLS:C0010323"
}

def structure_and_normalize():
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Standard spaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        print("'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found at '{INPUT_FILE}'. Please run '06_apply_ner_model.py' first.")
        return

    print(f"Loading data with entities from '{INPUT_FILE}'...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Structuring and Normalizing"):
            item = json.loads(line)
            text = item['text']
            entities = item['entities']
            doc = nlp(text)
            
            structured_record = {"text": text, "structured_entities": []}

            for entity in entities:
                entity_info = {
                    "text": entity['entity'],
                    "label": entity['label'],
                    "score": entity.get('score', 0.0),
                    "normalized_id": NORMALIZATION_MAP.get(entity['entity'].lower())
                }

                for token in doc:
                    if token.text in entity['entity']:
                        if any(child.dep_ == "neg" for child in token.head.children):
                            entity_info['is_negated'] = True
                        
                        for child in token.head.children:
                            child_text = child.text
                            for val_entity in entities:
                                if val_entity['label'] == 'VALUE' and child_text in val_entity['entity']:
                                    entity_info['related_value'] = val_entity['entity']
                        break
                
                structured_record["structured_entities"].append(entity_info)
            
            f_out.write(json.dumps(structured_record) + '\n')

    print(f"\nStructuring and normalization complete. Final knowledge base saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    structure_and_normalize()