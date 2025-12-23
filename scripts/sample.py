import pandas as pd
import os

PROCESSED_DATA_FOLDER = os.path.join("./data", "processed")
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_FOLDER, "train.csv")
ANNOTATION_SAMPLE_FILE = os.path.join(PROCESSED_DATA_FOLDER, "sample.txt")
SAMPLE_SIZE = 1000

def create_sample_for_annotation():
    print("Creating Annotation Sample")
    
    if not os.path.exists(PROCESSED_TRAIN_FILE):
        print("File not found.")
        return

    df = pd.read_csv(PROCESSED_TRAIN_FILE)
    unique_criteria = df['criterion_text'].dropna().unique()
    sample_criteria = unique_criteria[:SAMPLE_SIZE]
    with open(ANNOTATION_SAMPLE_FILE, 'w', encoding='utf-8') as f:
        for text in sample_criteria:
            f.write(f"{text.strip()}\n")
            
if __name__ == "__main__":
    create_sample_for_annotation()