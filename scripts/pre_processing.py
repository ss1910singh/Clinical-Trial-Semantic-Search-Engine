import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import csv
from ingest_data import fetch_data
from segmentation import segment_text_batch

DATA_FOLDER = "./data"
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")
DATASET_NAME = "louisbrulenaudet/clinical-trials"
EDA_PLOT_FILE = os.path.join(DATA_FOLDER, "criteria_length_distribution.png")

COLUMNS_TO_KEEP = [
    'nct_id', 'eligibility_criteria', 'overall_status', 'phases', 'study_type',
    'minimum_age', 'maximum_age', 'sex', 'healthy_volunteers', 'conditions',
    'keywords', 'interventions', 'mesh_terms', 'locations', 'brief_title',
    'official_title', 'brief_summary'
]

def main():
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Standard spaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        print("-> Model 'en_core_web_sm' not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    ds = fetch_data(DATASET_NAME)
    
    split = ds['train'].train_test_split(test_size=0.1, seed=42)
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    for split_name in split.keys():
        print(f"\n--- Processing '{split_name}' data ---")
        df = split[split_name].to_pandas()
        df = df[COLUMNS_TO_KEEP]
        df.dropna(subset=['eligibility_criteria'], inplace=True)
        print(f"  -> Started with {len(df)} trials after filtering.")

        if split_name == "train":
            df['criteria_length'] = df['eligibility_criteria'].str.len()
            plt.figure(figsize=(12, 6))
            sns.histplot(df['criteria_length'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Eligibility Criteria Text Length (Train Set)')
            plt.savefig(EDA_PLOT_FILE)
            plt.close()
            print(f"  -> EDA plot saved to '{EDA_PLOT_FILE}'.")

        criteria_list = df['eligibility_criteria'].tolist()
        segmented_results = segment_text_batch(criteria_list, nlp)
        df['segmented_criteria'] = segmented_results
        
        df_exploded = df.explode('segmented_criteria').rename(columns={'segmented_criteria': 'criterion_text'})
        df_exploded.dropna(subset=['criterion_text'], inplace=True)
        
        final_columns = [col for col in df.columns if col not in ['eligibility_criteria', 'criteria_length', 'segmented_criteria']]
        final_columns.insert(1, 'criterion_text')
        df_final = df_exploded[final_columns].copy()
        
        output_path = os.path.join(PROCESSED_DATA_FOLDER, f"{split_name}.csv")
        df_final.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"  -> Final processed file with {len(df_final)} criteria saved to '{output_path}'.")

if __name__ == "__main__":
    main()