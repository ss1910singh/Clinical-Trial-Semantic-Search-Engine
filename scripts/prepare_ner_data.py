import json
from datasets import Dataset, Features, Value, Sequence
import os
from transformers import AutoTokenizer


PROCESSED_DATA_FOLDER = os.path.join("./data", "processed")
ANNOTATION_FILE = os.path.join(PROCESSED_DATA_FOLDER, "all.jsonl") 
OUTPUT_DATASET_FOLDER = os.path.join("./data", "ner_dataset")
MODEL_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.1"
LABELS = ["CONDITION", "DRUG", "LAB_TEST", "VALUE", "OPERATOR", "PROCEDURE", "DEMOGRAPHIC"]

def prepare_data():
    if not os.path.exists(ANNOTATION_FILE):
        print(f"❌ Error: Annotation file not found at '{ANNOTATION_FILE}'.")
        return

    print(f"-> Loading annotation file from '{ANNOTATION_FILE}'...")
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    texts = [item["text"] for item in raw_data]
    spans = [
        [[str(start), str(end), label] for start, end, label in item.get("label", [])]
        for item in raw_data
    ]

    feature_schema = Features({
        'text': Value('string'),
        'ner_tags_spans': Sequence(Sequence(Value(dtype='string')))
    })

    dataset = Dataset.from_dict(
        {'text': texts, 'ner_tags_spans': spans},
        features=feature_schema
    )
    print(f"Successfully loaded {len(dataset)} annotated examples.")

    tag_names = ['O'] + [f'{prefix}-{tag}' for tag in LABELS for prefix in ['B', 'I']]
    tag2id = {tag: i for i, tag in enumerate(tag_names)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=False)
        all_labels = []
        for i, spans in enumerate(examples["ner_tags_spans"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [tag2id['O']] * len(word_ids)
            for start_str, end_str, label in spans:
                start = int(start_str)
                end = int(end_str)
                
                b_tag = f"B-{label}"
                i_tag = f"I-{label}"
                if b_tag not in tag2id: continue
                
                token_start = tokenized_inputs.char_to_token(i, start)
                token_end = tokenized_inputs.char_to_token(i, end - 1)
                if token_start is not None and token_end is not None:
                    label_ids[token_start] = tag2id[b_tag]
                    for token_idx in range(token_start + 1, token_end + 1):
                        label_ids[token_idx] = tag2id[i_tag]
            
            final_labels = [label_ids[idx] if word_id is not None else -100 for idx, word_id in enumerate(word_ids)]
            all_labels.append(final_labels)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    print("\nTokenizing text and aligning labels for the model...")
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)
    
    final_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    print("\nSplitting data into training and evaluation sets:")
    print(final_dataset)
    
    final_dataset.save_to_disk(OUTPUT_DATASET_FOLDER)
    print(f"\nProcessed dataset saved to '{OUTPUT_DATASET_FOLDER}'. You are now ready to train the model.")

if __name__ == "__main__":
    prepare_data()