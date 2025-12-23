import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report

DATASET_FOLDER = os.path.join("./data", "ner_dataset")
MODEL_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_MODEL_FOLDER = os.path.join("./models", "clinical-ner-model")

LABELS = ["CONDITION", "DRUG", "LAB_TEST", "VALUE", "OPERATOR", "PROCEDURE", "DEMOGRAPHIC"]
tag_names = ['O'] + [f'{prefix}-{tag}' for tag in LABELS for prefix in ['B', 'I']]
id2tag = {i: tag for i, tag in enumerate(tag_names)}
tag2id = {tag: i for i, tag in enumerate(tag_names)}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

def train_model():
    if not os.path.exists(DATASET_FOLDER):
        print(f"Processed dataset not found at '{DATASET_FOLDER}'.")
        return

    print("Loading processed dataset...")
    processed_dataset = load_from_disk(DATASET_FOLDER)

    print(f"Loading model and tokenizer from '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(tag_names), id2label=id2tag, label2id=tag2id
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_FOLDER,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_steps=50,
        push_to_hub=False,
        no_cuda=True,
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting Model Training (Simplified for Compatibility) ---")
    print("This may take several hours on a CPU. Please be patient.")
    trainer.train()
    
    print("\n-> Training finished. Evaluating final model on the test set...")
    eval_results = trainer.evaluate()
    print(f"-> Final Evaluation Results: {eval_results}")

    trainer.save_model(OUTPUT_MODEL_FOLDER)
    print(f"\nTraining complete. Model has been saved to '{OUTPUT_MODEL_FOLDER}'.")

if __name__ == "__main__":
    train_model()