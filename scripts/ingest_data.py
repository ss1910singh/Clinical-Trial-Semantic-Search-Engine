import os
from datasets import load_dataset, load_from_disk, DatasetDict

def fetch_data(dataset_name: str, base_data_folder: str = "data") -> DatasetDict:
    raw_data_folder = os.path.join(base_data_folder, "raw")
    sanitized_name = dataset_name.replace("/", "__")
    dataset_path = os.path.join(raw_data_folder, sanitized_name)
    if not os.path.exists(dataset_path):
        os.makedirs(raw_data_folder, exist_ok=True)
        print(f"Downloading dataset '{dataset_name}' from the Hugging Face Hub...")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to local cache at '{dataset_path}'.")
    else:
        print(f"Loading dataset from local cache: '{dataset_path}'")
        dataset = load_from_disk(dataset_path)
        
    return dataset