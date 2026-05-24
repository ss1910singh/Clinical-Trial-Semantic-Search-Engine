# ⚕️ Clinical Trial Semantic Search Engine

## 📖 Overview

Clinical trial protocols contain over **3.5 million unstructured eligibility criteria**, making it nearly impossible for doctors and patients to find relevant trials using simple keyword search.  
A query for `"diabetes"` would miss trials that list `"HbA1c > 6.5%"`, and a search for `"patients without hypertension"` would fail completely.

This project is an **end-to-end AI-driven system** that solves this problem. It uses **Natural Language Processing (NLP)** and a **fine-tuned Biomedical NER model** to read, understand, and structure all 3.5 million criteria.

The final result is an intelligent **Semantic Search Engine** that can understand complex queries (e.g., _“Find trials for patients with lung cancer but no history of heart failure”_) and return the most clinically relevant results in milliseconds.

This system bridges the gap between raw, unstructured text and a structured, searchable knowledge base, making clinical trial data:

- 🩺 **Accessible** for physicians matching patients  
- 🔬 **Computable** for researchers analyzing trends  
- 📊 **Scalable** for repositories managing millions of documents  

---

## 🎯 Problem Statement

Clinical trial eligibility criteria are written in complex, free-form medical language. This data is:

- ❌ **Unstructured:** A mix of full sentences, bullet points, and medical jargon  
- ❌ **Ambiguous:** `> 50%` means nothing without knowing it refers to LVEF  
- ❌ **Unsearchable:** Keyword-based search fails to capture semantic meaning  

### ✅ This project solves it by building a 5-phase pipeline that:
- ✔️ Segments 3.5M criteria into individual sentences  
- ✔️ Extracts 7 types of key medical entities using a fine-tuned BioBERT model  
- ✔️ Structures this data by linking entities together (e.g., LAB_TEST → VALUE)  
- ✔️ Indexes the semantic meaning of every criterion in a high-speed Vector Database  
- ✔️ Deploys the system as a simple, fast **FastAPI** search endpoint  

---

## 🚀 Key Features

- 🧠 **Biomedical NER:** `dmis-lab/biobert-base-cased-v1.1` fine-tuned to detect 7 entity types  
- 🏛️ **Knowledge Base Creation:** Extracts structured entities from 3.5M criteria into JSONL  
- 🔗 **Relation Extraction:** Rule-based spaCy system links LAB_TESTS with VALUES and OPERATORS  
- 🏷️ **Entity Normalization:** Maps extracted text (e.g., “NSCLC”) to UMLS codes  
- ⚡ **High-Speed Semantic Search:** Uses `all-MiniLM-L6-v2` for embedding-based semantic matching  
- 🗄️ **Vector Database:** Built with **ChromaDB** for millisecond-scale cosine similarity search  
- 🔌 **API-Ready:** FastAPI `/search/` endpoint for external applications  

---

## 🏗️ System Architecture & Methodology

### **Phase I: Data Ingestion & Segmentation**

**Goal:** Ingest 3.5M+ raw criteria and segment them into clean sentences.  
**Scripts:** `ingest_data.py`, `segmentation.py`, `01_preprocess_data.py`

**Action:**  
Uses `spaCy`’s `nlp.pipe()` for efficient batch processing. The script cleans and segments text, outputting `train.csv` and `test.csv`.

**Equation (Self-Attention):**

```math
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
````

---

### **Phase II: NER Model Fine-Tuning (Core Intelligence Engine)**

**Goal:** Train BioBERT to recognize 7 custom biomedical entities.
**Scripts:** `create_annotation_sample.py`, `prepare_ner_data.py`, `train_ner_model.py`, `test_ner_model.py`

#### Models & Training Details

* **Model:** `dmis-lab/biobert-base-cased-v1.1`
* **Loss Function:** Cross-Entropy Loss

```math
J(\theta) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
```

* **Evaluation Metrics:** Precision, Recall, F1

```math
P = \frac{TP}{TP + FP} \\
R = \frac{TP}{TP + FN} \\
F1 = 2 \times \frac{P \times R}{P + R}
```

**Output:** Fine-tuned model saved to `models/clinical-ner-model/`

---

### **Phase III: Structuring & Normalization**

**Goal:** Apply trained model on all 3.5M+ sentences and structure the results.
**Scripts:** `apply_ner_model.py`, `structure_and_normalize.py`

#### Techniques:

* **Rule-Based Relation Extraction:** Uses dependency parsing in spaCy
* **Entity Normalization:** Python dictionary maps entity to UMLS codes
* **Example Rule:**

  * If a LAB_TEST has a negation child → add `is_negated: True`
  * If a LAB_TEST links to VALUE → add `related_value`

**Output:** `train_structured_knowledge.jsonl`

---

### **Phase IV: Vectorization & Indexing**

**Goal:** Build the semantic search index.

**Script:** `vectorize_and_index.py`
**Model:** `all-MiniLM-L6-v2` (384D vector embeddings)
**Database:** `ChromaDB`

#### Search Equation (Cosine Similarity)

```math
Similarity(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}
```

**Output:** Stored in `vector_db/` for high-speed kNN search.

---

### **Phase V: API Deployment**

**Goal:** Serve the search engine through an API.
**Script:** `api.py`

#### Endpoint:

```
POST http://127.0.0.1:8000/search/
```

**Example Request:**

```json
{"query_text": "patients with diabetes but no heart failure"}
```

**Example Response:**

```json
{
  "query": "patients with diabetes but no heart failure",
  "top_k_results": [
    {
      "nct_id": "NCT00055314",
      "criterion_text": "Exclusion of patients with uncontrolled diabetes.",
      "similarity_score": 0.8521
    }
  ]
}
```

---

## 🛠️ Tech Stack

| Category           | Tools & Libraries                   |
| ------------------ | ----------------------------------- |
| **Core Language**  | Python 3.11                         |
| **Data Science**   | pandas, numpy, matplotlib, seaborn  |
| **NLP**            | spaCy                               |
| **Deep Learning**  | PyTorch                             |
| **Model Pipeline** | Hugging Face transformers, datasets |
| **Embeddings**     | sentence-transformers               |
| **Vector DB**      | chromadb                            |
| **API Server**     | FastAPI, uvicorn                    |
| **Evaluation**     | seqeval                             |
| **Utilities**      | tqdm, json                          |

---

## 📊 Results & Visualizations

1. **Training Performance:**

   * Model trained for 3 epochs on CPU
   * Decreasing training loss confirmed learning
   * Final Macro F1 ≈ **0.23**

2. **Entity Distribution:**

   * `VALUE` most frequent
   * Followed by `CONDITION` and `LAB_TEST`

3. **Example Predictions:**

```
Text: 'No prior chemotherapy within the last 6 months.'
Entities:
 - 'no' → OPERATOR
 - 'chemotherapy' → DRUG
 - 'within the last 6 months' → VALUE
```

---

## 📂 Repository Structure

```
clinical_trials_ner/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample.txt
│   │   ├── all.jsonl
│   │   ├── train_with_entities.jsonl
│   │   └── train_structured_knowledge.jsonl
│   └── ner_dataset/
│
├── plots/
│   ├── training_loss_curve.png
│   ├── final_evaluation_metrics.png
│   └── entity_distribution.png
│
├── scripts/
│   ├── ingest_data.py
│   ├── segmentation.py
│   ├── preprocess_data.py
│   ├── create_annotation_sample.py
│   ├── prepare_ner_data.py
│   ├── train_ner_model.py
│   ├── test_ner_model.py
│   ├── apply_ner_model.py
│   ├── structure_and_normalize.py
│   ├── visualize_results.py
│   ├── vectorize_and_index.py
│   └── api.py
│
├── models/
│   └── clinical-ner-model/
│
└── vector_db/
```

---

## 📥 Installation

```bash
# 1. Clone the repository
git https://github.com/ss1910singh/Clinical-Trial-Semantic-Search-Engine.git
cd Clinical-Trial-Semantic-Search-Engine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy model
python -m spacy download en_core_web_sm
```

---

## ▶️ How to Run the Pipeline

```bash
# Phase I: Preprocessing (~10-15 min)
python scripts/preprocess_data.py

# Phase II: Create and train the NER model
python scripts/create_annotation_sample.py
# (Annotate 'sample.txt' manually, save as 'all.jsonl')
python scripts/prepare_ner_data.py
python scripts/train_ner_model.py

# Optional: Test & visualize
python scripts/test_ner_model.py
python scripts/visualize_results.py

# Phase III: Apply NER and structure data
python scripts/apply_ner_model.py
python scripts/structure_and_normalize.py

# Phase IV: Build vector database
python scripts/vectorize_and_index.py

# Phase V: Launch FastAPI server
python scripts/api.py
