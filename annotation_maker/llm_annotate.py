import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROCESSED_DATA_FOLDER = os.path.join("../data", "processed")
ANNOTATION_SAMPLE_FILE = os.path.join(PROCESSED_DATA_FOLDER, "sample.txt")
OUTPUT_JSONL_FILE = os.path.join(PROCESSED_DATA_FOLDER, "all.jsonl")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

FEW_SHOT_EXAMPLES = [
    {"text": "Must have a diagnosis of metastatic breast cancer.", "label": [[24, 49, "CONDITION"]]},
    {"text": "History of Type 2 Diabetes Mellitus is an exclusion.", "label": [[11, 35, "CONDITION"]]},
    {"text": "Patients with active, uncontrolled hypertension are not eligible.", "label": [[16, 41, "CONDITION"]]},
    {"text": "Subject must have an ECOG performance status of 0 or 1.", "label": [[21, 48, "CONDITION"]]},
    {"text": "No evidence of brain metastases.", "label": [[0, 2, "OPERATOR"], [14, 31, "CONDITION"]]},
    {"text": "Prior treatment with cisplatin is not allowed.", "label": [[21, 30, "DRUG"]]},
    {"text": "Receiving systemic corticosteroids within the last 14 days.", "label": [[10, 31, "DRUG"], [41, 54, "VALUE"]]},
    {"text": "Must not be taking any investigational product.", "label": [[25, 47, "DRUG"]]},
    {"text": "History of allergy to monoclonal antibodies.", "label": [[21, 42, "DRUG"]]},
    {"text": "Concomitant use of strong CYP3A4 inhibitors is prohibited.", "label": [[20, 42, "DRUG"]]},
    {"text": "Hemoglobin must be >= 9.0 g/dL.", "label": [[0, 10, "LAB_TEST"], [19, 29, "VALUE"]]},
    {"text": "Platelet count of at least 100,000/microL.", "label": [[0, 14, "LAB_TEST"], [18, 40, "VALUE"]]},
    {"text": "Serum creatinine <= 1.5 x ULN.", "label": [[0, 16, "LAB_TEST"], [17, 27, "VALUE"]]},
    {"text": "Absolute Neutrophil Count (ANC) > 1,000/mm3.", "label": [[0, 30, "LAB_TEST"], [31, 42, "VALUE"]]},
    {"text": "Total bilirubin must be within normal limits.", "label": [[0, 15, "LAB_TEST"], [25, 44, "VALUE"]]},
    {"text": "Must be >= 18 years of age.", "label": [[8, 25, "DEMOGRAPHIC"]]},
    {"text": "Only female subjects are eligible for this study.", "label": [[5, 18, "DEMOGRAPHIC"]]},
    {"text": "Participant must be a non-smoker for at least 6 months.", "label": [[21, 31, "DEMOGRAPHIC"], [38, 51, "VALUE"]]},
    {"text": "Exclusion of pregnant or breastfeeding women.", "label": [[13, 42, "DEMOGRAPHIC"]]},
    {"text": "Must be a healthy volunteer.", "label": [[9, 26, "DEMOGRAPHIC"]]},
    {"text": "No prior radiation therapy to the pelvis.", "label": [[3, 26, "PROCEDURE"]]},
    {"text": "History of allogeneic stem cell transplant.", "label": [[11, 40, "PROCEDURE"]]},
    {"text": "Must undergo a baseline tumor biopsy.", "label": [[27, 39, "PROCEDURE"]]},
    {"text": "Previous major surgery within 4 weeks of screening.", "label": [[9, 22, "PROCEDURE"], [31, 51, "VALUE"]]},
    {"text": "Willing to provide blood samples for pharmacokinetic analysis.", "label": [[16, 29, "PROCEDURE"]]},
    {"text": "Life expectancy of more than 12 weeks.", "label": [[19, 28, "OPERATOR"], [29, 38, "VALUE"]]},
    {"text": "A score of less than 3 on the pain scale.", "label": [[8, 17, "OPERATOR"], [18, 19, "VALUE"]]},
    {"text": "Must have recovered from toxicities of prior therapy.", "label": [[39, 53, "CONDITION"], [57, 71, "DRUG"]]},
    {"text": "No active hepatitis B or C infection.", "label": [[3, 35, "CONDITION"]]},
    {"text": "Patient has symptomatic congestive heart failure.", "label": [[12, 49, "CONDITION"]]},
    {"text": "History of myocardial infarction within the past 6 months.", "label": [[11, 33, "CONDITION"], [43, 58, "VALUE"]]},
    {"text": "Documented HER2-positive breast cancer.", "label": [[11, 41, "CONDITION"]]},
    {"text": "Patients with a history of seizure disorders are excluded.", "label": [[26, 43, "CONDITION"]]},
    {"text": "Must not have clinically significant peripheral neuropathy.", "label": [[14, 52, "CONDITION"]]},
    {"text": "No prior treatment with any PARP inhibitor.", "label": [[3, 6, "OPERATOR"], [28, 42, "DRUG"]]},
    {"text": "Concurrent use of warfarin is not permitted.", "label": [[19, 27, "DRUG"]]},
    {"text": "Must be on a stable dose of metformin for at least 3 months.", "label": [[29, 38, "DRUG"], [44, 59, "VALUE"]]},
    {"text": "Patients receiving chronic immunosuppressive therapy.", "label": [[21, 51, "DRUG"]]},
    {"text": "Known hypersensitivity to pembrolizumab.", "label": [[26, 39, "DRUG"]]},
    {"text": "Calculated creatinine clearance >= 60 mL/min.", "label": [[10, 32, "LAB_TEST"], [33, 44, "VALUE"]]},
    {"text": "Fasting glucose must be < 100 mg/dL.", "label": [[0, 16, "LAB_TEST"], [26, 37, "VALUE"]]},
    {"text": "Left ventricular ejection fraction (LVEF) of > 50%.", "label": [[0, 37, "LAB_TEST"], [38, 47, "VALUE"]]},
    {"text": "White blood cell (WBC) count must be within normal limits.", "label": [[0, 26, "LAB_TEST"], [36, 55, "VALUE"]]},
    {"text": "Adequate hepatic function as shown by AST and ALT levels.", "label": [[9, 25, "LAB_TEST"], [39, 55, "LAB_TEST"]]},
    {"text": "Subject must be male and between 18 and 65 years of age.", "label": [[17, 21, "DEMOGRAPHIC"], [31, 55, "DEMOGRAPHIC"]]},
    {"text": "Postmenopausal women are eligible.", "label": [[0, 21, "DEMOGRAPHIC"]]},
    {"text": "Participant must be of Asian descent.", "label": [[21, 35, "DEMOGRAPHIC"]]},
    {"text": "Subject is neither pregnant nor breastfeeding.", "label": [[12, 20, "DEMOGRAPHIC"], [25, 38, "DEMOGRAPHIC"]]},
    {"text": "Must have a BMI between 18.5 and 30.0 kg/m2.", "label": [[13, 31, "LAB_TEST"], [40, 57, "VALUE"]]},
    {"text": "Must have completed prior radiation therapy at least 2 weeks ago.", "label": [[24, 42, "PROCEDURE"], [51, 66, "VALUE"]]},
    {"text": "No major surgery within 28 days of starting study drug.", "label": [[3, 16, "PROCEDURE"], [25, 35, "VALUE"], [46, 56, "DRUG"]]},
    {"text": "Patient has undergone a prior colonoscopy.", "label": [[28, 39, "PROCEDURE"]]},
    {"text": "Willing to abstain from alcohol during the trial.", "label": [[19, 36, "PROCEDURE"]]},
    {"text": "Must provide written informed consent.", "label": [[16, 39, "PROCEDURE"]]},
    {"text": "QTc interval must not exceed 470 ms.", "label": [[0, 12, "LAB_TEST"], [25, 33, "OPERATOR"], [34, 41, "VALUE"]]},
    {"text": "No history of stroke or transient ischemic attack.", "label": [[3, 6, "OPERATOR"], [18, 24, "CONDITION"], [28, 52, "CONDITION"]]},
    {"text": "Patient has diabetes mellitus controlled by diet alone.", "label": [[12, 29, "CONDITION"], [43, 54, "PROCEDURE"]]},
    {"text": "Must not have received a live vaccine within 30 days of Cycle 1, Day 1.", "label": [[25, 37, "DRUG"], [46, 56, "VALUE"]]},
    {"text": "ALT and AST must be less than 3 times the upper limit of normal.", "label": [[0, 11, "LAB_TEST"], [20, 28, "OPERATOR"], [29, 64, "VALUE"]]},
    {"text": "Age >= 18 years and a life expectancy of at least 12 weeks.", "label": [[0, 12, "DEMOGRAPHIC"], [32, 58, "VALUE"]]},
    {"text": "No active bleeding and a platelet count > 50,000/mm3.", "label": [[3, 18, "CONDITION"], [25, 39, "LAB_TEST"], [40, 52, "VALUE"]]},
    {"text": "Patients with a history of glaucoma are excluded.", "label": [[26, 34, "CONDITION"]]},
    {"text": "No use of systemic antibiotics within 7 days of enrollment.", "label": [[3, 6, "OPERATOR"], [10, 29, "DRUG"], [38, 59, "VALUE"]]},
    {"text": "Subject must have adequate renal function, defined as creatinine clearance > 30 mL/min.", "label": [[21, 41, "LAB_TEST"], [53, 75, "LAB_TEST"], [76, 87, "VALUE"]]},
    {"text": "Exclusion for patients with a history of cardiac arrhythmia.", "label": [[35, 54, "CONDITION"]]},
    {"text": "Must agree to not use any nicotine products during the study.", "label": [[23, 41, "DRUG"]]},
    {"text": "Participants must be ambulatory.", "label": [[21, 31, "DEMOGRAPHIC"]]},
    {"text": "No history of gastrointestinal perforation.", "label": [[15, 42, "CONDITION"]]},
    {"text": "Prior exposure to any investigational drug within 30 days.", "label": [[21, 42, "DRUG"], [51, 61, "VALUE"]]},
    {"text": "Hemoglobin A1c (HbA1c) < 8.0%.", "label": [[0, 23, "LAB_TEST"], [24, 30, "VALUE"]]},
    {"text": "Must be able to undergo MRI scans.", "label": [[21, 30, "PROCEDURE"]]},
    {"text": "Patient is not a candidate for surgical resection.", "label": [[11, 14, "OPERATOR"], [31, 49, "PROCEDURE"]]},
    {"text": "History of hypersensitivity to any of the study medications.", "label": [[11, 28, "CONDITION"], [40, 58, "DRUG"]]},
    {"text": "Must have a negative pregnancy test at screening.", "label": [[13, 33, "LAB_TEST"]]},
    {"text": "Systolic blood pressure must be < 140 mmHg and diastolic < 90 mmHg.", "label": [[0, 24, "LAB_TEST"], [34, 44, "VALUE"], [49, 69, "VALUE"]]},
    {"text": "Patient has no known allergies to contrast media.", "label": [[16, 25, "CONDITION"], [29, 43, "DRUG"]]},
    {"text": "Must provide a tissue block for central pathology review.", "label": [[16, 28, "PROCEDURE"]]},
    {"text": "Women of childbearing potential must use effective contraception.", "label": [[0, 31, "DEMOGRAPHIC"], [41, 62, "PROCEDURE"]]},
    {"text": "No other concurrent malignancy other than non-melanoma skin cancer.", "label": [[3, 29, "CONDITION"], [39, 66, "CONDITION"]]},
    {"text": "Patient has an indwelling catheter or central line.", "label": [[12, 47, "PROCEDURE"]]},
    {"text": "Serum sodium level within institutional normal range.", "label": [[0, 19, "LAB_TEST"], [28, 56, "VALUE"]]},
    {"text": "Must be willing to complete a daily electronic diary.", "label": [[25, 52, "PROCEDURE"]]},
    {"text": "No history of significant psychiatric disorders.", "label": [[15, 45, "CONDITION"]]},
    {"text": "Participants must not be institutionalized for a psychiatric illness.", "label": [[21, 38, "DEMOGRAPHIC"], [45, 65, "CONDITION"]]},
    {"text": "Fasting triglycerides < 150 mg/dL.", "label": [[8, 25, "LAB_TEST"], [26, 37, "VALUE"]]},
    {"text": "No known diagnosis of celiac disease.", "label": [[20, 34, "CONDITION"]]},
    {"text": "Must be able to self-administer subcutaneous injections.", "label": [[16, 56, "PROCEDURE"]]},
    {"text": "History of deep vein thrombosis or pulmonary embolism.", "label": [[11, 31, "CONDITION"], [35, 55, "CONDITION"]]},
    {"text": "Patient is not a candidate for high-dose chemotherapy.", "label": [[11, 14, "OPERATOR"], [31, 52, "DRUG"]]},
    {"text": "Must be at least 18 years old and weigh at least 50 kg.", "label": [[12, 28, "DEMOGRAPHIC"], [38, 52, "VALUE"]]},
    {"text": "No active substance abuse within the past year.", "label": [[3, 26, "CONDITION"], [36, 49, "VALUE"]]},
    {"text": "Patient has provided informed consent for genetic testing.", "label": [[12, 30, "PROCEDURE"], [35, 51, "PROCEDURE"]]},
    {"text": "Presence of a solitary bone lesion is acceptable.", "label": [[18, 36, "CONDITION"]]},
    {"text": "Must have a Karnofsky Performance Score >= 70%.", "label": [[13, 40, "LAB_TEST"], [41, 47, "VALUE"]]},
    {"text": "Exclusion for patients requiring chronic oxygen therapy.", "label": [[30, 52, "PROCEDURE"]]},
    {"text": "International Normalized Ratio (INR) must be < 1.5.", "label": [[0, 31, "LAB_TEST"], [41, 47, "VALUE"]]},
    {"text": "No prior exposure to anthracyclines.", "label": [[3, 28, "DRUG"]]},
    {"text": "Patient must be willing to undergo an eye examination.", "label": [[32, 48, "PROCEDURE"]]},
    {"text": "History of inflammatory bowel disease.", "label": [[11, 39, "CONDITION"]]},
    {"text": "Must not have received any blood transfusion within 28 days of randomization.", "label": [[28, 45, "PROCEDURE"], [54, 75, "VALUE"]]},
    {"text": "Patient is able to communicate in English.", "label": [[12, 38, "DEMOGRAPHIC"]]},
    {"text": "No evidence of clinically significant hearing loss.", "label": [[14, 46, "CONDITION"]]},
    {"text": "Participant must have a valid driver's license.", "label": [[25, 46, "DEMOGRAPHIC"]]},
    {"text": "No history of symptomatic congestive heart failure (CHF).", "label": [[15, 51, "CONDITION"]]},
    {"text": "Patient must have access to a telephone.", "label": [[25, 41, "DEMOGRAPHIC"]]},
    {"text": "Exclusion for subjects with a known coagulopathy.", "label": [[32, 44, "CONDITION"]]},
    {"text": "Must abstain from grapefruit products during the study.", "label": [[16, 34, "DRUG"]]},
    {"text": "Patient has a life expectancy of less than 6 months.", "label": [[12, 32, "OPERATOR"], [33, 46, "VALUE"]]},
    {"text": "No known intolerance to intravenous contrast agents.", "label": [[20, 52, "DRUG"]]},
    {"text": "Must complete a quality of life questionnaire.", "label": [[16, 49, "PROCEDURE"]]},
    {"text": "History of uncontrolled seizures is not permitted.", "label": [[11, 32, "CONDITION"]]},
    {"text": "Patient has a Body Surface Area (BSA) >= 1.5 m2.", "label": [[12, 33, "LAB_TEST"], [34, 43, "VALUE"]]},
    {"text": "No active dental issues requiring surgery.", "label": [[9, 22, "CONDITION"], [33, 40, "PROCEDURE"]]},
    {"text": "Must be able to tolerate oral medications.", "label": [[21, 37, "DRUG"]]},
    {"text": "Patient has not received prior radiation to the brain.", "label": [[27, 49, "PROCEDURE"]]},
    {"text": "No history of clinically significant arrhythmia.", "label": [[15, 49, "CONDITION"]]},
    {"text": "Must have adequate organ function.", "label": [[16, 30, "LAB_TEST"]]},
    {"text": "Patient is not currently lactating.", "label": [[11, 14, "OPERATOR"], [25, 34, "DEMOGRAPHIC"]]},
    {"text": "No known immunodeficiency.", "label": [[9, 26, "CONDITION"]]},
    {"text": "Willing to adhere to the study visit schedule.", "label": [[24, 48, "PROCEDURE"]]},
    {"text": "History of diverticulitis is an exclusion criterion.", "label": [[11, 26, "CONDITION"]]},
    {"text": "Patient must have a primary caregiver.", "label": [[21, 37, "DEMOGRAPHIC"]]},
    {"text": "No history of interstitial pneumonia.", "label": [[15, 38, "CONDITION"]]},
    {"text": "Must have discontinued aspirin for at least 7 days.", "label": [[24, 31, "DRUG"], [41, 54, "VALUE"]]},
    {"text": "Patient is legally an adult.", "label": [[12, 28, "DEMOGRAPHIC"]]},
    {"text": "No active, uncontrolled bacterial or fungal infections.", "label": [[3, 50, "CONDITION"]]},
    {"text": "Must be willing to undergo a skin biopsy.", "label": [[28, 40, "PROCEDURE"]]},
    {"text": "History of osteoporosis is permitted.", "label": [[11, 23, "CONDITION"]]},
    {"text": "Patient must have a negative test for tuberculosis.", "label": [[28, 48, "LAB_TEST"]]},
    {"text": "No significant trauma within the past 4 weeks.", "label": [[3, 23, "CONDITION"], [33, 48, "VALUE"]]},
    {"text": "Must be able to understand and sign the informed consent form.", "label": [[35, 58, "PROCEDURE"]]},
    {"text": "Patient has a history of non-compliance with medical regimens.", "label": [[26, 61, "CONDITION"]]},
    {"text": "No ongoing treatment with systemic steroids for more than 2 weeks.", "label": [[11, 35, "DRUG"], [41, 57, "VALUE"]]},
    {"text": "Must have a confirmed diagnosis of Alzheimer's disease.", "label": [[31, 50, "CONDITION"]]},
    {"text": "Patient must have a stable residence.", "label": [[21, 37, "DEMOGRAPHIC"]]},
    {"text": "No known allergy to peanuts.", "label": [[20, 27, "CONDITION"]]},
    {"text": "Subject is able to perform activities of daily living.", "label": [[21, 55, "PROCEDURE"]]},
    {"text": "History of pancreatitis within the last year.", "label": [[11, 23, "CONDITION"], [33, 48, "VALUE"]]},
    {"text": "Patient must not have a pacemaker.", "label": [[21, 30, "PROCEDURE"]]},
    {"text": "No active suicidal ideation.", "label": [[3, 25, "CONDITION"]]},
    {"text": "Must have a screening ECG without clinically significant abnormalities.", "label": [[22, 25, "PROCEDURE"], [34, 71, "CONDITION"]]},
    {"text": "Patient is a candidate for autologous stem cell transplantation.", "label": [[25, 62, "PROCEDURE"]]},
    {"text": "No clinically significant gastrointestinal disorders.", "label": [[3, 50, "CONDITION"]]},
    {"text": "Must not be a recipient of a solid organ transplant.", "label": [[28, 50, "PROCEDURE"]]},
    {"text": "Patient has a negative urine drug screen.", "label": [[12, 33, "LAB_TEST"]]},
    {"text": "No history of uncontrolled hyperlipidemia.", "label": [[15, 41, "CONDITION"]]},
    {"text": "Subject must be fluent in Spanish.", "label": [[25, 41, "DEMOGRAPHIC"]]},
    {"text": "No active dermatologic conditions that could interfere with study assessments.", "label": [[9, 33, "CONDITION"]]},
    {"text": "Patient must have an estimated glomerular filtration rate (eGFR) > 60 mL/min/1.73m2.", "label": [[21, 62, "LAB_TEST"], [63, 83, "VALUE"]]},
    {"text": "No known porphyria.", "label": [[9, 18, "CONDITION"]]},
    {"text": "Must be willing to avoid excessive sun exposure.", "label": [[24, 48, "PROCEDURE"]]},
    {"text": "History of psychiatric hospitalization is an exclusion.", "label": [[11, 40, "CONDITION"]]},
    {"text": "Patient agrees to not donate blood during the study.", "label": [[20, 32, "PROCEDURE"]]}
]

SYSTEM_PROMPT = """
You are an expert AI assistant specializing in clinical trial eligibility criteria. Your task is to perform Named Entity Recognition (NER) on a given criterion text.

You must identify and label entities using ONLY the following 7 labels:
- CONDITION
- DRUG
- LAB_TEST
- VALUE
- OPERATOR
- PROCEDURE
- DEMOGRAPHIC

For each "INPUT" text, you must return ONLY a single, valid JSON object as the "OUTPUT". The JSON object must have two keys: "text" (the original input text) and "label" (a list of annotations, where each annotation is a list of [start_index, end_index, "LABEL_NAME"]). Do not add any explanations or markdown.
"""

def format_few_shot_examples(examples: list) -> str:
    """Formats the examples into a string for the prompt."""
    formatted_examples = []
    for ex in examples:
        output_str = json.dumps({"text": ex["text"], "label": ex["label"]})
        formatted_examples.append(f"INPUT: {ex['text']}\nOUTPUT: {output_str}")
    return "\n---\n".join(formatted_examples)

def load_model_and_tokenizer():
    """Loads the Hugging Face model and tokenizer in a memory-efficient way."""
    print(f"-> Loading model: {MODEL_ID}. This will download ~14 GB and may take a while...")
    
    # Configure quantization to load the model in 4-bit, saving memory.
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Automatically use GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("✅ Model and tokenizer loaded successfully.")
    return model, tokenizer

def annotate_text_with_hf_model(model, tokenizer, text_to_annotate: str, prompt_base: str):
    """Generates an annotation for a single text using the local Hugging Face model."""
    full_prompt = prompt_base + text_to_annotate + "\nOUTPUT:"
    
    messages = [{"role": "user", "content": full_prompt}]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = tokenized_chat.to(model.device)

    try:
        outputs = model.generate(inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        
        # Clean the output to find the first valid JSON object
        json_match = re.search(r'\{.*\}', decoded_output)
        if json_match:
            json_text = json_match.group(0)
            return json.loads(json_text)
        else:
            print(f"   - Warning: No valid JSON found in model output for: '{text_to_annotate[:50]}...'")
            return {"text": text_to_annotate, "label": []}
            
    except Exception as e:
        print(f"   - Warning: Model generation error occurred: {e}")
        return {"text": text_to_annotate, "label": []}

def main():
    """Main function to run the automated annotation with a local HF model."""
    print("--- Starting Automated Annotation using Hugging Face Model ---")

    model, tokenizer = load_model_and_tokenizer()

    few_shot_prompt_part = format_few_shot_examples(FEW_SHOT_EXAMPLES)
    prompt_base = f"{SYSTEM_PROMPT}\n\nHere are the examples:\n{few_shot_prompt_part}\n\n---\nNow, perform the task on this new input.\nINPUT: "

    if not os.path.exists(ANNOTATION_SAMPLE_FILE):
        print(f"❌ Error: Annotation sample file not found at '{ANNOTATION_SAMPLE_FILE}'.")
        return

    with open(ANNOTATION_SAMPLE_FILE, 'r', encoding='utf-8') as f:
        texts_to_process = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts_to_process)} criteria to annotate.")

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
        for text in tqdm(texts_to_process, desc="Annotating criteria via local model"):
            result = annotate_text_with_hf_model(model, tokenizer, text, prompt_base)
            if result and "text" in result and "label" in result:
                f.write(json.dumps(result) + '\n')

    print(f"\n✅ Automated annotation complete!")
    print(f"   Output file saved to '{OUTPUT_JSONL_FILE}'.")
    print("\n   You can now proceed to the next step: '03_prepare_ner_data.py'.")

if __name__ == "__main__":
    main()
