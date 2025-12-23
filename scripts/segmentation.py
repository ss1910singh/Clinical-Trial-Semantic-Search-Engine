import re
from typing import List
from tqdm import tqdm

def segment_text_batch(texts: List[str], nlp_model) -> List[List[str]]:
    cleaned_texts_for_nlp = []
    print("Performing fast pre-cleaning of text...")
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            cleaned_texts_for_nlp.append("")
            continue
        
        text = text.replace('•', '\n-').replace('*', '\n-')
        lines = text.split('\n')
        
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned_line = re.sub(r'^\s*-\s*|\d+\.\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        
        cleaned_texts_for_nlp.append(" ".join(cleaned_lines))

    final_results = []
    print("Starting efficient NLP batch processing for sentence segmentation...")
    for doc in tqdm(nlp_model.pipe(cleaned_texts_for_nlp, batch_size=50), total=len(cleaned_texts_for_nlp), desc="Segmenting criteria"):
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        final_results.append(sentences)
        
    return final_results