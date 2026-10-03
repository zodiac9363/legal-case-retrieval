import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import hashlib
import time
import requests
import numpy as np
from src.pipeline import load_dataset

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
TEMPERATURE = 0.3
MAX_TOKENS = 400

def get_boilerplate_terms():
    return {
        "the", "of", "and", "in", "to", "a", "is", "that", "for", "it", "as", "was",
        "court", "held", "judge", "appellant", "respondent", "appeal", "dismissed",
        "statute", "section", "act", "law", "rule", "plaintiff", "defendant", "trial",
        "evidence", "fact", "issue", "finding", "therefore", "however", "furthermore",
        "stated", "argued", "claimed", "judgment", "order", "decision", "rights",
        "party", "parties", "proceedings", "jurisdiction", "matter", "case", "action",
        "liability", "damages", "claim", "duty", "breach", "contract", "tort", "property",
        "criminal", "family", "ip", "tax", "constitutional", "administrative", "civil",
        "pursuant", "accordance", "respect", "regarding", "concerning", "relating"
    }

def get_aspect_terms(text, boilerplate):
    words = text.lower().split()
    return [w for w in words if w not in boilerplate]

def get_cache_key(text, model, temperature):
    raw = f"{text}_{model}_{temperature}"
    return hashlib.md5(raw.encode()).hexdigest()

def rewrite_text(text, boilerplate, cache_dir, is_query=False):
    cache_key = get_cache_key(text, MODEL_NAME, TEMPERATURE)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)["rewritten"]
            
    aspect_terms = get_aspect_terms(text, boilerplate)
    
    prompt = (
        f"Rewrite the following {'search query' if is_query else 'legal case document'} "
        "into a fluent, grammatically correct legal-style paragraph. "
        "You MUST include as many of the original aspect-distinctive keywords as possible. "
        "Do not change the underlying legal topics.\n\n"
        f"Original text:\n{text}\n"
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS
                },
                "keep_alive": "30m"
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()
            rewritten = response.json().get("response", "").strip()
            
            # Validation
            orig_len = len(text.split())
            new_len = len(rewritten.split())
            
            if not (0.5 * orig_len <= new_len <= 2.0 * orig_len):
                print(f"Length validation failed (orig: {orig_len}, new: {new_len}). Retrying...")
                continue
                
            new_aspect_terms = get_aspect_terms(rewritten, boilerplate)
            new_aspect_terms_set = set(new_aspect_terms)
            orig_aspect_terms_set = set(aspect_terms)
            
            if len(orig_aspect_terms_set) > 0:
                survival_rate = len(orig_aspect_terms_set.intersection(new_aspect_terms_set)) / len(orig_aspect_terms_set)
                if survival_rate < 0.7:
                    print(f"Survival validation failed ({survival_rate:.2f} < 0.70). Retrying...")
                    continue
            
            # Cache and return
            with open(cache_path, "w") as f:
                json.dump({"rewritten": rewritten}, f)
                
            return rewritten
            
        except Exception as e:
            print(f"API Error: {e}")
            time.sleep(2 ** attempt)
            
    # Fallback to original if we fail to satisfy conditions after retries
    print("Failed to rewrite after retries, falling back to original.")
    return text

def create_rewritten_dataset(base_dataset_path, output_dataset_path, num_docs=10):
    corpus, doc_ids, doc_aspects, cluster_ids, queries, qrels = load_dataset(base_dataset_path)
    
    # Setup cache
    cache_dir = "data/llm_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dataset_path, exist_ok=True)
    
    boilerplate = get_boilerplate_terms()
    
    # 1. Select subset of docs
    np.random.seed(42)
    selected_doc_indices = np.random.choice(len(corpus), min(num_docs, len(corpus)), replace=False)
    selected_doc_ids = set([doc_ids[i] for i in selected_doc_indices])
    
    # 2. Select queries that have relevant docs in this subset
    selected_queries = []
    selected_qrels = {}
    for q in queries:
        qid = q['id']
        if qid in qrels:
            # Check if any relevant doc is in selected_doc_ids
            rel_docs = [doc for doc, rel in qrels[qid].items() if rel > 0 and doc in selected_doc_ids]
            if rel_docs:
                selected_queries.append(q)
                selected_qrels[qid] = {doc: rel for doc, rel in qrels[qid].items() if doc in selected_doc_ids}
                
    subset_dataset_path = output_dataset_path + "_bow"
    os.makedirs(subset_dataset_path, exist_ok=True)
    
    # 3. Save BoW Docs and Rewrite Docs
    print(f"Rewriting {len(selected_doc_indices)} documents via Ollama (qwen2.5:3b)...")
    with open(os.path.join(output_dataset_path, "corpus.jsonl"), "w") as f_rewritten, \
         open(os.path.join(subset_dataset_path, "corpus.jsonl"), "w") as f_bow:
        for idx in selected_doc_indices:
            doc_id = doc_ids[idx]
            text = corpus[idx]
            
            # Save BoW
            bow_obj = {
                "id": doc_id,
                "text": text,
                "aspects": doc_aspects[doc_id],
                "cluster_id": cluster_ids[doc_id]
            }
            f_bow.write(json.dumps(bow_obj) + "\n")
            
            # Rewrite and save
            rewritten = rewrite_text(text, boilerplate, cache_dir, is_query=False)
            doc_obj = {
                "id": doc_id,
                "text": rewritten,
                "aspects": doc_aspects[doc_id],
                "cluster_id": cluster_ids[doc_id]
            }
            f_rewritten.write(json.dumps(doc_obj) + "\n")
            
    # 4. Save BoW Queries and Rewrite Queries
    print(f"Rewriting {len(selected_queries)} queries...")
    with open(os.path.join(output_dataset_path, "queries.jsonl"), "w") as f_rewritten, \
         open(os.path.join(subset_dataset_path, "queries.jsonl"), "w") as f_bow:
        for q in selected_queries:
            # Save BoW
            f_bow.write(json.dumps(q) + "\n")
            
            # Rewrite and save
            rewritten = rewrite_text(q['text'], boilerplate, cache_dir, is_query=True)
            q_obj = {
                "id": q['id'],
                "text": rewritten,
                "aspects": q['aspects']
            }
            f_rewritten.write(json.dumps(q_obj) + "\n")
            
    # 5. Copy over qrels
    with open(os.path.join(output_dataset_path, "qrels.json"), "w") as f_rewritten, \
         open(os.path.join(subset_dataset_path, "qrels.json"), "w") as f_bow:
        json.dump(selected_qrels, f_rewritten)
        json.dump(selected_qrels, f_bow)
        
    print(f"Successfully generated rewritten dataset at {output_dataset_path} and BoW subset at {subset_dataset_path}")
    return True

if __name__ == "__main__":
    base_path = "data/R1_rho0.5_skew_low_seed0"
    out_path = "data/R1_rewritten"
    create_rewritten_dataset(base_path, out_path, num_docs=10)
