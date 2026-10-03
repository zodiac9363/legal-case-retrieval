import os
import json
from src.generator import LegalCorpusGenerator
from tqdm import tqdm

def save_dataset(path, documents, doc_aspects, doc_aspect_weights, cluster_ids, queries, qrels):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "corpus.jsonl"), "w") as f:
        for doc in documents:
            doc_id = doc['id']
            d = {
                'id': doc_id,
                'text': doc['text'],
                'aspects': doc_aspects[doc_id],
                'weights': doc_aspect_weights[doc_id],
                'cluster_id': cluster_ids[doc_id]
            }
            f.write(json.dumps(d) + "\n")
            
    with open(os.path.join(path, "queries.jsonl"), "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
            
    with open(os.path.join(path, "qrels.json"), "w") as f:
        json.dump(qrels, f, indent=2)

def generate_all_datasets():
    print("Generating datasets...")
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)
    
    regimes = []
    # R1: rho in {0, 0.25, 0.5, 0.75} x skew in {low, high}
    for rho in [0, 0.25, 0.5, 0.75]:
        for skew in ['low', 'high']:
            regimes.append({'name': f"R1_rho{rho}_skew_{skew}", 'rho': rho, 'skew': skew, 'single_aspect': False})
            
    # R2: NEGATIVE CONTROL: single-aspect queries, rho = 0
    regimes.append({'name': "R2_negative_control_1", 'rho': 0.0, 'skew': 'low', 'single_aspect': True})
    
    # R3: NEGATIVE CONTROL 2: high rho but single-aspect queries
    regimes.append({'name': "R3_negative_control_2", 'rho': 0.75, 'skew': 'low', 'single_aspect': True})
    
    for regime in tqdm(regimes, desc="Regimes"):
        for seed in range(5):
            dataset_name = f"{regime['name']}_seed{seed}"
            path = os.path.join(base_path, dataset_name)
            
            gen = LegalCorpusGenerator(seed=42+seed)
            documents, doc_aspects, doc_aspect_weights, cluster_ids = gen.generate_corpus(
                num_docs=100,  # small default version
                redundancy=regime['rho'], 
                skew=regime['skew']
            )
            queries = gen.generate_queries(num_queries=10, single_aspect=regime['single_aspect'])
            qrels = gen.compute_relevance(queries, doc_aspects, doc_aspect_weights)
            
            save_dataset(path, documents, doc_aspects, doc_aspect_weights, cluster_ids, queries, qrels)

if __name__ == "__main__":
    generate_all_datasets()
