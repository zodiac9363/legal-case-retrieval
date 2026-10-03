import json
import os
import numpy as np
import yaml
from sklearn.model_selection import KFold
from src.baselines import VectorizedBM25, build_tfidf_similarity, run_mmr
from src.optimization import greedy
from src.metrics import alpha_ndcg_at_k, ndcg_at_k, precision_at_k, s_recall_at_k, duplicate_free_ratio
from itertools import product
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path):
    corpus = []
    doc_ids = []
    doc_aspects = {}
    cluster_ids = {}
    
    with open(os.path.join(dataset_path, "corpus.jsonl"), "r") as f:
        for line in f:
            d = json.loads(line)
            corpus.append(d['text'])
            doc_ids.append(d['id'])
            doc_aspects[d['id']] = d['aspects']
            cluster_ids[d['id']] = d.get('cluster_id', -1)
            
    queries = []
    with open(os.path.join(dataset_path, "queries.jsonl"), "r") as f:
        for line in f:
            queries.append(json.loads(line))
            
    with open(os.path.join(dataset_path, "qrels.json"), "r") as f:
        qrels = json.load(f)
        
    return corpus, doc_ids, doc_aspects, cluster_ids, queries, qrels

def evaluate_run(ranked_lists, queries, qrels, doc_aspects, cluster_ids, K=10):
    metrics = {
        'alpha_ndcg': [],
        'ndcg': [],
        'precision': [],
        's_recall': [],
        'dup_free_ratio': []
    }
    
    for q in queries:
        qid = q['id']
        ranked_list = ranked_lists.get(qid, [])
        q_rels = qrels.get(qid, {})
        q_aspects = q['aspects']
        
        metrics['alpha_ndcg'].append(alpha_ndcg_at_k(ranked_list, K, q_aspects, doc_aspects))
        metrics['ndcg'].append(ndcg_at_k(ranked_list, q_rels, K))
        metrics['precision'].append(precision_at_k(ranked_list, q_rels, K))
        metrics['s_recall'].append(s_recall_at_k(ranked_list, K, q_aspects, doc_aspects))
        metrics['dup_free_ratio'].append(duplicate_free_ratio(ranked_list, cluster_ids))
        
    return {k: np.mean(v) for k, v in metrics.items()}

def cross_validate(dataset_path, config, method='proposed'):
    corpus, doc_ids, doc_aspects, cluster_ids, queries, qrels = load_dataset(dataset_path)
    
    # Precompute similarities
    sim_matrix, _ = build_tfidf_similarity(corpus)
    
    # Setup CV
    kf = KFold(n_splits=config['tuning']['folds'], shuffle=True, random_state=42)
    
    all_results = {}
    q_indices = np.arange(len(queries))
    
    bm25_cache = {}
    for k1 in config['grids']['bm25']['k1']:
        for b in config['grids']['bm25']['b']:
            bm25 = VectorizedBM25(k1=k1, b=b)
            bm25.fit(corpus)
            bm25_cache[(k1, b)] = bm25
            
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(q_indices)):
        print(f"Fold {fold_idx + 1}/{config['tuning']['folds']}")
        
        # Inner split
        inner_split_size = int(len(train_idx) * config['tuning']['inner_split'])
        inner_train_idx = train_idx[:inner_split_size]
        inner_val_idx = train_idx[inner_split_size:]
        
        inner_val_queries = [queries[i] for i in inner_val_idx]
        test_queries = [queries[i] for i in test_idx]
        
        best_val_score = -1
        best_params = {}
        
        # Grid Search
        param_grid = []
        if method == 'proposed':
            param_grid = product(
                config['grids']['bm25']['k1'],
                config['grids']['bm25']['b'],
                config['grids']['pool']['N'],
                config['grids']['proposed']['lambd']
            )
        elif method == 'mmr':
            param_grid = product(
                config['grids']['bm25']['k1'],
                config['grids']['bm25']['b'],
                config['grids']['pool']['N'],
                config['grids']['mmr']['mu']
            )
        elif method == 'bm25':
            param_grid = product(
                config['grids']['bm25']['k1'],
                config['grids']['bm25']['b'],
                config['grids']['pool']['N'],
                [0.0]
            )
            
        for params in param_grid:
            k1, b, N, param_val = params
            bm25 = bm25_cache[(k1, b)]
            
            val_ranked_lists = {}
            for q in inner_val_queries:
                q_text = q['text']
                scores = bm25.get_scores(q_text)
                top_n_idx = np.argsort(scores)[::-1][:N]
                
                pool_sim = sim_matrix[np.ix_(top_n_idx, top_n_idx)]
                pool_r_tilde = scores[top_n_idx]
                if pool_r_tilde.max() > pool_r_tilde.min():
                    pool_r_tilde = (pool_r_tilde - pool_r_tilde.min()) / (pool_r_tilde.max() - pool_r_tilde.min())
                else:
                    pool_r_tilde = np.ones(len(pool_r_tilde))
                    
                K = 10 # Eval K
                if method == 'proposed' or method == 'bm25':
                    sel_idx = greedy(pool_r_tilde, pool_sim, K, param_val)
                    val_ranked_lists[q['id']] = [doc_ids[top_n_idx[i]] for i in sel_idx]
                elif method == 'mmr':
                    sel_idx = run_mmr(pool_r_tilde, pool_sim, K, param_val)
                    val_ranked_lists[q['id']] = [doc_ids[top_n_idx[i]] for i in sel_idx]
                    
            val_metrics = evaluate_run(val_ranked_lists, inner_val_queries, qrels, doc_aspects, cluster_ids, K=10)
            score = val_metrics[config['tuning']['target_metric']]
            
            if score > best_val_score:
                best_val_score = score
                best_params = {'k1': k1, 'b': b, 'N': N, 'param_val': param_val}
                
        # Evaluate on test set with best params
        print(f"Best params for fold {fold_idx + 1}: {best_params}")
        bm25 = bm25_cache[(best_params['k1'], best_params['b'])]
        
        test_ranked_lists = {}
        for q in test_queries:
            scores = bm25.get_scores(q['text'])
            N = best_params['N']
            top_n_idx = np.argsort(scores)[::-1][:N]
            
            pool_sim = sim_matrix[np.ix_(top_n_idx, top_n_idx)]
            pool_r_tilde = scores[top_n_idx]
            if pool_r_tilde.max() > pool_r_tilde.min():
                pool_r_tilde = (pool_r_tilde - pool_r_tilde.min()) / (pool_r_tilde.max() - pool_r_tilde.min())
            else:
                pool_r_tilde = np.ones(len(pool_r_tilde))
                
            K = 10
            pv = best_params['param_val']
            if method == 'proposed' or method == 'bm25':
                sel_idx = greedy(pool_r_tilde, pool_sim, K, pv)
            elif method == 'mmr':
                sel_idx = run_mmr(pool_r_tilde, pool_sim, K, pv)
            test_ranked_lists[q['id']] = [doc_ids[top_n_idx[i]] for i in sel_idx]
            
        all_results.update(test_ranked_lists)
        
    final_metrics = evaluate_run(all_results, queries, qrels, doc_aspects, cluster_ids, K=10)
    print(f"CV final metrics for {method}: {final_metrics}")
    return final_metrics, all_results
