from flask import Flask, jsonify, request, render_template
import os
import json
import numpy as np
from src.pipeline import load_dataset, load_config
from src.baselines import build_tfidf_similarity, VectorizedBM25, run_mmr
from src.optimization import greedy
from src.metrics import alpha_ndcg_at_k, ndcg_at_k, precision_at_k, s_recall_at_k, duplicate_free_ratio

app = Flask(__name__)

# Global state to hold indices
state = {}

def init_app():
    config = load_config("config.yaml")
    path = os.path.join(config['data']['base_path'], config['data']['default_regime'])
    if not os.path.exists(path):
        return
        
    corpus, doc_ids, doc_aspects, cluster_ids, queries, qrels = load_dataset(path)
    state['corpus'] = corpus
    state['doc_ids'] = doc_ids
    state['doc_id_to_idx'] = {did: i for i, did in enumerate(doc_ids)}
    state['doc_aspects'] = doc_aspects
    state['cluster_ids'] = cluster_ids
    state['queries'] = queries
    state['qrels'] = qrels
    
    # Precompute TF-IDF & Sim Matrix
    sim_matrix, _ = build_tfidf_similarity(corpus)
    state['sim_matrix'] = sim_matrix
    
    # Precompute BM25 with default params
    bm25 = VectorizedBM25(k1=1.5, b=0.75)
    bm25.fit(corpus)
    state['bm25'] = bm25

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/queries')
def api_queries():
    if 'queries' not in state:
        return jsonify([])
    res = []
    for q in state['queries']:
        res.append({
            'id': q['id'],
            'preview': q['text'][:100] + '...',
            'aspects': q['aspects']
        })
    return jsonify(res)

@app.route('/api/search')
def api_search():
    qid = request.args.get('qid')
    method = request.args.get('method', 'proposed')
    K = int(request.args.get('K', 10))
    N = int(request.args.get('N', 100))
    lambd = float(request.args.get('lambda', 0.5))
    mu = float(request.args.get('mu', 0.5))
    custom_text = request.args.get('custom_text', None)
    
    if custom_text:
        q_text = custom_text
        q_aspects = []
    else:
        q_obj = next((q for q in state['queries'] if q['id'] == qid), None)
        if not q_obj:
            return jsonify({'error': 'Query not found'})
        q_text = q_obj['text']
        q_aspects = q_obj['aspects']
        
    bm25 = state['bm25']
    scores = bm25.get_scores(q_text)
    
    N = min(N, len(scores))
    top_n_idx = np.argsort(scores)[::-1][:N]
    pool_sim = state['sim_matrix'][np.ix_(top_n_idx, top_n_idx)]
    pool_r_tilde = scores[top_n_idx]
    
    if pool_r_tilde.max() > pool_r_tilde.min():
        pool_r_tilde = (pool_r_tilde - pool_r_tilde.min()) / (pool_r_tilde.max() - pool_r_tilde.min())
    else:
        pool_r_tilde = np.ones(len(pool_r_tilde))
        
    if method == 'proposed':
        sel_idx = greedy(pool_r_tilde, pool_sim, K, lambd)
    elif method == 'mmr':
        sel_idx = run_mmr(pool_r_tilde, pool_sim, K, mu)
    elif method == 'bm25':
        sel_idx = list(range(min(K, len(top_n_idx))))
    else:
        sel_idx = []
        
    results = []
    for i in sel_idx:
        real_idx = top_n_idx[i]
        doc_id = state['doc_ids'][real_idx]
        results.append({
            'id': doc_id,
            'score': float(scores[real_idx]),
            'snippet': state['corpus'][real_idx][:150] + '...',
            'aspects': state['doc_aspects'].get(doc_id, []),
            'cluster_id': state['cluster_ids'].get(doc_id, -1)
        })
        
    # Calculate metrics
    metrics = {}
    if not custom_text:
        doc_ids_ranked = [r['id'] for r in results]
        q_rels = state['qrels'].get(qid, {})
        metrics = {
            'P@K': float(precision_at_k(doc_ids_ranked, q_rels, K)),
            'NDCG@K': float(ndcg_at_k(doc_ids_ranked, q_rels, K)),
            'S-Recall@K': float(s_recall_at_k(doc_ids_ranked, K, q_aspects, state['doc_aspects'])),
            'alpha-NDCG@K': float(alpha_ndcg_at_k(doc_ids_ranked, K, q_aspects, state['doc_aspects'])),
            'Dup-Free': float(duplicate_free_ratio(doc_ids_ranked, state['cluster_ids']))
        }
        
    return jsonify({
        'results': results,
        'metrics': metrics
    })

@app.route('/api/summary')
def api_summary():
    return jsonify({'message': 'Dashboard placeholder'})

@app.route('/api/corpus-stats')
def api_corpus_stats():
    return jsonify({
        'num_docs': len(state.get('corpus', [])),
        'num_queries': len(state.get('queries', []))
    })

if __name__ == '__main__':
    init_app()
    app.run(debug=True, port=5000)
