import numpy as np

def precision_at_k(ranked_list, qrels, k):
    if not ranked_list:
        return 0.0
    k = min(k, len(ranked_list))
    relevant_count = sum(1 for doc_id in ranked_list[:k] if doc_id in qrels and qrels[doc_id] > 0)
    return relevant_count / k

def recall_at_k(ranked_list, qrels, k):
    if not qrels:
        return 0.0
    k = min(k, len(ranked_list))
    relevant_count = sum(1 for doc_id in ranked_list[:k] if doc_id in qrels and qrels[doc_id] > 0)
    return relevant_count / len(qrels)

def map_at_k(ranked_list, qrels, k):
    if not qrels:
        return 0.0
    k = min(k, len(ranked_list))
    num_relevant = len(qrels)
    avg_precision = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id in qrels and qrels[doc_id] > 0:
            relevant_count += 1
            avg_precision += relevant_count / (i + 1)
            
    return avg_precision / min(k, num_relevant) if num_relevant > 0 else 0.0

def ndcg_at_k(ranked_list, qrels, k):
    if not qrels:
        return 0.0
    k = min(k, len(ranked_list))
    
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / np.log2(i + 2)
            
    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0

def s_recall_at_k(ranked_list, k, query_aspects, doc_aspects_dict):
    """Fraction of query aspects covered by the top-K documents."""
    if not query_aspects:
        return 0.0
    k = min(k, len(ranked_list))
    
    covered_aspects = set()
    for doc_id in ranked_list[:k]:
        doc_aspects = doc_aspects_dict.get(doc_id, [])
        covered_aspects.update(a for a in doc_aspects if a in query_aspects)
        
    return len(covered_aspects) / len(query_aspects)

def alpha_ndcg_at_k(ranked_list, k, query_aspects, doc_aspects_dict, alpha=0.5):
    """
    Clarke et al. 2008 alpha-nDCG.
    Relevance of doc d to aspect a is 1 if it covers a, else 0.
    """
    if not query_aspects:
        return 0.0
    k = min(k, len(ranked_list))
    query_aspects = set(query_aspects)
    
    dcg = 0.0
    aspect_counts = {a: 0 for a in query_aspects}
    
    for i, doc_id in enumerate(ranked_list[:k]):
        doc_aspects = set(doc_aspects_dict.get(doc_id, []))
        gain = 0.0
        for a in query_aspects:
            if a in doc_aspects:
                gain += (1.0 - alpha) ** aspect_counts[a]
                aspect_counts[a] += 1
        dcg += gain / np.log2(i + 2)
        
    # Approximate IDCG with a greedy upper bound
    idcg = 0.0
    ideal_aspect_counts = {a: 0 for a in query_aspects}
    available_docs = list(doc_aspects_dict.keys())
    
    for i in range(k):
        best_gain = -1
        best_doc_idx = -1
        for j, doc_id in enumerate(available_docs):
            doc_aspects = set(doc_aspects_dict[doc_id])
            gain = sum((1.0 - alpha) ** ideal_aspect_counts[a] for a in query_aspects if a in doc_aspects)
            if gain > best_gain:
                best_gain = gain
                best_doc_idx = j
                
        if best_doc_idx == -1 or best_gain == 0:
            break
            
        doc_id = available_docs.pop(best_doc_idx)
        doc_aspects = set(doc_aspects_dict[doc_id])
        for a in query_aspects:
            if a in doc_aspects:
                ideal_aspect_counts[a] += 1
                
        idcg += best_gain / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def intra_list_distance(ranked_list, similarity_matrix, doc_to_idx):
    if len(ranked_list) < 2:
        return 0.0
    
    total_dist = 0.0
    pairs = 0
    for i in range(len(ranked_list)):
        for j in range(i + 1, len(ranked_list)):
            idx_i = doc_to_idx.get(ranked_list[i])
            idx_j = doc_to_idx.get(ranked_list[j])
            if idx_i is not None and idx_j is not None:
                sim = similarity_matrix[idx_i, idx_j]
                total_dist += (1.0 - sim)
                pairs += 1
                
    return total_dist / pairs if pairs > 0 else 0.0

def duplicate_free_ratio(ranked_list, cluster_ids):
    if not ranked_list:
        return 0.0
    distinct_clusters = set(cluster_ids.get(doc_id) for doc_id in ranked_list)
    return len(distinct_clusters) / len(ranked_list)

def pool_recall_at_n(pool, qrels):
    if not qrels:
        return 0.0
    relevant_in_pool = sum(1 for doc_id in pool if doc_id in qrels and qrels[doc_id] > 0)
    return relevant_in_pool / len(qrels)
