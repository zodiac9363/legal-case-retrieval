import pytest
from src.metrics import precision_at_k, ndcg_at_k, s_recall_at_k, duplicate_free_ratio

def test_precision_at_k():
    ranked_list = ['d1', 'd2', 'd3']
    qrels = {'d1': 1, 'd2': 0, 'd4': 2}
    
    assert precision_at_k(ranked_list, qrels, 1) == 1.0
    assert precision_at_k(ranked_list, qrels, 2) == 0.5
    assert precision_at_k(ranked_list, qrels, 3) == 0.3333333333333333
    
def test_ndcg_at_k():
    ranked_list = ['d1', 'd2', 'd3']
    qrels = {'d1': 2, 'd2': 1, 'd3': 0}
    
    # dcg@1: (2^2 - 1) / log2(2) = 3.0
    # idcg@1: 3.0
    assert ndcg_at_k(ranked_list, qrels, 1) == 1.0
    
    # dcg@2: 3.0 + (2^1 - 1) / log2(3) = 3 + 1/1.58 = 3.63...
    assert ndcg_at_k(ranked_list, qrels, 2) == 1.0
    
def test_s_recall_at_k():
    ranked_list = ['d1', 'd2']
    query_aspects = [1, 2, 3]
    doc_aspects = {'d1': [1], 'd2': [2, 4]}
    
    assert s_recall_at_k(ranked_list, 1, query_aspects, doc_aspects) == 1/3
    assert s_recall_at_k(ranked_list, 2, query_aspects, doc_aspects) == 2/3
    
def test_duplicate_free_ratio():
    ranked_list = ['d1', 'd2', 'd3']
    cluster_ids = {'d1': 1, 'd2': 1, 'd3': 2}
    
    assert duplicate_free_ratio(ranked_list, cluster_ids) == 2/3
