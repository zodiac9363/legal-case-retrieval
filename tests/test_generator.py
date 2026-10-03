import pytest
from src.generator import LegalCorpusGenerator

def test_seed_reproducibility():
    gen1 = LegalCorpusGenerator(seed=42)
    docs1, doc_aspects1, _, _ = gen1.generate_corpus(num_docs=100, redundancy=0.25, skew='low')
    
    gen2 = LegalCorpusGenerator(seed=42)
    docs2, doc_aspects2, _, _ = gen2.generate_corpus(num_docs=100, redundancy=0.25, skew='low')
    
    for d1, d2 in zip(docs1, docs2):
        assert d1['id'] == d2['id']
        assert d1['text'] == d2['text']
        assert doc_aspects1[d1['id']] == doc_aspects2[d2['id']]

def test_duplicates_inherit_aspects():
    gen = LegalCorpusGenerator(seed=123)
    docs, doc_aspects, doc_weights, cluster_ids = gen.generate_corpus(num_docs=100, redundancy=0.5, skew='low')
    
    # Check that documents in the same cluster share the same aspects
    cluster_to_aspects = {}
    for doc in docs:
        doc_id = doc['id']
        cid = cluster_ids[doc_id]
        if cid not in cluster_to_aspects:
            cluster_to_aspects[cid] = doc_aspects[doc_id]
        else:
            assert cluster_to_aspects[cid] == doc_aspects[doc_id]

def test_relevance_consistency():
    gen = LegalCorpusGenerator(seed=111)
    docs, doc_aspects, doc_weights, cluster_ids = gen.generate_corpus(num_docs=100, redundancy=0.0, skew='low')
    queries = gen.generate_queries(num_queries=10, single_aspect=False)
    qrels = gen.compute_relevance(queries, doc_aspects, doc_weights, tau=0.25)
    
    for q in queries:
        qid = q['id']
        q_aspects = set(q['aspects'])
        
        for doc_id, rel_score in qrels.get(qid, {}).items():
            d_aspects = doc_aspects[doc_id]
            d_weights = doc_weights[doc_id]
            
            # Check that there is indeed an overlap of at least rel_score aspects with weight >= tau
            overlap_count = 0
            for a, w in zip(d_aspects, d_weights):
                if a in q_aspects and w >= 0.25:
                    overlap_count += 1
            
            assert overlap_count == rel_score
