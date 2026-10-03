import pytest
import numpy as np
from rank_bm25 import BM25Okapi
from src.baselines import VectorizedBM25, run_mmr

def test_bm25_agreement():
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "never jump over the lazy dog quickly",
        "a quick brown dog outpaces a quick fox"
    ]
    query = "quick brown fox"
    
    # rank_bm25 uses tokenized lists
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25_ref = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    ref_scores = bm25_ref.get_scores(query.split(" "))
    
    # Custom vectorized implementation
    bm25_custom = VectorizedBM25(k1=1.5, b=0.75)
    bm25_custom.fit(corpus)
    custom_scores = bm25_custom.get_scores(query)
    
    np.testing.assert_allclose(custom_scores, ref_scores, rtol=1e-6, atol=1e-6)

def test_run_mmr():
    bm25_scores = np.array([0.5, 0.8, 0.2, 0.9])
    
    # sim matrix: doc 3 and doc 1 are highly similar
    sim_matrix = np.array([
        [1.0, 0.1, 0.2, 0.1],
        [0.1, 1.0, 0.3, 0.9],
        [0.2, 0.3, 1.0, 0.1],
        [0.1, 0.9, 0.1, 1.0]
    ])
    
    # With mu=1.0, should just be max bm25 score sort -> [3, 1, 0, 2]
    selected_mu1 = run_mmr(bm25_scores, sim_matrix, 4, mu=1.0)
    assert selected_mu1 == [3, 1, 0, 2]
    
    # With mu=0.1, doc 3 is selected first (best r_tilde), then doc 1 has high sim to doc 3 so it gets penalized
    # Let's see: r_tilde = [0.428, 0.857, 0.0, 1.0]
    # Selected: [3]
    # Doc 1 score = 0.1 * 0.857 - 0.9 * 0.9 = -0.724
    # Doc 0 score = 0.1 * 0.428 - 0.9 * 0.1 = -0.047
    # Doc 2 score = 0.1 * 0.0 - 0.9 * 0.1 = -0.09
    # Next should be doc 0
    selected_mu01 = run_mmr(bm25_scores, sim_matrix, 4, mu=0.1)
    assert selected_mu01[0] == 3
    assert selected_mu01[1] == 0
