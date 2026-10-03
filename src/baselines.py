import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class VectorizedBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        
    def fit(self, corpus):
        # corpus is a list of strings
        self.X = self.vectorizer.fit_transform(corpus)
        self.doc_len = self.X.sum(axis=1).A1
        self.avgdl = self.doc_len.mean()
        self.N = self.X.shape[0]
        
        # Compute IDF
        # Compute IDF exactly as rank_bm25 BM25Okapi does:
        # idf = log((N - n_q + 0.5) / (n_q + 0.5))
        df = np.bincount(self.X.indices, minlength=self.X.shape[1])
        idf = np.log((self.N - df + 0.5) / (df + 0.5))
        
        # rank_bm25 ATIRE variant negative idf floor
        df_mask = df > 0
        if df_mask.any():
            avg_idf = idf[df_mask].mean()
            eps = 0.25 * avg_idf
            idf[df_mask & (idf < 0)] = eps
        self.idf = idf
        
    def get_scores(self, query):
        query_vec = self.vectorizer.transform([query]).tocoo()
        q_indices = query_vec.col
        
        if len(q_indices) == 0:
            return np.zeros(self.N)
            
        scores = np.zeros(self.N)
        
        for q_idx in q_indices:
            # documents containing the term
            col = self.X[:, q_idx].tocoo()
            doc_indices = col.row
            tf = col.data
            
            # BM25 formula
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_indices] / self.avgdl)
            
            scores[doc_indices] += self.idf[q_idx] * (num / den)
            
        return scores

def run_mmr(bm25_scores, sim_matrix, K, mu=0.5):
    """
    MMR re-ranking.
    score(d) = μ·r̃(d) − (1 − μ)·max_{j∈S} s(d,j)
    """
    N = len(bm25_scores)
    K = min(K, N)
    if K == 0:
        return []
        
    # r̃(d) = min-max normalized BM25 score
    min_score = bm25_scores.min()
    max_score = bm25_scores.max()
    if max_score > min_score:
        r_tilde = (bm25_scores - min_score) / (max_score - min_score)
    else:
        r_tilde = np.ones(N)
        
    unselected = set(range(N))
    selected = []
    
    # First item is just argmax of r_tilde
    first_idx = np.argmax(r_tilde)
    selected.append(first_idx)
    unselected.remove(first_idx)
    
    while len(selected) < K and unselected:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in unselected:
            # max similarity to already selected
            max_sim = max(sim_matrix[idx, j] for j in selected)
            score = mu * r_tilde[idx] - (1 - mu) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
                
        selected.append(best_idx)
        unselected.remove(best_idx)
        
    return selected

def build_tfidf_similarity(corpus):
    vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2', lowercase=True)
    X = vectorizer.fit_transform(corpus)
    # Cosine similarity for L2 normalized vectors is X * X.T
    sim_matrix = X.dot(X.T).toarray()
    # ensure non-negative
    sim_matrix[sim_matrix < 0] = 0
    return sim_matrix, vectorizer
