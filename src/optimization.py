import numpy as np
import itertools
import heapq

def f_lambda_objective(S, r_tilde, sim_matrix, K, lambd):
    """
    Computes the F_lambda submodular objective for a given set S.
    S: list or set of indices.
    r_tilde: 1D numpy array of size N (normalized relevance scores).
    sim_matrix: 2D numpy array (N x N) of pairwise similarities.
    """
    if not S:
        return 0.0
        
    S = list(S)
    N = len(r_tilde)
    
    # Relevance term
    rel_term = np.sum(r_tilde[S]) / K
    
    # Coverage / facility location term
    max_sims = np.max(sim_matrix[:, S], axis=1)
    cov_term = np.sum(max_sims) / N
    
    return (1 - lambd) * rel_term + lambd * cov_term

def greedy(r_tilde, sim_matrix, K, lambd):
    N = len(r_tilde)
    S = []
    unselected = set(range(N))
    
    # Maintain the maximum similarity to S for each item in the pool
    m_i = np.zeros(N)
    
    for _ in range(K):
        best_gain = -float('inf')
        best_e = -1
        
        for e in unselected:
            # marginal relevance
            gain_rel = r_tilde[e] / K
            
            # marginal coverage: sum(max(0, sim_matrix[i, e] - m_i))
            gain_cov = np.sum(np.maximum(0, sim_matrix[:, e] - m_i)) / N
            
            gain = (1 - lambd) * gain_rel + lambd * gain_cov
            
            if gain > best_gain:
                best_gain = gain
                best_e = e
                
        if best_e == -1 or best_gain <= 1e-9:
            break
            
        S.append(best_e)
        unselected.remove(best_e)
        m_i = np.maximum(m_i, sim_matrix[:, best_e])
        
    return S

def lazy_greedy(r_tilde, sim_matrix, K, lambd):
    N = len(r_tilde)
    S = []
    
    # We maintain max similarities
    m_i = np.zeros(N)
    
    # Max-heap for lazy evaluation (store negative gains because heapq is min-heap)
    # Entry: (-upper_bound_gain, doc_id, iteration_last_updated)
    heap = []
    
    # Initialize heap with marginal gains of single elements
    for e in range(N):
        gain_rel = r_tilde[e] / K
        gain_cov = np.sum(sim_matrix[:, e]) / N
        gain = (1 - lambd) * gain_rel + lambd * gain_cov
        heapq.heappush(heap, (-gain, e, 0))
        
    iteration = 0
    while len(S) < K and heap:
        neg_gain, e, last_updated = heapq.heappop(heap)
        gain = -neg_gain
        
        if last_updated == iteration:
            # This bound is tight; it's the max!
            if gain <= 1e-9:
                break
            S.append(e)
            m_i = np.maximum(m_i, sim_matrix[:, e])
            iteration += 1
        else:
            # Re-evaluate
            gain_rel = r_tilde[e] / K
            gain_cov = np.sum(np.maximum(0, sim_matrix[:, e] - m_i)) / N
            new_gain = (1 - lambd) * gain_rel + lambd * gain_cov
            heapq.heappush(heap, (-new_gain, e, iteration))
            
    return S

def stochastic_greedy(r_tilde, sim_matrix, K, lambd, epsilon=0.1):
    N = len(r_tilde)
    S = []
    unselected = list(range(N))
    m_i = np.zeros(N)
    
    sample_size = int(np.ceil((N / K) * np.log(1 / epsilon)))
    sample_size = min(sample_size, N)
    
    for _ in range(K):
        if not unselected:
            break
            
        sample = np.random.choice(unselected, size=min(sample_size, len(unselected)), replace=False)
        
        best_gain = -float('inf')
        best_e = -1
        
        for e in sample:
            gain_rel = r_tilde[e] / K
            gain_cov = np.sum(np.maximum(0, sim_matrix[:, e] - m_i)) / N
            gain = (1 - lambd) * gain_rel + lambd * gain_cov
            
            if gain > best_gain:
                best_gain = gain
                best_e = e
                
        if best_e == -1 or best_gain <= 1e-9:
            break
            
        S.append(best_e)
        unselected.remove(best_e)
        m_i = np.maximum(m_i, sim_matrix[:, best_e])
        
    return S

def exact_search(r_tilde, sim_matrix, K, lambd):
    """
    Brute-force exactly optimal subset of size K.
    Returns (best_subset, max_val).
    """
    N = len(r_tilde)
    best_subset = None
    max_val = -float('inf')
    
    for subset in itertools.combinations(range(N), min(K, N)):
        val = f_lambda_objective(list(subset), r_tilde, sim_matrix, K, lambd)
        if val > max_val:
            max_val = val
            best_subset = list(subset)
            
    return best_subset
