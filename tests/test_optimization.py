import pytest
import numpy as np
from src.optimization import f_lambda_objective, greedy, lazy_greedy, stochastic_greedy, exact_search

def test_greedy_equals_lazy_greedy():
    np.random.seed(42)
    N = 20
    K = 5
    lambd = 0.5
    
    r_tilde = np.random.rand(N)
    features = np.random.rand(N, 10)
    # Cosine sim matrix
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    sim_matrix = features.dot(features.T)
    
    S_greedy = greedy(r_tilde, sim_matrix, K, lambd)
    S_lazy = lazy_greedy(r_tilde, sim_matrix, K, lambd)
    
    # Tie breaking can cause sets to be returned in different orders or different items if tied.
    # We will test that they yield the exact same objective value, and with high probability the same set.
    assert set(S_greedy) == set(S_lazy)

def test_submodularity():
    np.random.seed(42)
    N = 10
    lambd = 0.5
    
    r_tilde = np.random.rand(N)
    features = np.random.rand(N, 5)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    sim_matrix = features.dot(features.T)
    
    # Submodularity: F(A U {e}) - F(A) >= F(B U {e}) - F(B) for A subset B
    A = [0, 1]
    B = [0, 1, 2, 3]
    e = 4
    K = 5
    
    f_A = f_lambda_objective(A, r_tilde, sim_matrix, K, lambd)
    f_A_e = f_lambda_objective(A + [e], r_tilde, sim_matrix, K, lambd)
    gain_A = f_A_e - f_A
    
    f_B = f_lambda_objective(B, r_tilde, sim_matrix, K, lambd)
    f_B_e = f_lambda_objective(B + [e], r_tilde, sim_matrix, K, lambd)
    gain_B = f_B_e - f_B
    
    # allow some float tolerance
    assert gain_A >= gain_B - 1e-9

def test_monotonicity():
    np.random.seed(42)
    N = 10
    lambd = 0.5
    
    r_tilde = np.random.rand(N)
    features = np.random.rand(N, 5)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    sim_matrix = features.dot(features.T)
    
    # Monotonicity: F(A) <= F(B) when A subset B
    A = [0, 1]
    B = [0, 1, 2]
    K = 5
    
    f_A = f_lambda_objective(A, r_tilde, sim_matrix, K, lambd)
    f_B = f_lambda_objective(B, r_tilde, sim_matrix, K, lambd)
    
    assert f_B >= f_A - 1e-9

def test_greedy_guarantee():
    np.random.seed(42)
    N = 10
    K = 3
    lambd = 0.5
    
    r_tilde = np.random.rand(N)
    features = np.random.rand(N, 5)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    sim_matrix = features.dot(features.T)
    
    S_exact = exact_search(r_tilde, sim_matrix, K, lambd)
    f_exact = f_lambda_objective(S_exact, r_tilde, sim_matrix, K, lambd)
    
    S_greedy = greedy(r_tilde, sim_matrix, K, lambd)
    f_greedy = f_lambda_objective(S_greedy, r_tilde, sim_matrix, K, lambd)
    
    # F(S_greedy) >= (1 - 1/e) F(S_exact)
    assert f_greedy >= (1 - 1/np.e) * f_exact - 1e-9
    
def test_lambda_zero():
    # If lambda = 0, objective is purely modular, so greedy should pick exact top K
    N = 10
    K = 3
    lambd = 0.0
    r_tilde = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0])
    sim_matrix = np.eye(N)
    
    S_greedy = greedy(r_tilde, sim_matrix, K, lambd)
    assert set(S_greedy) == {1, 3, 5}
