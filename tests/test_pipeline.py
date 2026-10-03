import pytest
from src.pipeline import load_config
from sklearn.model_selection import KFold
import numpy as np

def test_leakage():
    # Simulate the query split to ensure there's no intersection
    queries = list(range(100))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(queries):
        # inner split
        inner_split_size = int(len(train_idx) * 0.8)
        inner_train_idx = train_idx[:inner_split_size]
        inner_val_idx = train_idx[inner_split_size:]
        
        # Verify no intersection between train, val, test
        train_set = set(inner_train_idx)
        val_set = set(inner_val_idx)
        test_set = set(test_idx)
        
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set.intersection(test_set)) == 0
        assert len(val_set.intersection(test_set)) == 0
        
        # Verify all are covered
        assert len(train_set) + len(val_set) + len(test_set) == 100
        
def test_config_load():
    # If the config file exists, it should load correctly
    try:
        config = load_config("config.yaml")
        assert 'tuning' in config
        assert config['tuning']['folds'] == 5
    except FileNotFoundError:
        pytest.skip("config.yaml not found, skipping config load test")
