import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline import cross_validate, load_config

def run_e1(config):
    print("Running E1: Main comparison")
    path = os.path.join(config['data']['base_path'], config['data']['default_regime'])
    
    # Run Baselines
    res_bm25, _ = cross_validate(path, config, method='bm25')
    res_mmr, _ = cross_validate(path, config, method='mmr')
    res_prop, _ = cross_validate(path, config, method='proposed')
    
    df = pd.DataFrame([
        {'Method': 'BM25', **res_bm25},
        {'Method': 'MMR', **res_mmr},
        {'Method': 'Proposed', **res_prop}
    ])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/e1_main_comparison.csv", index=False)
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.melt(id_vars='Method'), x='variable', y='value', hue='Method')
    plt.title("Main Comparison on Default Regime")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/e1_main_comparison.png")
    plt.close()

def run_all_experiments():
    config = load_config("config.yaml")
    run_e1(config)
    # E2-E9 stubs for full run
    print("Experiments E2-E9 stubs finished (placeholders for full run).")
    
if __name__ == "__main__":
    run_all_experiments()
