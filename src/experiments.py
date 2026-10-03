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

def run_e7b_robustness(config):
    print("Running E7b: LLM Robustness (Rewritten vs BoW)")
    import scipy.stats as stats
    
    path_rewritten = "data/R1_rewritten"
    path_bow = "data/R1_rewritten_bow"
    
    if not os.path.exists(path_rewritten) or not os.path.exists(path_bow):
        print(f"Skipping E7b: Rewritten dataset {path_rewritten} not found. Did you run llm_rewriter.py?")
        return
        
    _, res_prop_rewritten = cross_validate(path_rewritten, config, method='proposed')
    _, res_prop_bow = cross_validate(path_bow, config, method='proposed')
    
    # Paired differences (assumes same qids in both sets)
    from src.metrics import alpha_ndcg_at_k
    _, _, doc_aspects, _, queries, qrels = load_dataset(path_rewritten)
    
    diffs = []
    qids = [q['id'] for q in queries]
    
    for q in queries:
        qid = q['id']
        rl_r = res_prop_rewritten.get(qid, [])
        rl_b = res_prop_bow.get(qid, [])
        q_a = q['aspects']
        
        a_ndcg_r = alpha_ndcg_at_k(rl_r, 10, q_a, doc_aspects)
        a_ndcg_b = alpha_ndcg_at_k(rl_b, 10, q_a, doc_aspects)
        
        diffs.append(a_ndcg_r - a_ndcg_b)
        
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci = stats.t.interval(0.95, len(diffs)-1, loc=mean_diff, scale=stats.sem(diffs)) if len(diffs) > 1 else (np.nan, np.nan)
    
    print(f"E7b Results: Mean difference (Rewritten - BoW) in alpha-NDCG@10 = {mean_diff:.4f}, 95% CI: {ci}")
    
    with open("results/e7b_robustness.txt", "w") as f:
        f.write(f"E7b Results: Mean difference (Rewritten - BoW) in alpha-NDCG@10 = {mean_diff:.4f}, 95% CI: {ci}\n")

def run_all_experiments():
    config = load_config("config.yaml")
    run_e1(config)
    run_e7b_robustness(config)
    print("Experiments finished.")
    
if __name__ == "__main__":
    run_all_experiments()
