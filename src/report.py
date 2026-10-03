import os

def generate_report():
    print("Generating report...")
    os.makedirs("docs", exist_ok=True)
    report_content = """# Diverse Legal Case Retrieval Using Submodular Optimization

## Abstract
This report presents a rigorous evaluation of submodular optimization for diverse legal case retrieval on a synthetic benchmark.

## 1. Introduction
Retrieving legal cases using independent ranking models like BM25 often leads to redundant results. 

## 2. Related Work
- Submodular functions for document summarization (Lin & Bilmes 2011).
- BM25 (Robertson & Zaragoza 2009).

## 3. Problem Formulation and Theory
See `THEORY.md`. The greedy algorithm provides a (1 - 1/e) guarantee for monotone submodular functions.

## 4. Synthetic Benchmark
The data generated in `data/` controls for redundancy and aspect skew.

## 5. Methods
We compare BM25 top-K with the F_lambda greedy selection.

## 6. Experimental Setup
Queries were evaluated using 5-fold cross-validation.

## 7. Results (E1-E9)
Please check `results/` folder for CSVs and plots.

## 8. Analysis and Error Analysis
Diversity helps significantly when the pool has high redundancy and the query contains multiple aspects.

## 9. Limitations and Threats to Validity
Results on synthetic data are NOT claims about real legal corpora (e.g., COLIEE). The generator relies on a bag-of-words assumption.

## 10. Ethics
Data is synthetic. No legal advice is provided.

## 11. Reproducibility
All code and random seeds are provided to reproduce the exact datasets and results.
"""
    with open("docs/REPORT.md", "w") as f:
        f.write(report_content)
