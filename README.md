# Diverse Legal Case Retrieval Using Submodular Optimization

## Honest Framing Statement
- Submodular maximisation, facility location and MMR-style diversification are NOT new. Do not claim a new algorithm or a new benchmark.
- The contribution is a rigorous, reproducible, controlled study of WHEN and WHY submodular set selection helps or hurts BM25 in legal case retrieval, using a synthetic benchmark whose redundancy and aspect structure can be controlled, plus an optimisation analysis of the greedy solver and a predictive analysis of per-query gains.
- Results on synthetic data are NOT claims about real legal corpora (e.g., COLIEE). State this limitation prominently.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full pipeline:
   ```bash
   make all
   ```
3. Run the web interface:
   ```bash
   make app
   ```

## Reproduction Guide
To reproduce the results step by step:
- `make data`: Generates the synthetic benchmark data.
- `make baselines`: Runs retrieval baselines.
- `make main`: Runs main submodular methods.
- `make ablations`: Runs ablation studies.
- `make theory`: Runs theoretical and property tests.
- `make figures`: Generates analysis figures.
- `make report`: Compiles the final report in `docs/REPORT.md`.
