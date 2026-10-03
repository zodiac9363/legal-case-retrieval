# Implementation Plan and Task List

## M0: Project Setup & Architecture
- [ ] Initialize git repository.
- [ ] Create Python project structure (`src/`, `tests/`, `data/`, `results/`, `docs/`).
- [ ] Setup `requirements.txt` with pinned dependencies (numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, flask, pytest, rank_bm25).
- [ ] Create `Makefile` with CLI entry points: `data`, `baselines`, `main`, `ablations`, `theory`, `figures`, `report`, `app`, `test`, `all`.
- [ ] Create `README.md` with honest framing and instructions.

## M1: Synthetic Legal Benchmark Generator (Tier 1)
- [ ] Implement statistical lexicon and area/aspect definitions.
- [ ] Implement document and query generation with aspect skew and redundancy support.
- [ ] Generate default dataset (R1-R3 regimes).
- [ ] Write `docs/DATASET_CARD.md`.
- [ ] Add generator tests (invariants, duplicates, seed reproducibility).

## M2: Retrieval Baselines & Metrics (Tier 1)
- [ ] Implement preprocessing pipeline (TF-IDF, boilerplate stop-lists).
- [ ] Implement custom vectorized BM25 Okapi and unit test against `rank_bm25` (must match to 1e-6).
- [ ] Implement MMR re-ranking baseline.
- [ ] Implement all evaluation metrics: P@K, R@K, MAP@K, NDCG@K, S-recall@K, α-NDCG@K, duplicate-free ratio.
- [ ] Unit test metrics on hand-computed toy cases.

## M3: Submodular Optimization Methods (Tier 1)
- [ ] Implement $F_\lambda$ objective evaluating relevance and facility location.
- [ ] Implement exact brute-force, Greedy, Lazy Greedy, and Stochastic Greedy.
- [ ] Test equivalence of Greedy and Lazy Greedy.
- [ ] Test properties (submodularity and monotonicity on random instances).

## M4: Cross-Validation & Execution Pipeline (Tier 1)
- [ ] Implement 5-fold query-level cross-validation without test leakage.
- [ ] Build tuning loop maximizing α-NDCG@10.
- [ ] Connect configuration loading (YAML) to the main pipeline.

## M5: Core Experiments E1–E5 (Tier 1 / Tier 2)
- [ ] E1: Main method comparisons on default regime.
- [ ] E2: Factorial study (R1–R3).
- [ ] E3: λ sweep & Pareto frontier plots.
- [ ] E4: Sensitivity to K, N.
- [ ] E5: Ablations (TF-IDF vs LSA, normalization, saturation coverage).
- [ ] Save CSVs and plot figures.

## M6: Analysis Experiments E6–E9 (Tier 2 / Tier 3)
- [ ] E6: Optimization analysis (approximation ratio, evaluation counts).
- [ ] E7: Robustness (label noise, lengths).
- [ ] E8: Error analysis and qualitative examples.
- [ ] E9: Predictive analysis (per-query features -> ΔS-recall).

## M7: Theory & Documentation (Tier 1)
- [ ] Write `docs/THEORY.md` with proofs of submodularity and optimization explanations.
- [ ] Ensure docstrings and type hints are applied across `src/`.

## M8: Web Interface (Tier 2)
- [ ] Implement Flask backend loading precomputed indices.
- [ ] Create plain HTML/CSS/JS frontend (Bootstrap CDN ok).
- [ ] Build Search/Compare view with λ slider, aspect badges, and duplicate markers.
- [ ] Build Experiment Dashboard integrating Chart.js for results plots.
- [ ] Implement Method Page.

## M9: Report & Final Validation (Tier 1)
- [ ] Automate generating `docs/REPORT.md` (paper style) pulling from `results/`.
- [ ] Verify `make all` end-to-end functionality.
- [ ] Final walkthrough check against Tier requirements and Definition of Done.
