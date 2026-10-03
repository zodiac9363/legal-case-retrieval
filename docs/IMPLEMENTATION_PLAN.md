# Implementation Plan and Task List

## M0: Project Setup & Architecture
- [x] Initialize git repository.
- [x] Create Python project structure (`src/`, `tests/`, `data/`, `results/`, `docs/`).
- [x] Setup `requirements.txt` with pinned dependencies (numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, flask, pytest, rank_bm25).
- [x] Create `Makefile` with CLI entry points: `data`, `baselines`, `main`, `ablations`, `theory`, `figures`, `report`, `app`, `test`, `all`.
- [x] Create `README.md` with honest framing and instructions.

## M1: Synthetic Legal Benchmark Generator (Tier 1)
- [x] Implement statistical lexicon and area/aspect definitions.
- [x] Implement document and query generation with aspect skew and redundancy support.
- [x] Generate default dataset (R1-R3 regimes).
- [x] Write `docs/DATASET_CARD.md`.
- [x] Add generator tests (invariants, duplicates, seed reproducibility).

## M2: Retrieval Baselines & Metrics (Tier 1)
- [x] Implement preprocessing pipeline (TF-IDF, boilerplate stop-lists).
- [x] Implement custom vectorized BM25 Okapi and unit test against `rank_bm25` (must match to 1e-6).
- [x] Implement MMR re-ranking baseline.
- [x] Implement all evaluation metrics: P@K, R@K, MAP@K, NDCG@K, S-recall@K, α-NDCG@K, duplicate-free ratio.
- [x] Unit test metrics on hand-computed toy cases.

## M3: Submodular Optimization Methods (Tier 1)
- [x] Implement $F_\lambda$ objective evaluating relevance and facility location.
- [x] Implement exact brute-force, Greedy, Lazy Greedy, and Stochastic Greedy.
- [x] Test equivalence of Greedy and Lazy Greedy.
- [x] Test properties (submodularity and monotonicity on random instances).

## M4: Cross-Validation & Execution Pipeline (Tier 1)
- [x] Implement 5-fold query-level cross-validation without test leakage.
- [x] Build tuning loop maximizing α-NDCG@10.
- [x] Connect configuration loading (YAML) to the main pipeline.

## M5: Core Experiments (Tier 1 / Tier 2)
- [ ] E1: Main method comparisons on default regime.
- [ ] E2: Factorial study (R1–R3), include R2/R3 negative controls, 5 seeds (mean +/- std), Holm-corrected Wilcoxon, bootstrap CIs.
- [ ] E3: λ sweep & Pareto plots (add MMR curve and NDCG@K, label "analysis only").
- [ ] E4: Sensitivity to K, N.
- [ ] E5: Ablations. Run relevance-only / diversity-only / combined. Diversity variants (LSA kernel, saturated coverage, optional log-det), normalisation choices. Use LSA on synthetic corpus, MiniLM only on rewritten subset.
- [ ] Save CSVs and plot figures, log seed, config, package versions, git hash.

## M6: Analysis Experiments (Tier 2 / Tier 3)
- [ ] E6: Optimization analysis (add >=500 random instances (N<=30, K<=6), runtime vs N for greedy/lazy/stoch, randomized tests).
- [ ] E8: Error analysis. Bucket ALL queries (helps/neutral/hurts), fixed rule for examples including harmful cases; neutral wording.
- [ ] E9: Predictive analysis. Features (aspect count, dominance ratio, pool redundancy, score gap, query length). Targets (dS-recall, dNDCG). Ridge + shallow tree, report cross-validated R2 honestly.

## Post-M6 Sprint
- [ ] E7: Robustness (label noise, query length, corpus size, lexical overlap).
- [ ] E10: Component grid.

## M7: Theory & Documentation (Tier 1)
- [x] Write `docs/THEORY.md` with proofs of submodularity and optimization explanations.
- [x] Ensure docstrings and type hints are applied across `src/`.

## M8: Web Interface (Tier 2)
- [x] Implement Flask backend loading precomputed indices.
- [x] Create plain HTML/CSS/JS frontend (Bootstrap CDN ok).
- [x] Build Search/Compare view with λ slider, aspect badges, and duplicate markers.
- [x] Build Experiment Dashboard integrating Chart.js for results plots.
- [x] Implement Method Page.

## M9: Report & Final Validation (Tier 1)
- [ ] Automate generating `docs/REPORT.md` (paper style) pulling from `results/`.
- [ ] Verify `make all` end-to-end functionality.
- [ ] Final walkthrough check against Tier requirements and Definition of Done.
