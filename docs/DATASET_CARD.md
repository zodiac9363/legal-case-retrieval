# Synthetic Legal Corpus Dataset Card

## Purpose
This dataset is designed to provide a controlled environment for testing submodular optimization and diverse retrieval methods in legal case retrieval. Real legal corpora often lack comprehensive aspect-level ground truth and are subject to copyright or privacy constraints. This synthetic benchmark allows for rigorous, reproducible studies by precisely controlling vocabulary, legal issues (aspects), redundancy, and issue skew.

## Generative Process
- **Lexicon**: ~8,000 synthetic types with Zipfian background frequencies and a shared legal boilerplate list (~150 phrases).
- **Aspects**: 8 legal areas (contract, tort, property, criminal, family, labour, IP, tax) × 12 issues each = 96 distinct aspects.
- **Documents**: Each document mixes 1-3 aspects using Dirichlet weights. Documents include ~15% boilerplate and ~15% background noise.
- **Redundancy**: A fraction of documents ($\rho$) are generated as near-duplicates (70-90% token overlap with a base document).
- **Queries**: Generated similarly from 3-5 aspects (or 1 aspect for negative controls).

## Parameters
- `seed`: Controls deterministic generation.
- `redundancy` ($\rho$): Fraction of duplicate cases (0, 0.25, 0.5, 0.75).
- `skew`: Controls the aspect distribution ('low' for balanced, 'high' for dominated aspects).

## Known Biases and Limitations
- Circularity risk: The generator uses a bag-of-words assumption that is also relied upon by TF-IDF similarity.
- Results on this synthetic dataset do NOT directly translate to claims about real legal corpora (e.g., COLIEE).
- The legal boilerplate and vocabulary are entirely synthetic.

## Intended Use
To evaluate when and why submodular set selection improves retrieval quality (specifically coverage and diversity metrics) over simple independent ranking baselines like BM25.
