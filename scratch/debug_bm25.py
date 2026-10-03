from rank_bm25 import BM25Okapi
import numpy as np
from src.baselines import VectorizedBM25

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly",
    "a quick brown dog outpaces a quick fox"
]
query = "quick brown fox"
    
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25_ref = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

bm25_custom = VectorizedBM25(k1=1.5, b=0.75)
bm25_custom.fit(corpus)

print("REF idf:")
print(bm25_ref.idf)
print("REF doc_len:")
print(bm25_ref.doc_len)
print("REF avgdl:")
print(bm25_ref.avgdl)

q_words = query.split()
for w in q_words:
    print(f"word: {w}")
    print(f"  ref idf: {bm25_ref.idf.get(w)}")
    idx = bm25_custom.vectorizer.vocabulary_.get(w)
    print(f"  custom idf: {bm25_custom.idf[idx]}")
