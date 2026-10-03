import numpy as np
import json
from collections import defaultdict
import os

class LegalCorpusGenerator:
    def __init__(self, seed=42, vocab_size=8000, num_areas=8, issues_per_area=12,
                 lexical_overlap=0.05, within_area_overlap=0.25):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.num_areas = num_areas
        self.issues_per_area = issues_per_area
        self.num_aspects = num_areas * issues_per_area
        
        self.lexical_overlap = lexical_overlap
        self.within_area_overlap = within_area_overlap
        
        self.boilerplate_vocab = []
        self.aspect_vocabs = {}
        self._build_lexicon()
        
    def _build_lexicon(self):
        # 1. Boilerplate (approx 150 words)
        num_boilerplate = 150
        self.boilerplate_vocab = [f"boiler_{i}" for i in range(num_boilerplate)]
        self.boilerplate_probs = self.rng.dirichlet(np.ones(num_boilerplate) * 0.1)
        
        # 2. Base vocabulary pool (Zipfian)
        self.base_vocab = [f"word_{i}" for i in range(self.vocab_size)]
        
        # Area specific pools
        area_pool_size = self.vocab_size // self.num_areas
        area_pools = [self.base_vocab[i*area_pool_size:(i+1)*area_pool_size] for i in range(self.num_areas)]
        
        for area_id in range(self.num_areas):
            area_vocab = area_pools[area_id]
            
            # Shared words within the area (~25%)
            num_shared = int(len(area_vocab) * self.within_area_overlap)
            area_shared = area_vocab[:num_shared]
            
            for issue_id in range(self.issues_per_area):
                aspect_id = area_id * self.issues_per_area + issue_id
                
                # Each aspect gets ~60 distinctive terms
                # For simplicity, we just pick a random subset of words from the area pool
                distinctive_pool = area_vocab[num_shared:]
                aspect_words = list(self.rng.choice(distinctive_pool, size=45, replace=False))
                
                # Add area shared words (25% of final aspect vocab -> ~15 words)
                aspect_words.extend(self.rng.choice(area_shared, size=15, replace=False))
                
                # Add cross-area shared words (~5% -> 3 words)
                other_areas_words = [w for w in self.base_vocab if w not in area_vocab]
                aspect_words.extend(self.rng.choice(other_areas_words, size=3, replace=False))
                
                # Create Zipfian distribution for these words
                ranks = np.arange(1, len(aspect_words) + 1)
                probs = 1.0 / (ranks ** 1.5)
                probs /= probs.sum()
                
                self.aspect_vocabs[aspect_id] = {
                    'words': np.array(aspect_words),
                    'probs': probs
                }

    def generate_document(self, num_tokens=600, aspects=None, aspect_weights=None):
        if aspects is None:
            # Pick 1-3 aspects
            num_aspects = self.rng.integers(1, 4)
            aspects = self.rng.choice(self.num_aspects, size=num_aspects, replace=False)
            
        if aspect_weights is None:
            aspect_weights = self.rng.dirichlet(np.ones(len(aspects)))
            
        doc_tokens = []
        
        # ~15% boilerplate, 15% noise, 70% aspect-specific
        num_boilerplate = int(num_tokens * 0.15)
        num_noise = int(num_tokens * 0.15)
        num_aspect = num_tokens - num_boilerplate - num_noise
        
        if num_boilerplate > 0:
            doc_tokens.extend(self.rng.choice(self.boilerplate_vocab, p=self.boilerplate_probs, size=num_boilerplate))
        if num_noise > 0:
            doc_tokens.extend(self.rng.choice(self.base_vocab, size=num_noise))
            
        for aspect, weight in zip(aspects, aspect_weights):
            num_aspect_tokens = int(num_aspect * weight)
            if num_aspect_tokens > 0:
                vocab_info = self.aspect_vocabs[aspect]
                doc_tokens.extend(self.rng.choice(vocab_info['words'], p=vocab_info['probs'], size=num_aspect_tokens))
                
        self.rng.shuffle(doc_tokens)
        return " ".join(doc_tokens), aspects, aspect_weights

    def generate_corpus(self, num_docs=5000, redundancy=0.0, skew='low'):
        documents = []
        doc_aspects = {}
        doc_aspect_weights = {}
        cluster_ids = {}
        
        # Handle skew
        if skew == 'high':
            # Dominant aspects
            aspect_probs = np.random.dirichlet(np.ones(self.num_aspects) * 0.1)
        else:
            aspect_probs = np.ones(self.num_aspects) / self.num_aspects

        # Generate base documents
        num_base = int(num_docs * (1 - redundancy))
        if num_base == 0 and num_docs > 0:
            num_base = 1
            
        for i in range(num_base):
            doc_id = f"doc_{i}"
            length = int(self.rng.lognormal(mean=np.log(600), sigma=0.5))
            length = max(100, min(length, 2000))
            
            # Select aspects based on skew
            num_aspects = self.rng.integers(1, 4)
            aspects = self.rng.choice(self.num_aspects, size=num_aspects, p=aspect_probs, replace=False)
            weights = self.rng.dirichlet(np.ones(len(aspects)))
            
            text, asp, wt = self.generate_document(length, aspects, weights)
            documents.append({'id': doc_id, 'text': text})
            doc_aspects[doc_id] = asp.tolist()
            doc_aspect_weights[doc_id] = wt.tolist()
            cluster_ids[doc_id] = i
            
        # Generate duplicates
        for i in range(num_base, num_docs):
            base_doc_idx = self.rng.integers(0, num_base)
            base_doc_id = f"doc_{base_doc_idx}"
            
            # 70-90% token overlap
            base_tokens = documents[base_doc_idx]['text'].split()
            overlap_frac = self.rng.uniform(0.7, 0.9)
            num_keep = int(len(base_tokens) * overlap_frac)
            
            kept_tokens = self.rng.choice(base_tokens, size=num_keep, replace=False).tolist()
            
            # fill the rest with same aspects
            asp = doc_aspects[base_doc_id]
            wt = doc_aspect_weights[base_doc_id]
            length = len(base_tokens)
            added_text, _, _ = self.generate_document(length - num_keep, asp, wt)
            kept_tokens.extend(added_text.split())
            self.rng.shuffle(kept_tokens)
            
            doc_id = f"doc_{i}"
            documents.append({'id': doc_id, 'text': " ".join(kept_tokens)})
            doc_aspects[doc_id] = asp
            doc_aspect_weights[doc_id] = wt
            cluster_ids[doc_id] = cluster_ids[base_doc_id]

        return documents, doc_aspects, doc_aspect_weights, cluster_ids

    def generate_queries(self, num_queries=200, single_aspect=False):
        queries = []
        for i in range(num_queries):
            qid = f"q_{i}"
            num_aspects = 1 if single_aspect else self.rng.integers(3, 6)
            aspects = self.rng.choice(self.num_aspects, size=num_aspects, replace=False)
            weights = self.rng.dirichlet(np.ones(len(aspects)))
            
            length = self.rng.integers(100, 301)
            text, _, _ = self.generate_document(length, aspects, weights)
            
            queries.append({
                'id': qid,
                'text': text,
                'aspects': aspects.tolist(),
                'weights': weights.tolist()
            })
        return queries

    def compute_relevance(self, queries, doc_aspects, doc_weights, tau=0.25):
        qrels = defaultdict(dict)
        for q in queries:
            qid = q['id']
            q_aspects = set(q['aspects'])
            
            for doc_id, aspects in doc_aspects.items():
                weights = doc_weights[doc_id]
                rel_score = 0
                for a, w in zip(aspects, weights):
                    if a in q_aspects and w >= tau:
                        rel_score += 1
                
                if rel_score > 0:
                    qrels[qid][doc_id] = rel_score
        return qrels

