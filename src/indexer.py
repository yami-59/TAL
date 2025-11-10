 # 构建与保存倒排索引
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from .utils import tokenize

class InvertedIndex:
    def __init__(self):
        # postings: term -> dict(doc_id -> tf)
        self.postings = defaultdict(dict)
        # doc_len_terms: doc_id -> length in tokens
        self.doc_len_terms = {}
        # df: term -> document frequency
        self.df = {}
        self.N = 0

    def build(self, corpus: Dict[str, str]):
        self.N = len(corpus)
        for doc_id, text in corpus.items():
            toks = tokenize(text)
            self.doc_len_terms[doc_id] = len(toks)
            tf = Counter(toks)
            for t, c in tf.items():
                self.postings[t][doc_id] = c
        self.df = {t: len(docs) for t, docs in self.postings.items()}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, "rb") as f:
            obj.__dict__ = pickle.load(f)
        return obj

