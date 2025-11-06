from collections import defaultdict, Counter
from .preprocess import normalize, tokenize

class InvertedIndex:
    def __init__(self):
        self.df = defaultdict(int)            # term -> doc freq
        self.postings = defaultdict(list)     # term -> [(doc_id, tf)]
        self.doc_len = []                     # doc_id -> token count
        self.doc_names = []                   # doc_id -> filename

    @property
    def N(self): return len(self.doc_names)

    def build(self, doc_names, doc_texts):
        self.doc_names = doc_names
        for i, text in enumerate(doc_texts):
            tokens = tokenize(normalize(text))
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            for w, c in tf.items():
                self.df[w] += 1
                self.postings[w].append((i, c))
