 # TF-IDF / BM25 及余弦相似度
import math
from typing import Dict, List
from collections import Counter
from .utils import tokenize

class TFIDFScorer:
    def __init__(self, index, log_tf=True):
        self.index = index
        self.log_tf = log_tf
        # 预计算文档 TF-IDF 范数 ||d||
        self.doc_norm = self._precompute_doc_norm()

    def _idf(self, term: str) -> float:
        df = self.index.df.get(term, 0)
        if df == 0: return 0.0
        return math.log(self.index.N / df)

    def _precompute_doc_norm(self) -> Dict[str, float]:
        # ||d||^2 = sum_t (w_td)^2
        norm2 = {doc:0.0 for doc in self.index.doc_len_terms}
        for t, posting in self.index.postings.items():
            idf = self._idf(t)
            if idf == 0: continue
            for doc, tf in posting.items():
                w = (1 + math.log(tf)) if (self.log_tf and tf>0) else tf
                w *= idf
                norm2[doc] += w*w
        return {doc: math.sqrt(v) if v>0 else 1e-9 for doc, v in norm2.items()}

    def score(self, query: str, candidate_docs: List[str]=None) -> Dict[str,float]:
        q_tokens = tokenize(query)
        tf_q = Counter(q_tokens)
        # 计算 query 权重与 ||q||
        wq = {}
        for t, tf in tf_q.items():
            idf = self._idf(t)
            if idf == 0: continue
            wt = (1 + math.log(tf)) * idf
            wq[t] = wt
        qnorm = math.sqrt(sum(v*v for v in wq.values())) or 1e-9

        # 候选集合：查询涉及的 posting 并集
        if candidate_docs is None:
            cand = set()
            for t in wq.keys():
                cand |= set(self.index.postings.get(t, {}).keys())
        else:
            cand = set(candidate_docs)

        scores = {doc:0.0 for doc in cand}
        for t, wtq in wq.items():
            posting = self.index.postings.get(t, {})
            for doc, tf in posting.items():
                if doc not in scores: continue
                wtd = ((1 + math.log(tf)) if (self.log_tf and tf>0) else tf) * self._idf(t)
                scores[doc] += wtq * wtd
        # 余弦
        for doc in list(scores.keys()):
            scores[doc] = scores[doc] / (qnorm * self.doc_norm.get(doc, 1e-9))
        return scores

class BM25Scorer:
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.avgdl = sum(index.doc_len_terms.values()) / max(1, index.N)

    def _idf(self, t):
        df = self.index.df.get(t, 0)
        if df == 0: return 0.0
        return math.log((self.index.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, candidate_docs=None) -> Dict[str, float]:
        toks = tokenize(query)
        uniq = set(toks)
        if candidate_docs is None:
            cand = set()
            for t in uniq:
                cand |= set(self.index.postings.get(t, {}).keys())
        else:
            cand = set(candidate_docs)
        scores = {doc:0.0 for doc in cand}
        for t in uniq:
            posting = self.index.postings.get(t, {})
            idf = self._idf(t)
            for doc in cand:
                tf = posting.get(doc, 0)
                if tf == 0: continue
                dl = self.index.doc_len_terms[doc]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc] += idf * (tf * (self.k1 + 1) / denom)
        return scores
