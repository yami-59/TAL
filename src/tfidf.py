import math, heapq
from collections import defaultdict
from .preprocess import normalize, tokenize

class TfidfSearcher:
    def __init__(self, index):
        self.I = index
        self._idf_cache = {}

    def _idf(self, w):
        if w in self._idf_cache: return self._idf_cache[w]
        idf = math.log(self.I.N / (1 + self.I.df.get(w, 0)))
        self._idf_cache[w] = idf
        return idf

    def search(self, query: str, top_k=3, cosine=False):
        q_tokens = tokenize(normalize(query))
        q_tf = defaultdict(int)
        for w in q_tokens: q_tf[w] += 1

        # 稀疏 TF-IDF 的内积
        scores = defaultdict(float)
        qnorm = 0.0
        for w, tf in q_tf.items():
            idf = self._idf(w)
            q_w = tf * idf
            qnorm += q_w * q_w
            for doc_id, tf_d in self.I.postings.get(w, []):
                scores[doc_id] += q_w * (tf_d * idf)

        if cosine:
            qnorm = math.sqrt(qnorm) or 1.0
            # 需要文档范数：按词表近似计算
            dnorm = defaultdict(float)
            for w, tf in q_tf.items():
                idf = self._idf(w)
                for doc_id, tf_d in self.I.postings.get(w, []):
                    dnorm[doc_id] += (tf_d * idf) ** 2
            for d in scores:
                scores[d] /= (qnorm * (math.sqrt(dnorm[d]) or 1.0))

        top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [(self.I.doc_names[d], s) for d, s in top]
