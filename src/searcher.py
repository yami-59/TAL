# Top-K 检索（OR 检索 + 打分）
from typing import List, Tuple
from .indexer import InvertedIndex
from .scorer import TFIDFScorer, BM25Scorer

class Searcher:
    def __init__(self, index: InvertedIndex, method="tfidf"):
        self.index = index
        if method == "tfidf":
            self.scorer = TFIDFScorer(index)
        elif method == "bm25":
            self.scorer = BM25Scorer(index)
        else:
            raise ValueError("method must be 'tfidf' or 'bm25'")

    def search(self, query: str, k=10) -> List[Tuple[str,float]]:
        scores = self.scorer.score(query)
        # 排序并返回 Top-K
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
