# Top-K Search (OR Search + Rating)
from typing import List, Tuple
from .indexer import InvertedIndex
from .scorer import TFIDFScorer, BM25Scorer, EnhancedReranker


class Searcher:
    def __init__(self, index: InvertedIndex, method="tfidf"):
        self.index = index
        if method == "tfidf":
            self.scorer = TFIDFScorer(index)
        elif method == "bm25":
            self.scorer = BM25Scorer(index)
        else:
            raise ValueError("method must be 'tfidf' or 'bm25'")

    def search(self, query: str, k=10) -> List[Tuple[str, float]]:
        # Calculer les scores de base (avec TF-IDF ou BM25)
        scores = self.scorer.score(query)
        
        # Reranking amélioré avec EnhancedReranker
        reranker = EnhancedReranker(self.index)
        reranked_scores = reranker.score(query, candidate_docs=list(scores.keys()))
        
        # Combiner les scores : moyenne des deux (scores initiaux + reranking)
        final_scores = {
            doc: (scores.get(doc, 0) + reranked_scores.get(doc, 0)) / 2 
            for doc in scores.keys()
        }
        
        # Retourner les documents avec les scores les plus élevés (Top-K)
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]





