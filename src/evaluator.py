# Hits@k / MRR@k / nDCG@k
import math
from typing import List, Tuple, Dict, Optional

def hits_at_k(ranks: List[Optional[int]], k: int) -> float:
    # Hit = Existence and Rank <= k
    return sum(1 for r in ranks if r is not None and r <= k) / max(1, len(ranks))

def mrr_at_k(ranks: List[Optional[int]], k: int) -> float:
   # Only accumulate 1/r for the hit samples
    s = 0.0
    for r in ranks:
        if r is not None and r <= k:
            s += 1.0 / r
    return s / max(1, len(ranks))

def ndcg_at_k(rank: Optional[int], k: int) -> float:
    # Binary relevance; miss or >k is recorded as 0.
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def evaluate_run(results: Dict[str, List[Tuple[str, float]]],
                 gold: Dict[str, str],
                 k: int = 10) -> Dict[str, float]:
    ranks: List[Optional[int]] = []

    for q, ranking in results.items():
        ans = gold[q]
       # The evaluation only considers the top k results; if none of the top k results are correct, it is considered a miss (None).
        r: Optional[int] = None
        for i, (doc, _) in enumerate(ranking[:k], start=1):
            if doc == ans:
                r = i
                break
        ranks.append(r)

    # Capped Mean: Misses are counted as k+1 in the mean to avoid the mean being overestimated by extreme values.
    capped = [(r if r is not None else (k + 1)) for r in ranks]

    return {
        f"Hits@{k}": hits_at_k(ranks, k),
        f"MRR@{k}": mrr_at_k(ranks, k),
        f"nDCG@{k}": sum(ndcg_at_k(r, k) for r in ranks) / max(1, len(ranks)),
        "MeanRank": sum(capped) / max(1, len(capped))
    }
