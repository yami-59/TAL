# Hits@k / MRR@k / nDCG@k
import math
from typing import List, Tuple, Dict, Optional

def hits_at_k(ranks: List[Optional[int]], k: int) -> float:
    # 命中=存在且排名<=k
    return sum(1 for r in ranks if r is not None and r <= k) / max(1, len(ranks))

def mrr_at_k(ranks: List[Optional[int]], k: int) -> float:
    # 只对命中的样本累加 1/r
    s = 0.0
    for r in ranks:
        if r is not None and r <= k:
            s += 1.0 / r
    return s / max(1, len(ranks))

def ndcg_at_k(rank: Optional[int], k: int) -> float:
    # 单一相关（binary relevance）；没命中或>k 记 0
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def evaluate_run(results: Dict[str, List[Tuple[str, float]]],
                 gold: Dict[str, str],
                 k: int = 10) -> Dict[str, float]:
    ranks: List[Optional[int]] = []

    for q, ranking in results.items():
        ans = gold[q]
        # 评测只关心前k名；若前k名都没有答案，就视为未命中(None)
        r: Optional[int] = None
        for i, (doc, _) in enumerate(ranking[:k], start=1):
            if doc == ans:
                r = i
                break
        ranks.append(r)

    # 封顶均值：未命中按 k+1 计入均值，避免均值被极大值拉爆
    capped = [(r if r is not None else (k + 1)) for r in ranks]

    return {
        f"Hits@{k}": hits_at_k(ranks, k),
        f"MRR@{k}": mrr_at_k(ranks, k),
        f"nDCG@{k}": sum(ndcg_at_k(r, k) for r in ranks) / max(1, len(ranks)),
        "MeanRank": sum(capped) / max(1, len(capped))
    }
