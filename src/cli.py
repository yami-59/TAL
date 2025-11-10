# 统一命令行入口：index/search/eval 统一入口，便于老师“一键测试”
import argparse, os, json
from collections import defaultdict
from tqdm import tqdm
from .utils import read_documents, read_queries
from .indexer import InvertedIndex
from .searcher import Searcher
from .evaluator import evaluate_run

def cmd_index(args):
    docs = read_documents(args.docs)
    idx = InvertedIndex()
    idx.build(docs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    idx.save(args.out)
    print(f"[OK] Indexed {idx.N} docs; vocab={len(idx.df)}; saved to {args.out}")

def cmd_search(args):
    idx = InvertedIndex.load(args.index)
    searcher = Searcher(idx, method=args.method)
    res = searcher.search(args.query, k=args.k)
    for doc, score in res:
        print(f"{score:.4f}\t{doc}")

def cmd_eval(args):
    # 读取查询
    qset = read_queries(args.queries)
    # 将多条 Queries 视为多个独立查询
    gold = {}
    flat_queries = []
    for ans, qlist in qset:
        for q in qlist:
            gold[q] = ans
            flat_queries.append(q)

    # 搜索
    idx = InvertedIndex.load(args.index)
    searcher = Searcher(idx, method=args.method)
    results = {}
    for q in tqdm(flat_queries, desc="Searching"):
        results[q] = searcher.search(q, k=args.k)

    # 评估
    metrics = evaluate_run(results, gold, k=args.k)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx = sub.add_parser("index")
    ap_idx.add_argument("--docs", default="data/wiki_split_extract_2k")
    ap_idx.add_argument("--out", default="models/index.pkl")
    ap_idx.set_defaults(func=cmd_index)

    ap_s = sub.add_parser("search")
    ap_s.add_argument("--index", default="models/index.pkl")
    ap_s.add_argument("--method", choices=["tfidf","bm25"], default="tfidf")
    ap_s.add_argument("--query", required=True)
    ap_s.add_argument("-k", type=int, default=10)
    ap_s.set_defaults(func=cmd_search)

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--index", default="models/index.pkl")
    ap_e.add_argument("--queries", default="data/requetes.jsonl")
    ap_e.add_argument("--method", choices=["tfidf","bm25"], default="tfidf")
    ap_e.add_argument("-k", type=int, default=10)
    ap_e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
