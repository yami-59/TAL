import argparse, os, json, csv
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
    # Read query
    qset = read_queries(args.queries)
    # Treat multiple Queries as multiple independent queries
    gold = {}
    flat_queries = []
    for ans, qlist in qset:
        for q in qlist:
            gold[q] = ans
            flat_queries.append(q)

    # search
    idx = InvertedIndex.load(args.index)
    searcher = Searcher(idx, method=args.method)
    results = {}
    for q in tqdm(flat_queries, desc="Searching"):
        results[q] = searcher.search(q, k=args.k)

    os.makedirs("results", exist_ok=True)
    name_file = f"results/resultats_{args.method}_top{args.k}.csv"
    with open(name_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        entetes = ["moteur", "requete"]
        for i in range(1, args.k+1): 
            entetes.append(f"doc_top{i}")
            entetes.append(f"score_top{i}")
        entetes.append("trouve_ou_non")
        writer.writerow(entetes)

        for q, docs in results.items():
            row = [args.method, q] 
            trouve = "NON"
            for i in range(args.k):
                if i < len(docs):
                    doc, score = docs[i]
                    row += [doc, score]
                    trouve = "OUI"
                else:
                    row += ["", ""]
            row.append(trouve)
            writer.writerow(row)

    print("Fichier CSV sauvegardé → {name_file}")
    # Evaluate
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
