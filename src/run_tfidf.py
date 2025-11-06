import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import load_docs
from src.index import InvertedIndex
from src.tfidf import TfidfSearcher

import argparse



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cosine", action="store_true")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))  # C:\TAL
    data_dir = os.path.join(base_dir, "data")
    names, texts = load_docs(data_dir)


    I = InvertedIndex(); I.build(names, texts)
    S = TfidfSearcher(I)
    results = S.search(args.query, top_k=args.k, cosine=args.cosine)
    print("Query:", args.query)
    for i,(n,sc) in enumerate(results,1):
        print(f"{i}. {n}  score={sc:.4f}")
