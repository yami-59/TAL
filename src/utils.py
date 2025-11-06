import os, json

def load_docs(data_dir="data"):
    docs, names = [], []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("wiki_") and fn.endswith(".txt"):
            p = os.path.join(data_dir, fn)
            with open(p, "r", encoding="utf8") as f:
                docs.append(f.read())
                names.append(fn)
    return names, docs

def load_queries(path="data/requetes.jsonl"):
    items = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            answer = obj["Answer file"].strip()
            queries = [q.strip() for q in obj["Queries"]]
            items.append((answer, queries))
    return items
