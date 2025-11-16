#  Information Retrieval Project — TF-IDF & BM25

A functional IR system for French Wikipedia articles.  
Works on **Windows, Linux** — no external dependencies beyond `numpy` and `tqdm`.

---

##  Quick Start 

### 1️ (Optional) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2️ Install dependencies
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing:
```bash
pip install numpy tqdm
```

### 4️ Build the index
```bash
python -m src.cli index --docs data --out models/index.pkl
```

### 5️ Run a single query demo (TF-IDF)
```bash
python -m src.cli search --index models/index.pkl --method tfidf --query "course à pied" -k 10
```

### 6️ Run batch evaluation (50 queries from requetes.jsonl)
```bash
python -m src.cli eval --index models/index.pkl --method tfidf --queries data/requetes.jsonl -k 10
```

### 7️ Compare BM25
```bash
python -m src.cli eval --index models/index.pkl --method bm25 --queries data/requetes.jsonl -k 10
```


---

##  Expected Behavior

- `search` prints top-K results as  
  `score<TAB>doc_id`  
  Example:
  ```
  0.25    wiki_042186.txt
  ```

- `eval` prints JSON with metrics:
  ```json
  {
    "Hits@10": 0.66,
    "MRR@10": 0.52,
    "nDCG@10": 0.58,
    "MeanRank": 4.12
  }
  ```

---

##  Repository Layout

```
.
├── data/
│   ├── requetes.jsonl             # one JSON per line: {"Answer file": "...", "Queries": ["...", "..."]}
│   └── wiki_*.txt                 # ~2000 UTF-8 documents
├── models/
│   └── index.pkl                  # serialized inverted index (created by step 1)
├── src/
│   ├── __init__.py
│   ├── stopwords_fr.py            # small French stopword set (editable)
│   ├── utils.py                   # normalization, tokenization, robust JSONL reader
│   ├── indexer.py                 # InvertedIndex class (postings, df, doc_len, N)
│   ├── scorer.py                  # TFIDFScorer (cosine), BM25Scorer (k1,b configurable)
│   ├── searcher.py                # Searcher(method).search(query, k)
│   ├── evaluator.py               # Hits@k / MRR@k / nDCG@k / MeanRank
│   └── cli.py                     # unified CLI: index / search / eval
├── README.md
└── requirements.txt
└── .gitignore
```

---

##  Data & Formats

### Documents
All files in `data/wiki_*.txt` (UTF-8).  

### Queries (`data/requetes.jsonl`)
Each line is a JSON object:
```json
{"Answer file": "wiki_042186.txt", "Queries": ["course à pied", "trail"]}
```

 The reader in `utils.py` is robust to:
- BOM characters  
- trailing commas  
- malformed lines (it logs the offending line)

---

##  How It Works

1. **Preprocessing**
   - Lowercasing  
   - Accent stripping (`é→e`, `œ→oe`, `æ→ae`)  
   - Tokenization on `\W+`  
   - Optional stopword removal (see `stopwords_fr.py`)  
   - Min-length filter  

2. **Indexing**
   - Build inverted index `term → {doc_id: tf}`  
   - Compute document frequency (`df`), collection size (`N`), and document lengths  

3. **Scoring**
   - **TF-IDF (cosine):**  
     `w_td = (1 + log(tf)) * log(N/df)`  
     Cosine similarity with precomputed document norms  
   - **BM25:**  
     Robertson–Walker formula with `k1=1.5`, `b=0.75` (tunable)

4. **Evaluation**
   - For each query variant, check rank of the correct (gold) document  
   - Metrics: `Hits@k`, `MRR@k`, `nDCG@k`, `MeanRank (≤ k+1)`

---

##  CLI Usage Summary

```bash
# Build index
python -m src.cli index --docs data --out models/index.pkl

# Search with TF-IDF or BM25
python -m src.cli search --index models/index.pkl --method tfidf --query "..." -k 10
python -m src.cli search --index models/index.pkl --method bm25  --query "..." -k 10

# Evaluate on all queries
python -m src.cli eval --index models/index.pkl --method tfidf --queries data/requetes.jsonl -k 10
python -m src.cli eval --index models/index.pkl --method bm25  --queries data/requetes.jsonl -k 10
```

---
