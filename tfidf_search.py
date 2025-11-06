import os, re, math, json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text)
    return text.split()


data_dir = "data"
documents = []
doc_names = []

for file in os.listdir(data_dir):
    if file.startswith("wiki_") and file.endswith(".txt"):
        with open(os.path.join(data_dir, file), "r", encoding="utf8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

N = len(documents)


inverted_index = defaultdict(list)
df = defaultdict(int)

for doc_id, text in enumerate(documents):
    words = set(preprocess(text))
    for w in words:
        inverted_index[w].append(doc_id)
        df[w] += 1


vocab = list(df.keys())
vocab_index = {w: i for i, w in enumerate(vocab)}
tfidf_matrix = np.zeros((N, len(vocab)))

for doc_id, text in enumerate(documents):
    words = preprocess(text)
    for w in words:
        if w in vocab_index:
            tf = words.count(w)
            idf = math.log(N / (1 + df[w]))
            tfidf_matrix[doc_id, vocab_index[w]] = tf * idf


def search(query, top_k=3):
    q_words = preprocess(query)
    q_vec = np.zeros((1, len(vocab)))
    for w in q_words:
        if w in vocab_index:
            tf = q_words.count(w)
            idf = math.log(N / (1 + df[w]))
            q_vec[0, vocab_index[w]] = tf * idf
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(doc_names[i], sims[i]) for i in top_indices]


query = "course à pied"
results = search(query)
print("Query:", query)
for rank, (doc, score) in enumerate(results, 1):
    print(f"{rank}. {doc} (score={score:.4f})")
