from collections import defaultdict
import math, re, numpy as np

# Step 1: 文本清洗
def preprocess(text):
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', text.lower())
    return text.split()

# Step 2: 统计词频
def build_freq(texts):
    df = defaultdict(int)
    for text in texts:
        for word in set(preprocess(text)):
            df[word] += 1
    return df

# Step 3: 计算TF-IDF并比较
docs = ["Paris est belle", "La tour Eiffel est à Paris", "J’aime la France"]
DF = build_freq(docs)
doc_tokens = [preprocess(t) for t in docs]

# 计算TF-IDF向量并求余弦相似度
def cosine_sim(doc1, doc2):
    words = list(set(doc1 + doc2))
    v1 = np.array([doc1.count(w) for w in words])
    v2 = np.array([doc2.count(w) for w in words])
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Cosine similarity:", cosine_sim(doc_tokens[0], doc_tokens[1]))
