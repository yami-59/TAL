# information retrieval  IR
从大量文本文件中 查询出 用户 查询 最相关的  内容  并排序
如 google搜索

# Bag of words
仅统计词频 每篇文档被统计为一个词频向量
要建立一个词汇表
然后统计文档中词语出现的次数  不管词语意思和语序和词性变化  维度高  稀疏性强

# inverted index
建立从词 到 文档的 索引结构

# TF-IDF
衡量一个词 在一个文档中的 重要性
- **公式**：TFIDF(t,d) = TF(t,d) \times \log\frac{N}{DF(t)}
- **TF(t,d)**：词 t 在文档 d 中出现的次数。
- **DF(t)**：包含词 t 的文档数 一个文档中多次出现记得去重 set(text.split()) set去重。
- **N**：总文档数。

# Cosine Similarity
计算两个向量的相似度 值越接近1 越相似
- **公式**：cos_sim(A,B)= A⋅B​ / ∥A∥×∥B∥
- **计算**：numpy库 cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# 评估指标
- **Accuracy**：预测正确的比例；
- **Precision@k**：前 k 个检索结果中与查询真正相关的比例；
- **Recall**：找出的相关文档占所有相关文档的比例。

