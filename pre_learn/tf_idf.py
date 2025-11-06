import math
import numpy as np

docs = {
    "doc1": "bonjour paris",
    "doc2": "paris soleil",
    "doc3": "bonjour france bonjour"
}


#有一群文档，先 生成倒排，再 计算TF

feq={}
index={}

for w,c in docs.items():
    for i in docs[w].split():
        feq[i]=feq.get(i,0) + 1
    for i in docs[w].split():
        index[i]=index.get(i,[]) + [w] #未去重 会出现重复的文档

print(feq)
print(index)

print("____________________________________")
# the count of doc
N = len(docs)
print(N)
#df word feq in the doc1 
df={}
for d, text in docs.items():
    words = set(text.split())  # set 去重 
    for w in words:
        df[w] = df.get(w,0) + 1
print(df)

idf={}
for i in df:
    idf[i] = math.log(N/df[i])
print(idf)

# tfidf = tf * idf  in the doc1
tfidf={}
for i in docs:
    words = docs[i].split()
    tfidf[i] = {}
    for j in words:
        tf = words.count(j)
        tfidf[i][j] = tf * idf[j]

print(tfidf)


A = np.array([1, 2, 3])
B = np.array([2, 3, 4])

#cos_sim(A,B)= A⋅B​ / ∥A∥×∥B∥
cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
print(cos_sim)










