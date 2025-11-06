import json
from collections import defaultdict

feq={"bonjour":3 , "ai":2}

print(feq["ai"])

for w, c in feq.items():
    print(w, ":", c)

for w in feq.keys():
    print(w)

for c in feq.values():
    print(c)

feq["GPT"]=1

# words feq static
text=["b","b","fg","hello","ai","b"]
fequ={}

for w in text:
    fequ[w]=fequ.get(w,0) + 1
    print(fequ)

del feq["GPT"]

cou=feq.pop("GP",2)

print(cou)
print(feq)

#fequ.clear()

print(fequ)

for i in ["b","hello"]:
    print(i, "---", fequ[i])

#inverse-------------------------------------------
#dict
docs = {
    "doc1": "bonjour paris",
    "doc2": "paris soleil",
    "doc3": "bonjour france"
}
print(type(docs))

for p in ["doc1","doc2"]:
    print(p,"--",docs[p])

index={}

#index  bonjour  doc1  text split()
for doc,text in docs.items():
    for w in text.split():
        index[w] = index.get(w,[]) + [doc]

print("______________________________________")
print(index)

#search le words bonjour
for doc in index["bonjour"]:
    print(doc , "----", docs[doc])


#inverted_index
inverted_index=defaultdict(int)
text=["b","b","fg","hello","ai","b"]
feq={}

for i in text:
    feq[i] = feq.get(i,0) + 1
print(feq)

# avanced dic
inverted_index=defaultdict(list)
inverted_index["paris"].append("doc1")
inverted_index["paris"].append("doc2")
print(inverted_index)