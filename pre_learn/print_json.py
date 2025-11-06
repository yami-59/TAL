import json

with open('../data/requetes.jsonl','r',encoding='utf8') as f:
    for i,line in enumerate(f):
        if i>=5:
            break
        q = json.loads(line)
        print(i,q)
        


with open('../data/wiki_013117.txt','r',encoding='utf8') as fl:
    print(fl.read()[:10000])