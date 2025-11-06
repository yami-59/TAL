import json
import re

with open("wiki_split_extract_2k/wiki_013117.txt",'+r',encoding='utf8') as f:
    content = f.read()
    print(content[:200])
    for line in f:
        print(line.strip())

with open("./request.text","+w",encoding="utf8") as f:
    f.write("ww,wwww,wwwwww,wwwww")

with open("request.text","r",encoding="utf8") as  f:
    text = f.read()
print(text)
#split by commas
print(text.split(","))    

text = "Bonjour,je suis à Paris!"
print(text.lower())           # 全部小写
#split by any whitespace
print(text.split())           # ['bonjour,', 'je', 'suis', 'à', 'paris!']


print(text.upper())

cleaned = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]' , '' , text.lower())
print(cleaned)

stopwords={"je","à"}
tokens = [w for w in cleaned.split() if w not in stopwords]
print(tokens)
#print by character
for i in cleaned:
    print(i)