# 读写 实现文本清洗 分词 
import os, json, re, unicodedata
from typing import Dict, List, Tuple
from .stopwords_fr import STOPWORDS_FR

def read_documents(folder: str) -> Dict[str, str]:
    docs = {}
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            with open(os.path.join(folder, fn), "r", encoding="utf-8", errors="ignore") as f:
                docs[fn] = f.read()
    return docs

_ws = re.compile(r"\W+", flags=re.UNICODE)

def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("œ","oe").replace("æ","ae")   # 常见连写
    return text

def tokenize(text: str, remove_stop=True, min_len=2) -> List[str]:
    text = normalize(text)
    toks = [t for t in _ws.split(text) if t]
    if remove_stop:
        toks = [t for t in toks if t not in STOPWORDS_FR]
    toks = [t for t in toks if len(t) >= min_len and not t.isdigit()]
    return toks


def read_queries(jsonl_path: str):

    out = []
    with open(jsonl_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            # 清理不可见控制字符（保留常规空白）
            s = "".join(ch for ch in s if (ord(ch) >= 32 or ch in "\t\r\n"))
            # 去掉行尾在 } 或 ] 前面的多余逗号
            s = re.sub(r",\s*([}\]])\s*$", r"\1", s)
            # 去掉可能残留的首部 BOM
            s = s.lstrip("\ufeff")

            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # 打印出问题行，便于定位（然后抛出让你能看到具体是哪一行坏了）
                print(f"[read_queries] JSON error at line {i}: {e}\nRAW: {s[:160]}")
                raise

            ans = obj.get("Answer file")
            qs  = obj.get("Queries", [])
            if ans and isinstance(qs, list) and qs:
                out.append((ans, qs))
    return out

