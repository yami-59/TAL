# Reading and writing to achieve text cleaning and word segmentation
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

def remove_accents(text: str) -> str:  # ← AJOUT n°2 (nouvelle fonction)
    """Supprime tous les accents (é → e, à → a, etc.)"""
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return str(text)

def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("œ","oe").replace("æ","ae")   # Common Continuous Writing
    text = remove_accents(text) #supprime les accents
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
            # Clean up invisible control characters (preserve regular whitespace)
            s = "".join(ch for ch in s if (ord(ch) >= 32 or ch in "\t\r\n"))
            # Remove extra commas before the } or ] at the end of a line.
            s = re.sub(r",\s*([}\]])\s*$", r"\1", s)
            # Remove any remaining initial BOM (Bill of Materials)
            s = s.lstrip("\ufeff")

            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # Print out the problematic line to help locate it
                print(f"[read_queries] JSON error at line {i}: {e}\nRAW: {s[:160]}")
                raise

            ans = obj.get("Answer file")
            qs  = obj.get("Queries", [])
            if ans and isinstance(qs, list) and qs:
                out.append((ans, qs))
    return out

