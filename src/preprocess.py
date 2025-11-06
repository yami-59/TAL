# 预处理

import re

# 保留法语字母表
ALLOWED = r"a-zàâçéèêëîïôûùüÿñæœ"

def normalize(text: str) -> str:
    text = text.lower()
    # 非字母与空白替换为空格
    text = re.sub(fr"[^ {ALLOWED}\s]", " ", text)
    # 压缩多空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list:
    return text.split()
