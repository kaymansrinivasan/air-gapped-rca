"""Dependency-free lexical vector + exact-token retrieval baseline.

TF-IDF vectors are lexical, NOT learned semantic embeddings. Rebuild only from
stage-allowed evidence, so future records cannot affect ranking statistics.
"""
import math
import re
from collections import Counter


def tokens(text):
    return re.findall(r'[a-z0-9_]+', text.lower())


def retrieve(chunks, query, top_k=12):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    counts = [Counter(tokens(c['text'])) for c in chunks]
    df = Counter(t for count in counts for t in count)
    idf = {t: math.log((1 + len(chunks)) / (1 + n)) + 1 for t, n in df.items()}
    def vector(count):
        return {t: (1 + math.log(n)) * idf[t] for t, n in count.items() if t in idf}
    q = vector(Counter(tokens(query)))
    qnorm = math.sqrt(sum(v*v for v in q.values()))
    cosine, keyword = [], []
    for i, count in enumerate(counts):
        v = vector(count)
        norm = math.sqrt(sum(x*x for x in v.values()))
        score = sum(q[t]*v.get(t, 0) for t in q) / (qnorm*norm) if qnorm*norm else 0
        exact = sum(idf[t] for t in set(q) & set(count))
        if score > 0:
            cosine.append((i, score))
            keyword.append((i, exact))
    scores = Counter()
    for ranking in (cosine, keyword):
        for rank, (i, _) in enumerate(sorted(ranking, key=lambda x: (-x[1], x[0])), 1):
            scores[i] += 1 / (60 + rank)
    return [dict(chunks[i], retrieval_score=score)
            for i, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
