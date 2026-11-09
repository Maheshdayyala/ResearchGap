"""Semantic representation. Uses Sentence-Transformers when available, else hashed TF vectors (offline fallback)."""
import hashlib
import logging
import math
from collections import Counter

import numpy as np

from settings import settings
from utils import tokens

log = logging.getLogger(__name__)
_model, _tried, _cache = None, False, {}
DIM = 1024


def _load():
    global _model, _tried
    if _tried:
        return _model
    _tried = True
    if settings.USE_SBERT == "false":
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.EMBED_MODEL)
    except Exception as e:
        log.warning("Sentence-Transformers unavailable (%s); using hashed TF fallback", e)
    return _model


def backend():
    return settings.EMBED_MODEL if _load() else "hashed-tf-fallback"


def _hash_vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    t = tokens(text)
    for g, c in Counter(t + [a + "_" + b for a, b in zip(t, t[1:])]).items():
        v[int(hashlib.md5(g.encode()).hexdigest(), 16) % DIM] += 1 + math.log(c)
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(texts):
    """Return L2-normalised matrix (len(texts), d)."""
    texts = [t or "" for t in texts]
    m = _load()
    if m:
        todo = [t for t in set(texts) if t not in _cache]
        if todo:
            for t, e in zip(todo, m.encode(todo, normalize_embeddings=True, show_progress_bar=False)):
                _cache[t] = e
        return np.vstack([_cache[t] for t in texts]) if texts else np.zeros((0, 384))
    return np.vstack([_hash_vec(t) for t in texts]) if texts else np.zeros((0, DIM))


def sim_matrix(a, b):
    return embed(a) @ embed(b).T
