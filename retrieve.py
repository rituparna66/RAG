import numpy as np
import faiss
from rank_bm25 import BM25Okapi

def build_faiss_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def hybrid_retrieve(query, chunks, index, bm25, embedder, k=4, rrf_k=60):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, faiss_idxs = index.search(q_emb, len(chunks))
    faiss_ranks = {idx: rank for rank, idx in enumerate(faiss_idxs[0])}

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(-bm25_scores))}

    rrf_scores = {}
    for idx in range(len(chunks)):
        score = 1 / (rrf_k + faiss_ranks.get(idx, len(chunks))) + 1 / (rrf_k + bm25_ranks.get(idx, len(chunks)))
        rrf_scores[idx] = score

    top_idxs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunks[i] for i in top_idxs]
