# RAG Pipeline: Fake News Baseline + PDF Q&A

Two-part project: a text classification baseline, and a retrieval-augmented generation (RAG) system over a real PDF with hybrid search and citations.

## Part 1: Classification Baseline
SMS Spam Collection dataset (5,572 messages, ham/spam). MiniLM/mpnet sentence embeddings + Logistic Regression.

**Accuracy: 0.98

An earlier baseline on the FakeNewsNet dataset (422 articles) scored near-random (~50-55%) due to small sample size and noisy scraped text — swapped to a cleaner dataset for a meaningful result.

## Part 2: PDF RAG Pipeline
Built a question-answering system over a 93-page physics reference PDF.

**Pipeline:**
1. Extracted text per page with PyMuPDF, stripped repeated footers, cleaned whitespace
2. Chunked into 100-word segments with 20-word overlap
3. Embedded chunks with `all-mpnet-base-v2`, indexed with FAISS (cosine similarity)
4. Added BM25 keyword search, combined with FAISS via Reciprocal Rank Fusion for hybrid retrieval
5. Generated answers with OpenAI's `gpt-4o-mini`, grounded strictly in retrieved context, with page citations

**Evaluation:** 5 test questions (3 direct, 1 requiring synthesis across chunks, 1 out-of-scope). All 5 answered correctly, including the out-of-scope question, which the model correctly declined to answer rather than hallucinate.

## Stack
Python, PyMuPDF, Sentence Transformers, FAISS, rank-bm25, OpenAI API, scikit-learn.

## Files
- `notebook.ipynb`: full pipeline, classification baseline through PDF RAG with hybrid retrieval

## Notes
- FAISS index and chunk metadata aren't included (excluded via `.gitignore`); rebuild by running the notebook top to bottom with a PDF of your own.
