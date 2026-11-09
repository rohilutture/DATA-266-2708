# METRICS - HW10 RAG (Reranking / Chunking / Conversational Memory)

- SID4=2708, SEED=2708
- Dataset: ccdv/arxiv-summarization [document], first 100 docs | DB: ChromaDB
- Embedder: sentence-transformers/all-MiniLM-L6-v2 | Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 | K=5
- Chunks: S1 fixed=4094, S2 sentence=3570 | eval queries=12

## Reranking (Strategy 1 index)

| setting | P@5 | R@5 | MRR |
|---|---|---|---|
| without reranking | 0.950 | 0.301 | 1.000 |
| with reranking | 0.933 | 0.296 | 0.958 |

## Chunking strategy

| strategy | P@5 | R@5 | MRR |
|---|---|---|---|
| S1 fixed sliding window | 0.950 | 0.301 | 1.000 |
| S2 sentence packing | 0.867 | 0.300 | 1.000 |

## Conversational memory (turn-2 follow-up)

| setting | P@5 | R@5 | MRR |
|---|---|---|---|
| without memory | 0.000 | 0.000 | 0.000 |
| with memory | 1.000 | 0.250 | 1.000 |
