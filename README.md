## Agentic RAG

A high-performance, multi-agent Retrieval-Augmented Generation system built with LangGraph and Qdrant. This architecture supports adaptive routing, self-correction, and hybrid search with advanced re-ranking.

### Key Features
Cyclic graph architecture using LangGraph for complex reasoning loops.
Combines Dense (Vector) and Sparse (BM25/SPLADE) retrieval using Qdrant.
Implements "Corrective RAG" (CRAG) where agents grade retrieved documents and rewrite queries if necessary.
Configured for Binary Quantization to handle 10M+ vectors efficiently.
PostgreSQL persistence layer for state recovery and Human-in-the-Loop workflows.
