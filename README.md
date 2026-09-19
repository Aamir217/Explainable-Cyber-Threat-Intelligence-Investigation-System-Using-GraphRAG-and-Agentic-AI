# Explainable Cyber Threat Intelligence & Investigation System (GraphRAG + Agentic AI)

An end-to-end system that answers multi-hop cyber threat intelligence questions by
combining a **knowledge graph** (MITRE ATT&CK + CVE/NVD), **hybrid document retrieval**
(vector + BM25), **GraphRAG** multi-hop traversal, and a lightweight **agentic planner**
with **evidence verification** — and quantitatively compares that stack against
conventional RAG.

It implements the four systems described in the project spec and lets you run all of
them, side by side, on the same question set:

| # | System | Pipeline |
|---|--------|----------|
| 1 | **Naive RAG** | Vector search → LLM |
| 2 | **Hybrid RAG** | Vector + BM25 → Reciprocal Rank Fusion → Reranker → LLM |
| 3 | **GraphRAG** | Graph traversal + document retrieval → LLM |
| 4 | **Agentic GraphRAG** | Planner → graph/vector/BM25/CVE tools → Verification agent → LLM |

Every answer comes back with **citations**, an explicit **graph reasoning path**, and
(for the agentic system) a **faithfulness score** — the explainability layer the naive
baseline cannot produce.

## Runs entirely on local models — no cloud API required

By default this system uses **no cloud LLM API**:

- **LLM**: [Ollama](https://ollama.com) running locally (`LLM_PROVIDER=ollama`, the
  default). Install Ollama, pull a model, and the system talks to it over
  `http://localhost:11434` — nothing leaves your machine.
- **Embeddings**: a local `sentence-transformers` model (`EMBEDDING_BACKEND=sentence-transformers`,
  the default, `all-MiniLM-L6-v2`) for real semantic similarity. Downloaded once from
  HuggingFace, cached, then runs fully offline on CPU.
- **Graph store**: in-memory (`networkx`-backed) by default; a real `Neo4jGraphStore`
  implementing the identical interface is included if you want to run your own local
  Neo4j instance instead (`docker-compose.yml`).
- **Data**: a curated sample corpus (`data/sample/`) modeled on real MITRE ATT&CK /
  NVD entities (APT28, APT29 and their associated malware, techniques, and CVEs), plus
  five short CTI report fixtures — enough to demonstrate genuine multi-hop reasoning
  that pure text retrieval cannot do (see "Why GraphRAG wins" below).

### Setting up the local LLM (Ollama)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model (llama3.2 is the default; any Ollama model works)
ollama pull llama3.2
# 3. Start the server (often already running as a background service)
ollama serve &
```

That's it — `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.2` are the defaults, so
`python scripts/ask.py "..."` and the dashboard will use it automatically.

### Zero-setup fallback

If Ollama isn't running, or `sentence-transformers` isn't installed / can't reach
HuggingFace to download its model, the system **automatically and silently falls back**
to a dependency-free deterministic `TemplateLLM` and hashed bag-of-words embedder
respectively (with a one-line warning), so it still runs end to end with nothing
installed at all. This is also what the test suite pins to, for fast, fully
reproducible, network-independent tests.

This means every code path — ingestion, graph traversal, hybrid retrieval, the
agentic planner, verification, evaluation, ablation, and the API — is exercised by the
test suite without any external service. Real data sources and real LLMs are one
environment variable away (see [Configuration](#configuration)).

## Architecture

```
 MITRE ATT&CK   CVE / NVD   CTI Reports (Markdown)
      │             │              │
      └─────────────┼──────────────┘
                     ▼
     ┌────────────────────────────────┐
     │  Knowledge Graph (GraphStore)   │   Document Store
     │  in-memory (default) / Neo4j    │   chunk → hashed/ST embeddings
     └───────────────┬────────────────┘         │
                      │                    Vector Index + BM25 Index
                      │                          │
       ┌──────────────┴──────────┬───────────────┘
       ▼                         ▼
  Graph traversal          Fusion (RRF) → Reranker
  (entity link + BFS)              │
       └──────────────┬───────────┘
                       ▼
              Evidence Collection
                       │
          ┌────────────┴────────────┐
          │   (Agentic system only)  │
          │   AI Planner → tools     │
          │   Verification Agent     │
          └────────────┬────────────┘
                        ▼
                 Answer Generator (LLM)
                        │
                        ▼
        Answer + Citations + Graph Reasoning Path
```

## Why GraphRAG wins on multi-hop questions

Ask: *"Which threat actors use malware that exploits vulnerabilities affecting
Microsoft Exchange Server, and which attack techniques are involved?"*

- **Naive/Hybrid RAG** can only return document chunks that happen to mention these
  entities together in the same sentence/paragraph — they never explicitly connect
  the dots.
- **GraphRAG/Agentic GraphRAG** link "Microsoft Exchange Server" in the graph, walk
  `Product ←AFFECTS– CVE ←EXPLOITS– Malware ←USES– ThreatActor`, and return the
  literal reasoning chain, e.g.:

  ```
  Microsoft Exchange Server <--AFFECTS-- CVE-2020-0688 <--EXPLOITS-- Zebrocy <--USES-- APT28
  Microsoft Exchange Server <--AFFECTS-- CVE-2021-26855 <--EXPLOITS-- WellMess <--USES-- APT29
  ```

Run `python -m cti_graphrag.evaluation.run_evaluation` and compare `semantic_similarity`
and `entity_coverage` across systems — GraphRAG/Agentic GraphRAG score materially
higher than Naive/Hybrid RAG on multi-hop questions in `data/eval/eval_questions.json`,
precisely because they surface entities/relationships that never co-occur in a single
document chunk.

## Project layout

```
src/cti_graphrag/
  ingestion/        MITRE ATT&CK (STIX-like), NVD/CVE, CTI report loaders
  graph/            schema, GraphStore (in-memory + Neo4j), builder
  retrieval/        chunking, vector store, BM25, graph retriever, fusion, reranker
  agents/           rule-based planner + tool wrappers (graph/vector/bm25/cve)
  verification/     claim extraction + evidence-grounding checker
  rag/              NaiveRAG, HybridRAG, GraphRAGSystem, AgenticGraphRAG
  evaluation/       eval dataset, metrics, run_evaluation, ablation, error_analysis
  api/              FastAPI backend
  llm.py            OllamaLLM (default, local) / TemplateLLM (fallback) / AnthropicLLM / OpenAILLM
  embeddings.py     HashingEmbedder / SentenceTransformerEmbedder
  corpus.py         builds the shared graph + indices used by every system
data/
  sample/           curated MITRE ATT&CK + NVD sample bundles, CTI report fixtures
  eval/             16 evaluation questions (single/two/three/multi-hop)
frontend/           static dashboard (vanilla HTML/JS, no build step)
scripts/            ask.py (CLI), serve.py (API server)
tests/              pytest suite covering every layer
reports/            generated evaluation/ablation/error-analysis output (git-ignored inputs, committed sample run)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ask a question from the CLI
python scripts/ask.py "Which malware is used by APT28?" --all

# Run the dashboard (http://localhost:8000)
python scripts/serve.py

# Run the test suite
PYTHONPATH=src pytest -q

# Compare all 4 systems on the evaluation set
python -m cti_graphrag.evaluation.run_evaluation

# Run the ablation study (Naive -> ... -> Agentic+Verification)
python -m cti_graphrag.evaluation.ablation

# Categorize incorrect answers (Section 13 error taxonomy)
python -m cti_graphrag.evaluation.error_analysis
```

The dashboard lets you ask a question against any of the 4 systems, or click
**"Compare all 4 systems"** to see them side by side, with the answer, tool calls,
graph reasoning path, citations, and (for the agentic system) a faithfulness bar.

## Configuration

Everything is controlled by environment variables (see `src/cti_graphrag/config.py`),
all optional:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `template` \| `anthropic` \| `openai` |
| `LLM_MODEL` | `llama3.2` | Ollama model tag (or Anthropic/OpenAI model id if using those providers) |
| `OLLAMA_HOST` | `http://localhost:11434` | where the local Ollama server is listening |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | required only for those (non-local) providers |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `sentence-transformers` \| `hashing` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | any local sentence-transformers model name |
| `GRAPH_BACKEND` | `memory` | `memory` \| `neo4j` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `password` | used when `GRAPH_BACKEND=neo4j`; see `docker-compose.yml` |
| `RETRIEVAL_TOP_K` | `5` | final evidence chunks returned per query |
| `MAX_GRAPH_HOPS` | `3` | default multi-hop traversal depth |

Both `ollama` and `sentence-transformers` fall back automatically (with a warning) to
`TemplateLLM`/`HashingEmbedder` if the local server/model isn't available, so the
defaults are always safe to leave as-is even without Ollama installed.

To use real MITRE ATT&CK data instead of the sample bundle:

```python
from cti_graphrag.corpus import build_corpus
corpus = build_corpus(use_live_sources=True)  # fetches the full Enterprise ATT&CK STIX bundle
```

## Evaluation framework

`data/eval/eval_questions.json` contains 16 hand-annotated questions (4 per hop
category: single/two/three/multi-hop), each with a ground-truth answer, the relevant
graph entities, the required relationship types, the supporting source documents, and
an expected reasoning path.

`evaluation/metrics.py` implements:

- **Answer quality**: exact match, token-level F1, embedding-based semantic similarity,
  entity coverage.
- **Retrieval quality**: Precision@K, Recall@K, MRR, nDCG (document-level, deduplicated
  across chunks from the same source).
- **RAG quality**: faithfulness (claim-level lexical grounding against retrieved
  evidence), citation correctness, answer/context relevance.

`evaluation/ablation.py` runs seven configurations in increasing order of
sophistication (Naive → Hybrid → GraphRAG → +Reranking → +Hybrid retrieval →
Agentic → Agentic+Verification) so each row isolates the effect of exactly one added
component.

`evaluation/error_analysis.py` classifies every answer that scores below an F1
threshold into the failure taxonomy from the spec (entity recognition failure,
missing/incorrect graph traversal, document retrieval failure, hallucination,
insufficient evidence, incorrect citation, reasoning failure).

### A note on the numbers you'll see

The evaluation/ablation/error-analysis scripts (and the pytest suite) pin
`TemplateLLM` + the hashing embedder for speed and full reproducibility. Against
those hand-written gold answers, exact-match/F1 will look low — the template
stitches together evidence rather than paraphrasing it into the gold phrasing.
**`semantic_similarity`, `entity_coverage`, and the number of graph paths surfaced
are the metrics that isolate retrieval quality** from generation style, and those
show the expected ordering (GraphRAG/Agentic > Hybrid > Naive) regardless of which
LLM is generating the prose. Point the evaluation scripts at a real local Ollama
model (it's the runtime default; the eval scripts just don't force it) to see F1
rise across the board without changing the retrieval story.

## Swapping components

- **A different local model**: `ollama pull <model>` then `LLM_MODEL=<model>` — no
  code changes. Any Ollama-supported model works (Llama, Mistral, Qwen, Gemma, ...).
- **Cloud LLM instead of local**: set `LLM_PROVIDER=anthropic` (or `openai`) and the
  corresponding API key.
- **Neo4j** instead of the in-memory graph: `docker compose up -d`, then
  `GRAPH_BACKEND=neo4j` — `Neo4jGraphStore` implements the exact same interface
  (including multi-hop `traverse()` and `shortest_path()`), so nothing else changes.
- **LLM-based planner**: `agents/planner.py` documents exactly the `plan(question,
  corpus) -> Plan` signature to implement if you want a ReAct-style planner (e.g.
  driven by the same local Ollama model) instead of the current rule-based one —
  everything downstream (`agents/tools.py`, `AgenticGraphRAG`) is planner-agnostic.
- **Larger corpora**: `ingestion/mitre_attack.py` and `ingestion/nvd_cve.py` both expose
  live-fetch functions for the full ATT&CK STIX bundle and specific NVD CVE records.
