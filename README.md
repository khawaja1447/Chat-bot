# 📄 PDF Q&A — Hybrid RAG with a measured retriever

Ask questions across a library of PDFs and get answers that cite the document and
page they came from. Built to be **measured, not asserted**: retrieval quality is
scored against a labelled golden set on every CI run, and the numbers below come
out of that harness rather than out of a README.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/Vector-FAISS-009999)
![BM25](https://img.shields.io/badge/Lexical-BM25-6aa84f)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## The headline

Adding a cross-encoder reranking stage moved retrieval accuracy substantially on a
109-question labelled set over a deliberately hard 6-document corpus:

| Configuration | Recall@1 | Recall@5 | MRR@10 | p50 latency |
|---|---|---|---|---|
| Dense only (FAISS + MiniLM) | 0.661 | 0.927 | 0.774 | 9 ms |
| \+ BM25 hybrid (RRF) | 0.743 | 0.972 | 0.844 | 9 ms |
| **\+ cross-encoder rerank** | **0.927** | **0.991** | **0.958** | 137 ms |

**Recall@1 +40%, MRR +24%** over the dense baseline, for about 130 ms of
extra retrieval latency. Accuracy is deterministic and reproducible; the latency
column is one CPU under load and will differ on your machine — regenerate it with
`python eval/run_eval.py --write`.

Two things that table says which most RAG READMEs would quietly omit:

- **Hybrid retrieval does not help once reranking is on.** BM25 is worth a large
  +12% Recall@1 on its own, but hybrid+rerank scored 0.908 against dense+rerank's
  0.927 — a two-question difference on 109, stable across every candidate depth
  tested, and not a gap this set can resolve. It is left on by default because it
  protects the exact-rare-token case (a SKU, an error code, `p99`) that the golden
  set under-represents, and it costs no measurable latency — but the honest reading
  is "neutral here", not "helps".
- **One case still fails.** "Can I use my own phone for work?" retrieves the
  handbook's device section instead of the security policy's. It is listed in
  [`docs/EVALUATION.md`](docs/EVALUATION.md) rather than quietly dropped.

Full report, including the chunk-size sweep that chose the defaults:
**[docs/EVALUATION.md](docs/EVALUATION.md)**

---

## How retrieval works

```
                        ┌─ FAISS dense search ──┐
question ─▶ rewrite ─▶ ─┤                        ├─▶ RRF ─▶ cross-encoder ─▶ Groq LLM     
            (if a       └─ BM25 lexical search ─┘  fuse     rerank            + citations
             follow-up)
```

Each stage earns its place:

| Stage | Why it is there |
|---|---|
| **Query rewrite** | "And the second one?" embeds to noise. One cheap LLM call turns it into a standalone question *before* retrieval — passing history only to the generator is too late, the wrong chunks are already fetched. |
| **Dense (FAISS + MiniLM)** | Finds passages that mean the same thing. "How much holiday do I get" → "22 days of paid annual leave", no shared content words. |
| **BM25** | Catches the exact rare token a 384-dim embedding smears into its neighbours: a product code, an error string, `p99`. |
| **RRF** | Fuses the two on *rank*, so a BM25 score of 14.2 and a cosine similarity of 0.83 never have to be made commensurate. |
| **Cross-encoder rerank** | Reads the query and the passage *together* instead of comparing two independently-computed vectors. Far more accurate, far too slow for the whole corpus — which is exactly why it runs last over a short candidate list. |

Embeddings and reranking run locally on CPU. Only the question and the retrieved
excerpts are sent to Groq.

---

## Evaluation

The part worth reading the code for.

`eval/golden_set.yaml` holds 109 questions, each labelled with the document and
page that actually answers it. `eval/run_eval.py` scores retrieval against it with
**no LLM in the loop** — deterministic, no API key, runs in CI.

The corpus is built to be hard rather than flattering:

- **Two annual reports**, 2023 and 2024, mirroring each other heading for heading
  with different figures. Every numeric question has a plausible wrong answer one
  document away, so topic matching alone fails.
- **Overlapping policy documents.** Device rules, access revocation and secret
  handling each appear in two of the handbook, runbook and security policy, with
  different specifics.
- **Questions tagged by type** — `lookup`, `paraphrase`, `numeric`, `distractor`,
  `temporal` — so a regression shows up as *which kind* of question broke.

```bash
python eval/run_eval.py --sweep-chunks --write
```

CI fails the build if Recall@1, Recall@5 or MRR drops more than 0.02 below the
recorded floor (`eval/check_thresholds.py`). A gate you only look at when it is
green is decoration.

> The chunk-size sweep is why the default is 900 characters and not the round 1000
> this project started with. Across 600–2500 the spread in Recall@1 was 0.780 to
> 0.908 — a bigger difference than between two of the retrieval strategies above,
> and not something you can reason your way to.

---

## Quick start

**Docker** — everything, including model weights, baked in:

```bash
docker compose up --build
```

**Local:**

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env && streamlit run app.py
```

Put a free [Groq API key](https://console.groq.com/keys) in `.env` as
`GROQ_API_KEY`, or paste it into the sidebar. Then upload PDFs — or point it at
the six documents in `samples/`.

Groq retires model ids and gates some behind Enterprise plans, so the id is not
hard-coded. To see what a key can actually use:

```bash
python scripts/list_models.py
```

Then set `RAGBOT_LLM_MODEL` in `.env` to override the default
(`openai/gpt-oss-120b`). A retired id surfaces as a plain message telling you to
do exactly this, rather than a raw 404.

The index persists to `.index/` and reloads on restart, so you re-embed only when
documents change.

---

## What is in the app

- **Document library** — upload several PDFs, remove individual ones, or restrict
  a question to a subset.
- **Streamed answers** — tokens appear as they are generated.
- **Citations** — every answer cites `document p.N`; excerpts are shown with the
  retrieval trace that produced them (`dense #2 · bm25 #1 · rerank +4.31`), which
  makes a bad answer diagnosable instead of mysterious.
- **Live retrieval settings** — toggle hybrid and reranking in the sidebar and
  watch the passages change.
- **Structured JSON logs** — one object per line, carrying the query, the rewrite,
  the citations served, and per-stage latency.

---

## Project structure

```
ragbot/
  config.py         one frozen dataclass of knobs — what the eval sweeps
  ingest.py         PDF → per-page text → chunks, page identity preserved
  store.py          FAISS + BM25 over one chunk list, persisted to disk
  retrieval.py      dense + sparse → RRF → cross-encoder rerank
  llm.py            Groq calls, blocking and streaming
  pipeline.py       rewrite → retrieve → generate
  observability.py  structured JSON logging
eval/
  golden_set.yaml     109 labelled questions
  run_eval.py         metrics, sweeps, report generation
  check_thresholds.py CI regression gate
scripts/
  build_samples.py  regenerates the sample corpus
  list_models.py    lists the Groq models a key can actually use
samples/            six-document corpus, generated by scripts/build_samples.py
tests/              55 tests; the fast suite uses fake embeddings and stays offline
app.py              Streamlit UI; Dockerfile and compose.yaml at the repo root
```

`ragbot/` imports no Streamlit, which is what lets the eval harness and the tests
run headless.

---

## Tests

```bash
pip install -r requirements-dev.txt && pytest && ruff check .
```

The default run is offline and takes a few seconds — embeddings are faked
deterministically. Tests that need real model weights are marked `slow` and run
separately (`pytest -m slow`), including one that guards the assumption the whole
rerank stage rests on: that the cross-encoder actually scores a relevant passage
above an irrelevant one.

---

## Limitations

- **Text PDFs only.** Scanned documents are rejected with a clear message rather
  than silently indexed as nothing. OCR is not wired up.
- **The corpus is small.** 6 documents, 29 pages, 58 chunks. Recall@5 saturates at
  that size, which is why Recall@1 and MRR are the metrics quoted. The numbers are
  a fair comparison *between retrieval strategies*; they are not evidence about
  behaviour at 10k or 1M chunks, where ANN recall and index cost start to matter.
- **The golden set is synthetic** and written by the same author as the corpus.
  It measures retrieval mechanics honestly, but it is not a substitute for
  evaluation on real user questions. Running this harness against a public
  QA-over-documents benchmark is the most useful next step.
- **Retrieval only.** Answer faithfulness and citation correctness are not scored;
  the grounding here is prompt-level, not measured.
- **In-process index.** FAISS in memory with a disk snapshot — fine to a few
  thousand chunks, not a substitute for a real vector database.
- **No auth or rate limiting.** Suitable for local use and review, not for an
  untrusted public deployment. On a shared deployment, leave `GROQ_API_KEY` unset
  so each visitor supplies their own.

## Next

Evaluation against a third-party benchmark rather than a self-authored set,
faithfulness scoring on generated answers, an OCR path for scanned PDFs, and
swapping the flat FAISS index for HNSW once the corpus outgrows exact search.

---

## License
[MIT](LICENSE)
