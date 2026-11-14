# Reproduction Roadmap: MMGraph-Agent → MMGraph-Med

This document maps the paper's sections to code, lists the benchmark data, and
gives a milestone checklist for a full reproduction.

## 0. Chosen reproduction mode (decided 2026-11-12)

**Main path = adapted version**: DeepSeek for Chinese clinical text extraction and
Qwen-VL for image decoding; the memory (§3.2) and GRPO policy (§3.3) are
reproduced faithfully and are independent of the construction method.

- Config: [`configs/adapted.toml`](../configs/adapted.toml)
- Optional baseline: zero-API (spaCy + frozen CLIP) via
  [`configs/cluster.toml`](../configs/cluster.toml), for a cost-vs-quality
  comparison / ablation.

See [`docs/API_CHOICES.md`](API_CHOICES.md) §7 for the feasibility assessment.

## 1. Module-to-paper mapping

| Paper | File(s) | What is implemented | What is a stub |
| --- | --- | --- | --- |
| §3.1 zero-API hypergraph | `hypergraph/parser.py`, `encoder.py`, `builder.py`, `store.py` | modality-tagged units, n-ary hyperedges, entity bank, image links, JSON persistence | real CLIP weights (adapter exists), FAISS index |
| §3.2 L1/L2 memory | `memory/cache.py` | two-tier cache, digest-bounded admission, Top-C serve, prompt-token budget | learned digest (paper uses the policy) |
| §3.3 GRPO policy | `policy/actions.py`, `grpo.py`, `infer.py` | action space, prompt template, hit-aware reward, heuristic demo policy | GRPO training loop, Qwen2.5-VL-3B weights |
| §4 evaluation | `eval/*` | token-F1, answer recall, episode runner | 23 real dataset loaders |

## 2. Benchmark datasets (23, three regimes)

- **Knowledge-only (9)**: DocBench, MMLongBench-Doc, ViDoSeek, 2Wiki, HotpotQA,
  Musique, NQ, PopQA, TriviaQA.
- **Web-only (5)**: FVQA, InfoSeek, MMSearch, SimpleVQA, LiveVQA.
- **Knowledge+Web (9)**: MedXpertQA, FinMME, We-Math2.0, RBench, MicroVQA,
  EEmo-Bench, ArtQuest, AstroLLaVA, GeoGrid-Bench.

ViDoSeek, PopQA, TriviaQA and all Web-only datasets are held out as OOD.

## 3. Milestones

### M1 — CPU skeleton (done in this repo)

- [x] modality-aware parser + regex entity fallback
- [x] hashing embedder + CLIP adapter interface
- [x] n-ary hypergraph builder + persistence
- [x] two-tier L1/L2 memory with digest-bounded admission
- [x] unified offline/web retrieval interface
- [x] action space + prompt template + parse
- [x] GRPO hit-aware reward (pure Python, unit tested)
- [x] heuristic demo policy + end-to-end smoke evaluation
- [x] interactive bipartite hypergraph viewer (adapted from ai-knowledge-graph)
- [x] Chinese sentence/token handling + clinical seed lexicon fallback
- [x] pluggable extractor: zero-API regex (default) or LLM API (DeepSeek/Qwen)
- [x] optional API captioner for raw image decoding
- [x] 39 unit tests

### M2 — real components + first real dataset

- [x] configure `[extractor]` with a DeepSeek/Qwen key and run on the Chinese corpus
  (verified 2026-11-12 via `scripts/check_api.py` + `run_build_zh_local.bat`)
- [x] configure `[captioner]` with a Qwen-VL model for raw medical images
  (verified with `qwen3-vl-plus` on the synthetic ECG demo image)
- [ ] `parser = "spacy"` with `en_core_web_sm` (or a Chinese NER model)
- [ ] `embedder = "clip"` with a frozen checkpoint; embed real image files
- [ ] build a FAISS index over image/entity vectors
- [ ] implement one dataset loader (DocBench or FVQA) in `eval/datasets.py`
- [ ] run K-only and Web-only evaluation; report token-F1
- [ ] add an image-search evaluation split (video/audio in the paper)

### M3 — full paper scale

- [ ] all 23 datasets and three regimes
- [ ] Qwen2.5-VL-3B policy; implement `GRPOTrainer.train()`
- [ ] GRPO: G=8, 75 steps, 600 examples, 8×A100 (per paper §4.1)
- [ ] ablations: w/o hypergraph, w/o memory, w/o policy, w/o retrieval
- [ ] cost/context Pareto and long-chain episode analysis
- [ ] report per-dataset and per-regime tables

## 4. Commands

```bash
# CPU smoke
python scripts/run_smoke.py --verbose

# build + persist hypergraph
python scripts/build_hypergraph.py --config configs/smoke.toml --out out/mmkg.json

# tests
python -m unittest discover -s tests

# GPU training template (requires extras)
python -m pip install -e ".[gpu]"
python scripts/train_grpo.py --config configs/default.toml --output-dir out/grpo
```

## 5. Known gaps / honest notes

- The heuristic policy is **not** the paper's GRPO agent; smoke F1 validates the
  plumbing, not the reported numbers.
- The hashing embedder is a stand-in for CLIP and has no real cross-modal
  semantics; image retrieval only uses captions in the CPU path.
- Entity extraction falls back to capitalised phrases when spaCy is absent.
- The 23 dataset loaders and the GRPO loop are the main remaining work.

## 6. Attribution and academic integrity

This is an independent reproduction. If you publish results from it you must
cite the original MMGraph-Agent and M³KG-RAG papers and disclose the upstream
code base (ai-knowledge-graph, Apache-2.0). Do not present upstream code as your
own original work.
