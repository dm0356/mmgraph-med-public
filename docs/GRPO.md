# GRPO training (public pilot)

This document describes the **functional GRPO trainer** added for the public ECG
pilot. It is a simplified single-GPU implementation, not the paper's distributed
Qwen2.5-VL-3B recipe.

## 1. What is implemented

| Component | File | What it does |
| --- | --- | --- |
| Retrieval environment | `src/mmgraph_agent/policy/agent_env.py` | parses `<search>…</search>` / `<answer>…</answer>`, executes hypergraph retrieval, appends evidence, enforces max rounds |
| Reward | `src/mmgraph_agent/policy/grpo.py` | paper Eq. (6): token-F1 × concave call regulator − structural penalty |
| Trainer | `src/mmgraph_agent/policy/grpo_trainer.py` | Qwen2.5-7B + LoRA, group-relative advantage, PPO-style ratio clipping, KL to frozen reference, per-segment backward |
| SFT warm-up | `scripts/sft_warmup_public.py` | 300-step LoRA SFT on legal `<search>/<answer>` trajectories |
| Data | `scripts/make_public_qa.py` | group-aware QA from VFDB windows and PTB-XL metadata |
| Entry point | `scripts/train_grpo_public.py` | loads store/embedder, optional SFT adapter, trains, evaluates before/after, saves the LoRA adapter |

## 2. What differs from the paper

| Paper | This implementation |
| --- | --- |
| Qwen2.5-VL-3B, 8×A100, distributed GRPO | Qwen2.5-7B text, 1×L40S, LoRA |
| token-level GRPO ratio | sequence-level mean log-prob ratio |
| multimodal actions (text/image search) | text search + answer (image action not exercised) |
| 23 datasets | 800 QA generated from the public ECG pilot |
| vLLM rollouts | `transformers.generate` rollouts |

These differences are deliberate: the goal here is to **complete and validate the
algorithm**, not to reproduce the paper's numbers.

## 3. Commands (cluster)

```bash
# 1) generate group-aware QA (VFDB records / PTB-XL patients never cross splits)
python scripts/make_public_qa.py --limit 12000
# -> 6354 train / 846 eval

# 2) SFT warm-up on legal trajectories (1000 pairs, 300 steps)
PYTHONPATH=/path/to/pylibs python scripts/sft_warmup_public.py \
    --store /path/to/public_pilot/pilot_big/mmkg_big_bge.json \
    --qa-train /path/to/public_pilot/pilot_big/qa_train.jsonl \
    --limit 500 --steps 300 --device cuda:2

# 3) GRPO initialised from the SFT adapter
PYTHONPATH=/path/to/pylibs python scripts/train_grpo_public.py \
    --store /path/to/public_pilot/pilot_big/mmkg_big_bge.json \
    --qa-train /path/to/public_pilot/pilot_big/qa_train.jsonl \
    --qa-eval /path/to/public_pilot/pilot_big/qa_eval.jsonl \
    --init-adapter /path/to/public_pilot/sft_adapter \
    --steps 10 --group-size 2 --limit 32 --eval-limit 16 \
    --device cuda:2 --lr 5e-5 --output-dir /path/to/public_pilot/grpo_sft_v2
```

Adapters are saved to `--output-dir` together with `history.json`.

## 4. Results (same 84 held-out prompts, 7 categories, greedy decoding)

| Policy | exact | token-F1 |
| --- | --- | --- |
| base Qwen2.5-7B | 0.214 | 0.258 |
| + SFT warm-up (300 steps) | **0.631** | **0.738** |
| + GRPO 100 steps, lr 5e-5 (SFT init) | 0.536 | 0.661 |
| + GRPO 30 steps, lr 1e-5, beta 0.1 (SFT init) | 0.631 | 0.738 |

Category breakdown (SFT):

| category | exact | notes |
| --- | --- | --- |
| `ptbxl_code` | 0.917 | diagnosis code retrieval |
| `sex` | 0.917 | metadata lookup |
| `label` | 0.750 | pre/va/baseline |
| `rhythm` | 0.750 | N/VT/VFL/… |
| `hf_ratio` | 0.583 | numeric |
| `entropy` | 0.333 | numeric |
| `pnn50` | 0.167 | numeric |

**Interpretation (honest):**

1. **SFT is the main driver**: 0.214 → 0.631 exact, 0.258 → 0.738 F1.
2. **High-LR GRPO collapses**: 100 steps at `lr=5e-5` degrade to 0.536; around
   step 50 the model emits English or query-unrelated actions
   (`<search>person</search>`, `<answer>greeting</answer>`).
3. **Stabilised GRPO is flat**: 30 steps at `lr=1e-5`, `beta=0.1` stays exactly
   at the SFT level (0.631 / 0.738) — no gain, no degradation.
4. Therefore, on this setup **GRPO does not yet beat SFT**. The most likely
   causes are:
   - the QA tasks are too easy/saturated for categorical answers;
   - token-F1 is a weak reward for numeric answers (`0.64` vs `0.63` gets ~0.5);
   - the SFT adapter already captures the retrieval-and-answer format;
   - the policy has little headroom without harder multi-hop questions.

This is a **valid negative result**, not a bug: the algorithm is implemented and
trainable, but the current reward/data do not give GRPO an advantage.

Other observations:

* The loop samples `G` rollouts per question, executes retrieval actions,
  computes rewards, group-normalises advantages, applies the clipped surrogate
  and KL penalty, and updates LoRA weights.
* Gradients are non-zero, so weights really update.
* **Format compliance**: Qwen chat template + stopping at `</search>` /
  `</answer>` is required; without it the base model produces plain text and
  always scores −0.5 with zero gradient.
* **Adapter bug**: creating a fresh LoRA and then loading the SFT adapter under a
  second name silently lost GRPO updates (SFT+GRPO scored exactly like base).
  Loading with `PeftModel.from_pretrained` as the only adapter fixes it.
* Some numeric categories (`hr`, `rr_sd`, `rmssd`, `age`) are missing from the
  eval because the gold string format (`85次/分`) does not exactly match the
  document text (`85 次/分`); fix the QA formatter before the next evaluation.

Saved adapters: `/path/to/public_pilot/sft_adapter`,
`/path/to/public_pilot/grpo_sft_v3` (100 steps),
`/path/to/public_pilot/grpo_sft_v4` (30 steps, stabilised).

## 5. Known limitations / next steps

1. **Format compliance**: wrap the prompt in the Qwen chat template and add a
   short supervised warm-up on valid `<search>/<answer>` trajectories.
2. **More QA**: 800 is enough for a smoke run, not for convergence; expand to
   5k–20k with harder, multi-hop questions.
3. **Model scale**: 7B LoRA is a stand-in; the paper uses Qwen2.5-VL-3B. Moving
   to 3B + vision is the next step.
4. **Rollout engine**: `transformers.generate` is slow; vLLM/SGLang would be
   needed for a real run (and vLLM must be built for CUDA 12.1 on this cluster).
5. **Evaluation**: reward is token-F1; clinical evaluation needs AUROC,
   sensitivity at fixed lead time and calibration on held-out patients.
6. **Image action**: `IMAGE_SEARCH` exists in the action space but the public
   pilot has no images; adding ACDC/echo data would exercise it.

## 6. Why public data is valid for GRPO

GRPO only needs **questions with verifiable answers** and an environment that can
execute retrieval. The public pilot provides 660 documents and 800 QA pairs, so
the full loop (sample → retrieve → answer → reward → update) can be validated
without any hospital data. It does **not** provide clinical validity; that still
requires the hospital cohort.
