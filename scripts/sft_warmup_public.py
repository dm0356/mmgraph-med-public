"""SFT warm-up for the retrieval-and-answer policy.

Builds legal two-turn trajectories with a rule-based teacher:
  turn 1: <search>question</search>
  turn 2: <answer>gold</answer>
and fine-tunes a LoRA adapter with cross-entropy on the completion tokens only.

Usage:
    PYTHONPATH=/path/to/pylibs python scripts/sft_warmup_public.py \
        --store /path/to/public_pilot/pilot_big/mmkg_big_bge.json \
        --qa-train /path/to/public_pilot/pilot_big/qa_train.jsonl \
        --limit 500 --steps 300 --device cuda:2
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.hypergraph import Store, get_embedder  # noqa: E402
from mmgraph_agent.policy.agent_env import SYSTEM_PROMPT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/public_pilot_bge.toml")
    parser.add_argument("--store", default="/path/to/public_pilot/pilot_big/mmkg_big_bge.json")
    parser.add_argument("--qa-train", default="/path/to/public_pilot/pilot_big/qa_train.jsonl")
    parser.add_argument("--model", default="/path/to/qwen-instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-dir", default="/path/to/public_pilot/sft_adapter")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    store = Store.load(args.store)
    embedder = get_embedder(config)

    rows = [json.loads(line) for line in Path(args.qa_train).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def chat(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def make_pairs(row):
        question, gold = row["question"], row["answer"]
        vec = embedder.encode_texts([question])[0]
        hits = store.search_text(vec, top_k=2, threshold=0.0)
        observation = "\n".join(hit["hyperedge"]["text"][:300] for hit in hits)
        base = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n请只输出一个动作。"},
        ]
        search = f"<search>{question}</search>"
        answer_context = base + [
            {"role": "assistant", "content": search},
            {"role": "user", "content": observation + "\n请继续：只输出一个动作。"},
        ]
        return [
            (chat(base), search),
            (chat(answer_context), f"<answer>{gold}</answer>"),
        ]

    samples = [pair for row in rows for pair in make_pairs(row)]
    print(f"training pairs: {len(samples)}", flush=True)
    history = []
    for step in range(args.steps):
        context, completion = samples[step % len(samples)]
        context_ids = tokenizer(context, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to(args.device)
        full_ids = tokenizer(context + completion, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to(args.device)
        if full_ids.shape[1] <= context_ids.shape[1]:
            continue
        labels = full_ids.clone()
        labels[:, : context_ids.shape[1]] = -100
        output = model(input_ids=full_ids, labels=labels)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0:
            entry = {"step": step + 1, "loss": float(loss.detach().cpu())}
            history.append(entry)
            print(json.dumps(entry), flush=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print("saved adapter to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
