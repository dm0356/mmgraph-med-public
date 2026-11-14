"""GRPO smoke training on the public ECG pilot.

Example (cluster, GPU 2):
    PYTHONPATH=/path/to/pylibs python scripts/train_grpo_public.py \
        --steps 5 --group-size 2 --limit 16 --eval-limit 8 --device cuda:2
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.hypergraph import Store, get_embedder  # noqa: E402
from mmgraph_agent.policy.agent_env import RetrievalEnv  # noqa: E402
from mmgraph_agent.policy.grpo_trainer import GRPOTrainer  # noqa: E402


def load_pairs(path: Path, limit: int) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["question"], row["answer"]) for row in rows[:limit]]


def evaluate(trainer: GRPOTrainer, pairs: list[tuple[str, str]]) -> dict:
    rewards = [trainer.rollout(question, gold, greedy=True).reward for question, gold in pairs]
    return {"n": len(rewards), "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/public_pilot_bge.toml")
    parser.add_argument("--store", default="/path/to/public_pilot/pilot/mmkg_public_pilot_bge_v2.json")
    parser.add_argument("--qa-train", default="/path/to/public_pilot/pilot/qa_train.jsonl")
    parser.add_argument("--qa-eval", default="/path/to/public_pilot/pilot/qa_eval.jsonl")
    parser.add_argument("--model", default="/path/to/qwen-instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--output-dir", default="/path/to/public_pilot/grpo_adapter")
    parser.add_argument("--init-adapter", default=None, help="optional SFT adapter to initialise from")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    store = Store.load(args.store)
    embedder = get_embedder(config)
    env = RetrievalEnv(store, embedder, config, max_rounds=args.max_rounds, top_k=args.top_k)

    train_pairs = load_pairs(Path(args.qa_train), args.limit)
    eval_pairs = load_pairs(Path(args.qa_eval), args.eval_limit)

    trainer = GRPOTrainer(
        config, env, args.model, device=args.device,
        group_size=args.group_size, lr=args.lr, max_new_tokens=args.max_new_tokens, beta=args.beta,
    )
    print("loading policy ...", flush=True)
    trainer.load(args.init_adapter)
    if args.eval_only:
        print("eval:", json.dumps(evaluate(trainer, eval_pairs), ensure_ascii=False), flush=True)
        return 0
    print("eval before:", json.dumps(evaluate(trainer, eval_pairs), ensure_ascii=False), flush=True)
    result = trainer.train(train_pairs, steps=args.steps, output_dir=args.output_dir)
    print("eval after:", json.dumps(evaluate(trainer, eval_pairs), ensure_ascii=False), flush=True)
    print("history:", json.dumps(result["history"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
