"""A minimal but real GRPO trainer for the retrieval-and-answer policy.

This is a *simplified* GRPO: group-relative advantage, PPO-style ratio clipping
and a KL penalty to the frozen reference policy. It is deliberately sequential
(one prompt at a time, ``G`` sampled rollouts) so it runs on a single 48 GB GPU
with a 7B Qwen model + LoRA. It is not the paper's full distributed recipe.

Requires: torch, transformers, peft. Imported lazily so CPU users can still run
the rest of the package.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval.metrics import token_f1
from .agent_env import SYSTEM_PROMPT, RetrievalEnv
from .grpo import compute_reward


@dataclass
class Rollout:
    question: str
    gold: str
    answer: str
    reward: float
    searches: int
    well_formed: bool
    segments: list[tuple[str, str, float]]  # (context, completion, old_mean_logp)
    tokens: int


class GRPOTrainer:
    def __init__(
        self,
        config: dict,
        env: RetrievalEnv,
        model_path: str,
        device: str = "cuda:2",
        lora_r: int = 8,
        lr: float = 1e-4,
        group_size: int = 2,
        max_new_tokens: int = 96,
        temperature: float = 0.8,
        beta: float = 0.02,
        clip: float = 0.2,
    ) -> None:
        self.config = config
        self.env = env
        self.model_path = model_path
        self.device = device
        self.lora_r = lora_r
        self.lr = lr
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.beta = beta
        self.clip = clip
        self._loaded = False

    # -- model ----------------------------------------------------------
    def load(self, adapter_path: str | None = None) -> None:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.config.use_cache = False
        if adapter_path:
            # load the SFT adapter directly as the only trainable adapter;
            # avoids the "default vs sft" naming bug that silently dropped GRPO updates
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
        else:
            lora = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_r * 2,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.lr,
        )
        self._loaded = True

    # -- generation / log-prob -----------------------------------------
    def _chat(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate(self, context: str, greedy: bool = False) -> str:
        torch = self.torch
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=3072).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=not greedy,
                temperature=self.temperature if not greedy else None,
                top_p=0.9 if not greedy else None,
                repetition_penalty=1.05,
                stop_strings=["</search>", "</answer>"],
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _logp(self, context: str, completion: str, grad: bool, reference: bool = False):
        torch = self.torch
        context_ids = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=3072).input_ids.to(self.device)
        full_ids = self.tokenizer(context + completion, return_tensors="pt", truncation=True, max_length=3072).input_ids.to(self.device)
        if full_ids.shape[1] <= context_ids.shape[1]:
            return None
        context_len = context_ids.shape[1]
        tokens = full_ids[:, context_len:]
        if reference:
            with torch.no_grad(), self.model.disable_adapter():
                logits = self.model(input_ids=full_ids).logits[:, context_len - 1:-1, :]
        elif grad:
            logits = self.model(input_ids=full_ids).logits[:, context_len - 1:-1, :]
        else:
            with torch.no_grad():
                logits = self.model(input_ids=full_ids).logits[:, context_len - 1:-1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(), tokens.shape[1]

    # -- rollout --------------------------------------------------------
    def rollout(self, question: str, gold: str, greedy: bool = False) -> Rollout:
        self.env.reset(question)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n请只输出一个动作。"},
        ]
        context = self._chat(messages)
        segments: list[tuple[str, str, float]] = []
        total_tokens = 0
        for _ in range(self.env.max_rounds + 1):
            completion = self._generate(context, greedy=greedy)
            logp = self._logp(context, completion, grad=False)
            old_mean = float(logp[0]) / max(1, logp[1]) if logp is not None else 0.0
            segments.append((context, completion, old_mean))
            total_tokens += logp[1] if logp is not None else 0
            step = self.env.parse(completion)
            if step.done:
                break
            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": step.observation + "\n请继续：只输出一个动作。"})
            context = self._chat(messages)

        well_formed = bool(self.env.answer)
        reward = compute_reward(
            token_f1(self.env.answer, gold) if well_formed else 0.0,
            self.env.search_count,
            well_formed,
            self.config,
        )
        return Rollout(
            question=question,
            gold=gold,
            answer=self.env.answer,
            reward=float(reward),
            searches=self.env.search_count,
            well_formed=well_formed,
            segments=segments,
            tokens=total_tokens,
        )

    # -- update ---------------------------------------------------------
    def train(self, pairs: list[tuple[str, str]], steps: int, log_every: int = 1,
              output_dir: str | None = None) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        torch = self.torch
        history: list[dict[str, Any]] = []
        data = pairs * max(1, math.ceil(steps / max(1, len(pairs))))
        for step in range(steps):
            question, gold = data[step % len(data)]
            rollouts = [self.rollout(question, gold) for _ in range(self.group_size)]
            rewards = torch.tensor([rollout.reward for rollout in rollouts], dtype=torch.float32, device=self.device)
            mean, std = rewards.mean(), rewards.std(unbiased=False)
            advantages = (rewards - mean) / (std + 1e-6)

            loss_value = torch.zeros((), device=self.device)
            self.optimizer.zero_grad()
            for rollout, advantage in zip(rollouts, advantages):
                for context, completion, old_mean in rollout.segments:
                    new = self._logp(context, completion, grad=True)
                    ref = self._logp(context, completion, grad=False, reference=True)
                    if new is None:
                        continue
                    new_sum, tokens = new
                    new_mean = new_sum / max(1, tokens)
                    ratio = torch.exp(new_mean - old_mean)
                    surrogate = torch.minimum(
                        ratio * advantage,
                        torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage,
                    )
                    ref_mean = float(ref[0]) / max(1, ref[1]) if ref is not None else new_mean.detach()
                    kl = new_mean - ref_mean
                    segment_loss = -(surrogate - self.beta * kl) / (
                        self.group_size * max(1, len(rollout.segments))
                    )
                    # backward per segment: frees the activation graph immediately
                    segment_loss.backward()
                    loss_value = loss_value + segment_loss.detach()
                self.torch.cuda.empty_cache()

            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad], 1.0
            )
            self.optimizer.step()
            grad_norm = sum(
                float(parameter.grad.detach().norm())
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )

            entry = {
                "step": step + 1,
                "reward_mean": float(mean.detach().cpu()),
                "reward_max": float(rewards.max().detach().cpu()),
                "loss": float(loss_value.detach().cpu()),
                "grad_norm": grad_norm,
                "answer": rollouts[0].answer,
                "first_completion": rollouts[0].segments[0][1][:160] if rollouts[0].segments else "",
                "gold": gold,
                "searches": rollouts[0].searches,
            }
            history.append(entry)
            if log_every and (step + 1) % log_every == 0:
                print(json.dumps(entry, ensure_ascii=False), flush=True)

        if output_dir:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            Path(output_dir, "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return {"steps": steps, "history": history}
