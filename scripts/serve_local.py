"""Minimal OpenAI-compatible server: Qwen2.5 chat + BGE-M3 embeddings.

Useful when the cluster cannot run vLLM (old driver) and Ollama downloads are
too slow. Loads local HF weights with transformers and exposes the two endpoints
the project needs.

    python scripts/serve_local.py \
        --llm /path/to/qwen-instruct \
        --embed /path/to/bge \
        --port 18000 --llm-gpu 0 --embed-gpu 1
"""

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


class LocalModels:
    def __init__(self, llm_path, embed_path, llm_gpu=0, embed_gpu=1):
        self.llm_tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{llm_gpu}", trust_remote_code=True
        ).eval()
        self.emb_tok = AutoTokenizer.from_pretrained(embed_path)
        self.emb = AutoModel.from_pretrained(
            embed_path, torch_dtype=torch.float32, device_map=f"cuda:{embed_gpu}"
        ).eval()
        self.llm_lock = threading.Lock()
        self.emb_lock = threading.Lock()

    def chat(self, messages, max_new_tokens=512, temperature=0.0):
        text = self.llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_tok(text, return_tensors="pt").to(self.llm.device)
        kwargs = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature))
        with self.llm_lock, torch.inference_mode():
            out = self.llm.generate(**inputs, **kwargs)
        return self.llm_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def embed(self, texts):
        batch = self.emb_tok(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.emb.device)
        with self.emb_lock, torch.inference_mode():
            out = self.emb(**batch)
        vecs = torch.nn.functional.normalize(out.last_hidden_state[:, 0].float(), dim=-1)
        return vecs.cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    models = None

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "qwen2.5-7b", "object": "model"},
                {"id": "bge-m3", "object": "model"},
            ]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/v1/chat/completions"):
                messages = body.get("messages") or []
                max_tokens = int(body.get("max_tokens") or body.get("max_new_tokens") or 1024)
                temperature = float(body.get("temperature") or 0.0)
                content = self.models.chat(
                    messages, max_new_tokens=min(max_tokens, 4096), temperature=temperature
                )
                self._send(200, {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "model": body.get("model", "qwen2.5-7b"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                })
            elif self.path.startswith("/v1/embeddings"):
                inp = body.get("input")
                texts = [inp] if isinstance(inp, str) else list(inp or [])
                vectors = self.models.embed(texts)
                self._send(200, {
                    "object": "list",
                    "model": body.get("model", "bge-m3"),
                    "data": [{"object": "embedding", "index": i, "embedding": v}
                             for i, v in enumerate(vectors)],
                })
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--embed", required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--embed-gpu", type=int, default=1)
    args = parser.parse_args()
    print("loading models ...", flush=True)
    Handler.models = LocalModels(args.llm, args.embed, args.llm_gpu, args.embed_gpu)
    print(f"serving on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
