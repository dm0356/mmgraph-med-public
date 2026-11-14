"""Probe both configured AI endpoints:

* ``[extractor]`` -- text entity/relation extraction (e.g. DeepSeek)
* ``[captioner]`` -- image decoding (e.g. Qwen-VL)

Usage:
    # PowerShell
    $env:DEEPSEEK_API_KEY  = "sk-..."
    $env:DASHSCOPE_API_KEY = "sk-..."
    $env:PYTHONPATH = "src"
    python scripts/check_api.py

The two keys are intentionally separate (``[extractor].api_key_env`` and
``[captioner].api_key_env``), so one can be DeepSeek and the other Qwen.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:  # nicer Chinese output on Windows terminals
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse  # noqa: E402

from mmgraph_agent.config import load_config  # noqa: E402
from mmgraph_agent.hypergraph.captioner import OpenAICompatibleCaptioner  # noqa: E402
from mmgraph_agent.hypergraph.llm_extractor import LLMExtractor  # noqa: E402

SAMPLE = "患者李女士有2型糖尿病病史。空腹血糖为8.6毫摩尔每升，医生建议服用降糖药。"


def _show_key(key_env: str) -> bool:
    present = bool(os.environ.get(key_env))
    print(f"  key env  : {key_env} -> {'set' if present else 'NOT SET'}")
    return present


def _hint_for(exc: Exception) -> str:
    text = str(exc)
    if "404" in text:
        return (
            "  hint: 404 常见原因：模型名不可用/未开通/地域不符。\n"
            "        Qwen-VL 可试 qwen3-vl-plus / qwen3-vl-max；\n"
            "        若账号是国际站，base_url 用 https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
    if "401" in text or "invalid_api_key" in text.lower():
        return "  hint: 401/无效 key，检查环境变量是否设置、是否复制完整。"
    return ""


def check_extractor(config: dict) -> bool | None:
    ext = config.get("extractor", {})
    base_url = (ext.get("base_url") or "").strip()
    key_env = ext.get("api_key_env") or "DEEPSEEK_API_KEY"
    print("[extractor] text entity/relation extraction")
    print(f"  base_url : {base_url or '(empty)'}")
    print(f"  model    : {ext.get('model')}")
    _show_key(key_env)
    if not base_url:
        print("  [!] 未配置 [extractor].base_url，跳过（零 API 正则路径仍可用）。")
        return None
    try:
        extractor = LLMExtractor(config)
        print("  ping     :", extractor.ping().strip()[:80])
        result = extractor.extract_rich(SAMPLE)
        print("  entities :", result["entities"])
        print("  relations:", result["relations"])
        return True
    except Exception as exc:  # noqa: BLE001 - surface the real network/config error
        print("  ping FAILED:", exc)
        print(_hint_for(exc))
        return False


def check_captioner(config: dict, image: str | None = None) -> bool | None:
    cap = config.get("captioner", {})
    if not cap.get("enabled"):
        print("[captioner] image decoding: disabled (enabled = false) -- skipped")
        return None
    base_url = (cap.get("base_url") or "").strip()
    key_env = cap.get("api_key_env") or "DASHSCOPE_API_KEY"
    print("[captioner] image decoding")
    print(f"  base_url : {base_url or '(empty)'}")
    print(f"  model    : {cap.get('model')}")
    _show_key(key_env)
    if not base_url:
        print("  [!] 未配置 [captioner].base_url，跳过。")
        return False
    try:
        captioner = OpenAICompatibleCaptioner(config)
        print("  ping     :", captioner.ping().strip()[:80])
        if image:
            description = captioner.describe({"path": image})
            print("  image    :", image)
            print("  description:", description or "(none)")
        else:
            print("  hint     : 传 --image <路径> 可测试真实图片描述")
        return True
    except Exception as exc:  # noqa: BLE001
        print("  ping FAILED:", exc)
        print(_hint_for(exc))
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the configured extraction and captioning APIs.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--text", default=SAMPLE, help="sample text for the extractor")
    parser.add_argument("--image", default=None, help="optional image path to test captioning")
    args = parser.parse_args()

    config = load_config(args.config)
    extractor_ok = check_extractor(config)
    print()
    captioner_ok = check_captioner(config, args.image)

    configured = [ok for ok in (extractor_ok, captioner_ok) if ok is not None]
    print("\nsummary:")
    if not configured:
        print("  未配置任何 API：当前走零 API 路径（regex + hashing）。")
        return 2
    if all(configured):
        print("  配置的接口全部可用。")
        return 0
    print("  有接口失败，请检查 base_url / 模型名 / 环境变量 key。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
