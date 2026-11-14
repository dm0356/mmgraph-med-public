"""Check the cluster GPU stack (torch/CUDA, transformers, faiss, spaCy).

Run this on the cluster login/compute node after installing dependencies.
Returns non-zero when a required component is missing, or when CUDA is not
available (use ``--allow-cpu`` for CPU-only machines).
"""

import argparse
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the GPU stack.")
    parser.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is unavailable")
    args = parser.parse_args()

    print("python :", sys.version.split()[0], "|", platform.platform())
    problems: list[str] = []

    try:
        import torch

        print("torch  :", torch.__version__)
        cuda = torch.cuda.is_available()
        print("cuda   :", cuda)
        if cuda:
            count = torch.cuda.device_count()
            print("devices:", count)
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                memory = props.total_memory / (1024 ** 3)
                print(f"  [{index}] {props.name} | {memory:.1f} GB | sm_{props.major}{props.minor}")
            print("bf16   :", torch.cuda.is_bf16_supported())
        else:
            print(f"         torch was built for CUDA {torch.version.cuda}")
            print("         If the driver is older, either upgrade the driver or install a")
            print("         torch/vLLM build matching the driver (or use configs/ollama.toml).")
            if not args.allow_cpu:
                problems.append("cuda")
    except ImportError:
        print("torch  : NOT installed")
        problems.append("torch")

    for module in ("transformers", "faiss", "spacy"):
        try:
            imported = __import__(module)
            print(f"{module:<11}:", getattr(imported, "__version__", "?"))
        except ImportError:
            print(f"{module:<11}: NOT installed")
            problems.append(module)

    if problems:
        print("\nFAILED:", ", ".join(problems))
        print("Install with: pip install -r requirements-gpu.txt")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
