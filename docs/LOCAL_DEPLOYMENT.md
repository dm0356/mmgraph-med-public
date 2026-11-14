# 本地部署（集群内跑全部模型）

## 1. 为什么要本地部署

- **隐私合规**：合作医院患者数据不出集群，不经过任何外部 API。
- **可复现**：模型版本固定，不受云端服务变更影响。
- **无 API 费用**：只消耗 GPU/电费。
- **可微调**：后续可在自己的脱敏数据上 LoRA。

## 2. API 会"干扰"吗？

**不会**。云端 API 和本地模型是**互相独立的两条路径**，通过配置文件切换：

| 路径 | 配置文件 | 适用场景 |
|---|---|---|
| 云端 API 适配版 | `configs/adapted.toml` | 快速验证、公开数据/去标识化数据 |
| **本地部署版** | `configs/local.toml` | **合作医院患者数据（推荐）** |
| zero-API（spaCy+CLIP） | `configs/cluster.toml` | 论文忠实复现基线 |

原则：**患者数据只在本地路径跑**；云端只用于公开数据或已去标识化数据。

## 3. 用 vLLM 在集群上起三个服务

L40S 48G×8 足够同时放三个 7B 级模型。

> **先在正确的 conda 环境里安装**：提示符应是 `(mmkg)` 而不是 `(base)`。
> 如果装在 base 里，后续 `vllm` 命令可能找不到或版本冲突。

```bash
conda activate mmkg
python -c "import vllm; print(vllm.__version__)"
```

### tmux 是什么（新手说明）

tmux 是"终端复用器"：在**一个 SSH 窗口**里开多个"会话"，每个会话独立运行一个程序，
**SSH 断开后程序继续跑**。

| 操作 | 命令 / 按键 |
|---|---|
| 新建会话 | `tmux new -s 名字` |
| 离开（程序继续跑） | 按 `Ctrl+B`，松开，再按 `D` |
| 看所有会话 | `tmux ls` |
| 回到某个会话 | `tmux attach -t 名字` |
| 关闭某个会话 | `tmux kill-session -t 名字` |

"三个服务各自一个 tmux 会话" = `llm` / `vl` / `embed` 各开一个，互不影响。
`start_local_servers.sh` 会自动建好这三个会话，不用手动敲。

### 一键启动（推荐）

```bash
bash scripts/start_local_servers.sh
bash scripts/check_local_servers.sh
```

`start_local_servers.sh` 会在三个 tmux 会话里分别启动：

| 服务 | 模型 | 端口 | 默认 GPU |
|---|---|---|---|
| 文本抽取 | Qwen/Qwen2.5-7B-Instruct | 18000 | 0 |
| 图像理解 | Qwen/Qwen2.5-VL-7B-Instruct | 18001 | 1 |
| 文本嵌入 | BAAI/bge-m3 | 18002 | 2 |

可用环境变量覆盖，例如：

```bash
LLM_GPU=4 VL_GPU=5 EMBED_GPU=6 bash scripts/start_local_servers.sh
```

### 手动启动（等价）

```bash
conda activate mmkg

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 18000 --max-model-len 32768
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 18001 --limit-mm-per-prompt image=1
CUDA_VISIBLE_DEVICES=2 vllm serve BAAI/bge-m3 --task embed --port 18002
```

验证：

```bash
curl http://127.0.0.1:18000/v1/models
curl http://127.0.0.1:18001/v1/models
curl http://127.0.0.1:18002/v1/models
```

每个都应返回包含模型名的 JSON；`Connection refused` 说明对应服务没起来。

### CUDA 版本兼容性

集群驱动是 535（CUDA 12.2）。如果安装的 vLLM 是 CUDA 13 构建，可能启动失败。
装完先验证：

```bash
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"
```

若 `torch.cuda.is_available()` 为 False 或 vLLM 报驱动版本不足，换一个 CUDA 12 构建的 vLLM 版本。

### 常见错误

| 现象 | 原因 | 解决 |
|---|---|---|
| `{"detail":"Not Found"}` | 端口被别的服务占用 / 路由不对 | 换端口（已用 18000+），`ss -ltnp` 查占用 |
| `Connection refused` | 服务没启动 | 检查 tmux 窗口、`vllm serve` 是否报错 |
| `can't open file '.../scripts/...'` | 不在项目目录 / 项目没上传 | `cd ~/mmgraph-agent`，先 scp 上传 |
| `vllm: command not found` | 没装 vLLM 或不在 mmkg 环境 | `conda activate mmkg; pip install vllm` |
| CUDA driver 版本不足 | vLLM 是 cu13 构建，驱动 535 只到 12.2 | 换 cu12 兼容的 vLLM 版本 |

> vLLM 参数在不同版本略有差异；若 `--limit-mm-per-prompt` 报错，用 `vllm serve --help` 查看当前版本写法。
> 国内下载模型慢可设置：`export HF_ENDPOINT=https://hf-mirror.com`。

## 4. 显存规划（8×48G）

| 服务 | 模型 | 约显存（bf16） |
|---|---|---|
| 文本抽取 | Qwen2.5-7B | ~15–18 GB |
| 图像理解 | Qwen2.5-VL-7B | ~16–19 GB |
| 文本嵌入 | BGE-M3 | ~2 GB |
| 训练（GRPO） | Qwen2.5-VL-3B/7B | 剩余卡 |

建议：三个服务各占 1 张卡（或 2 张），剩余 5–6 张留给训练。
用 `CUDA_VISIBLE_DEVICES` 指定卡，例如 `CUDA_VISIBLE_DEVICES=0 vllm serve ...`。

## 5. 切换配置

```bash
# 本地部署路径（患者数据）
python scripts/build_hypergraph.py --config configs/local.toml \
    --data-root data/your_corpus --out out/mmkg_local.json
```

`configs/local.toml` 已把：
- `[hypergraph].embedder = "api"` → 调 `[embedding].base_url`（BGE-M3）
- `[extractor].base_url` → 本地 Qwen2.5-7B
- `[captioner].base_url` → 本地 Qwen2.5-VL-7B

## 6. 当前限制

- 图像单元目前用**Qwen-VL 的描述文本**再经 BGE-M3 编码（文本空间），
  还不是 CLIP 的"图像像素向量"。真正的图文对齐（Chinese-CLIP / BiomedCLIP）
  是下一步工作，见 `docs/API_CHOICES.md`。
- 检索空间维度由 `[embedding].dim` 决定（BGE-M3 为 1024），
  换模型时记得同步。

## 7. 长任务保持运行

SSH 断开后进程会退出。用 tmux：

```bash
tmux new -s mmkg
# 在里面跑任务，按 Ctrl+B 然后 D 脱离
tmux attach -t mmkg      # 下次登录后恢复
```

## 10. 驱动太旧（CUDA 不匹配）时：用 Ollama 替代 vLLM

**现象**：`torch.cuda.is_available()` 返回 `False`，提示
`The NVIDIA driver on your system is too old`；例如 torch 显示 `+cu130`，
而集群驱动是 535（CUDA 12.2）。

**原因**：CUDA 13 需要驱动 ≥ 580。

**方案 A（推荐）**：请管理员把驱动升到 ≥ 550（CUDA 12.4）或 ≥ 580（CUDA 13），
然后安装匹配的 torch/vLLM。

**方案 B（无需升级驱动）**：用 **Ollama**，它自带 CUDA 运行时，535 驱动可用；
而且一个服务就同时提供聊天和嵌入接口（端口 11434），比三个 vLLM 服务简单。

> ⚠️ **Ollama 要装在集群上（Linux），不是 Windows 的 `D:\Ollama`**。
> 模型必须跑在集群的 L40S 上；Windows 那份只能给你本机测试用。

安装（无需 root）：

```bash
mkdir -p ~/ollama && cd ~/ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz
tar -xzf ollama.tgz
export PATH=$HOME/ollama/bin:$PATH
export OLLAMA_MAX_LOADED_MODELS=3   # 允许同时加载文本/视觉/嵌入三个模型
export OLLAMA_SCHED_SPREAD=1        # 多卡分摊
ollama serve                        # 放到 tmux 里长期运行
```

或直接用脚本：

```bash
bash scripts/install_ollama.sh
```

> `install_ollama.sh` 只做**下载/解压/提示 PATH**，**不会启动** `ollama serve`；
> 启动仍需按上面的命令在 tmux 里执行。两种方式等价，已手动装过就不用再跑脚本。

**另开一个集群终端**（再开一个 PowerShell 窗口 `ssh cluster`，或在 tmux 里新建会话），
拉模型：

```bash
bash scripts/pull_ollama_models.sh
```

`ollama pull` 必须和 `ollama serve` 在**同一台机器**（集群）上执行。

然后用 `configs/ollama.toml` 跑（模型名已按 Ollama 的命名配好）：

```bash
python scripts/build_hypergraph.py --config configs/ollama.toml \
    --data-root data/smoke_zh --out out/mmkg_ollama.json
```

模型映射：

| 用途 | Ollama 模型 | 备注 |
|---|---|---|
| 文本抽取 | `qwen2.5:7b` | 中文指令模型 |
| 图像理解 | `qwen2.5vl:7b` | 视觉语言模型 |
| 文本嵌入 | `bge-m3` | 1024 维；若不可用可换 `nomic-embed-text`（768 维，需同步 `[embedding].dim`） |

## 11. 方案 C：本地 transformers 服务（无需 vLLM / Ollama）

适用场景：驱动太旧（CUDA 13 的 vLLM 跑不了），外网下载又很慢，但集群上**已有本地模型权重**。

原理：`scripts/serve_local.py` 用 transformers 加载本地权重，暴露 OpenAI 兼容接口：

| 服务 | 模型（本集群路径） | 接口 |
|---|---|---|
| 聊天 | `/path/to/qwen-instruct`（Qwen2.5-7B） | `/v1/chat/completions` |
| 嵌入 | `/path/to/bge`（BGE-M3，1024 维） | `/v1/embeddings` |

启动（放在 tmux 里长期跑）：

```bash
cd /path/to/mmgraph-agent
PYTHONPATH=/path/to/pylibs /path/to/conda/envs/m3kg/bin/python \
  scripts/serve_local.py \
  --llm /path/to/qwen-instruct \
  --embed /path/to/bge \
  --port 18000 --llm-gpu 0 --embed-gpu 1
```

说明：`PYTHONPATH=/path/to/pylibs` 指向隔离安装的 `transformers==4.51.3`
（匹配 m3kg 环境里的 `torch 2.5.1+cu121`，不会污染原环境）。

验证：

```bash
curl http://127.0.0.1:18000/v1/models
```

建图（用 `configs/cluster_local.toml`）：

```bash
cd /path/to/mmgraph-agent
PYTHONPATH=src /path/to/conda/envs/m3kg/bin/python scripts/build_hypergraph.py \
  --config configs/cluster_local.toml \
  --data-root data/smoke_zh \
  --out /path/to/out/mmkg_local.json
```

注意：
- 这是**环境相关示例**（路径按本集群写）；换集群需改路径。
- 图像目前仍用 caption 文本编码；图像像素向量（CLIP image encoder）是后续工作。
- 验证结果：smoke_zh 建出 22 实体 / 11 超边 / 3 图像。
