# 集群环境搭建（L40S × 8，48 GB/卡）

学校集群通常已装好驱动，只需准备 Python 环境。以下命令在集群登录节点执行。

## 0. 如何登录集群（Windows 用户）

集群不是本地软件，而是远程 Linux 服务器，**用 SSH 连上去**，在它的终端里敲命令。
（截图里的 `bash` 命令不能直接粘到 Windows PowerShell 里跑。）

1. 向学校/导师要：**登录节点地址、账号、密码或 SSH key、是否需要 VPN、是否用 Slurm**。
2. 打开 Windows 的 **PowerShell**（`Win+R` → 输入 `powershell` → 回车），执行：

   ```powershell
   ssh 你的用户名@登录节点地址
   ```

   第一次会问 `continue connecting?` → 输入 `yes`；密码输入时不显示，输完回车即可。
3. 登录成功后提示符类似 `zhangsan@login01:~$`，此时才能粘贴 bash 命令。
4. 登录后先验证：

   ```bash
   hostname
   whoami
   python3 --version
   nvidia-smi     # 只有 GPU 节点能看到；登录节点可能没有
   sinfo          # 若使用 Slurm
   ```

5. 上传代码：

   ```bash
   # 方式一：先 push 到 GitHub，再在集群 clone
   git clone https://github.com/你的用户名/mmgraph-agent.git

   # 方式二：在 Windows PowerShell 里 scp
   scp -r D:\path\to\mmgraph-agent 你的用户名@登录节点地址:~/
   ```

6. **注意 Slurm**：不要在登录节点跑 GPU 任务；用
   `srun --gres=gpu:1 --pty bash` 申请交互环境，或写 `sbatch` 脚本排队。

> 图形化替代：MobaXterm / Xshell（SSH + 文件管理）、VS Code Remote-SSH（推荐），
> 或学校提供的网页终端 / JupyterHub。

### 0.1 常见错误：把 SSH 公钥当成命令

`ssh-ed25519 AAAA... 学号` 是 **SSH 公钥**，不是命令，直接敲进 PowerShell 会报
"无法识别为 cmdlet / 命令"。认证方式通常是下面三种之一，先问清学校用哪种：

1. **密码登录（最常见）**

   ```powershell
   ssh 学号@登录节点地址
   ```

   然后输入密码（不显示）。

2. **密钥登录**
   - `ssh-ed25519 ...` 是**公钥**，需要**上传到学校平台 / 交给管理员**；
   - 你本地要保存对应的**私钥文件**（如 `id_ed25519`，没有 `.pub` 后缀）；
   - 连接：

     ```powershell
     ssh -i C:\Users\你\.ssh\id_ed25519 学号@登录节点地址
     ```

   - **私钥绝不能外泄、不能提交 git。**

3. **网页终端 / JupyterHub**
   - 浏览器登录平台，直接开终端，无需 SSH。

> 只有公钥字符串还不够；还需要登录地址、用户名（是否学号）、认证方式、是否需 VPN。

### 0.2 用 `~/.ssh/config` 简化登录

把下面这段写进 Windows 的 `C:\Users\你\.ssh\config`：

```text
Host cluster
    HostName your.cluster.example.com
    User your_user
    Port 22
```

之后在 PowerShell 里直接：

```powershell
ssh cluster
```

**Windows 上创建配置的步骤**（PowerShell 里执行）：

```powershell
# 1. 创建 .ssh 目录（已存在会跳过）
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null

# 2. 写入 config（没有旧内容时直接覆盖；用 ASCII 避免 BOM）
$cfg = "$env:USERPROFILE\.ssh\config"; Set-Content -Path $cfg -Value @("Host cluster","    HostName your.cluster.example.com","    User your_user","    Port 22") -Encoding ascii
```

> ⚠️ PowerShell 5.1 里 `if { }` 和 `else { }` **不要分两行分别回车/粘贴**，否则会报
> "无法将 else 识别为 cmdlet"。用上面这种**单行**写法最稳；若 config 里已有其他 Host，
> 把 `Set-Content` 换成 `Add-Content` 追加。

**检查私钥是否在默认位置**：

```powershell
dir $env:USERPROFILE\.ssh
```

- 如果有 `id_ed25519`（无 `.pub` 后缀），说明私钥在默认位置，`ssh cluster` 可直接用；
- 如果私钥在别处（如 `C:\Users\你\Downloads\cluster.pem`），在 config 的 Host 块里加一行：
  `IdentityFile C:\Users\你\Downloads\cluster.pem`
- 公钥（`ssh-ed25519 ...`）是已经授权给服务器的那串，**不要**再敲进 PowerShell。

> ⚠️ 集群登录地址通常是内网地址（如 `10.x.x.x`），一般需要连**校园网或学校 VPN** 才能访问。
> 如果 `ssh` 卡住/超时，先连 VPN 再试。

## 1. 创建环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果集群用 conda：

```bash
conda create -n mmkg python=3.12 -y
conda activate mmkg
```

## 2. 安装 PyTorch（CUDA 12.1）

L40S 是 Ada 架构（sm_89），CUDA 12.1 及以上都支持。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 若集群 CUDA 是 12.4，可把 `cu121` 换成 `cu124`。国内网络可加
> `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 3. 安装项目依赖

```bash
pip install -r requirements-gpu.txt
pip install -e .
python -m spacy download zh_core_web_sm   # 中文 NER（zero-API 路径需要）
python -m spacy download en_core_web_sm   # 英文语料时
```

## 4. 验证 GPU

```bash
python scripts/check_gpu.py
```

预期看到 8 张 L40S、每张约 48 GB、`bf16 supported: True`。缺包会打印 `MISSING: [...]` 并返回非零。

## 5. 数据上传

```bash
rsync -av ./data user@cluster:/path/to/mmgraph-agent/data
```

> 患者数据请遵守医院数据管理规定；必要时只上传**去标识化副本**。

## 6. 构建（zero-API 忠实复现）

`configs/cluster.toml` 默认就是零 API 路径：

```toml
[hypergraph]
parser = "spacy"
embedder = "clip"
[extractor]
mode = "regex"        # spaCy NER，不调 API
[captioner]
enabled = false       # 图像由 CLIP 编码，不调视觉 API
```

构建：

```bash
cp configs/cluster.toml configs/cluster-local.toml
python scripts/build_hypergraph.py --config configs/cluster-local.toml \
    --data-root data/your_corpus --out out/mmkg.json
```

若要用 **API 适配版**（中文临床推荐），在 `cluster-local.toml` 里：

```toml
[extractor]
mode = "llm"
base_url = "https://api.deepseek.com/v1/chat/completions"
model = "deepseek-chat"

[captioner]
enabled = true
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
model = "qwen3-vl-plus"
```

## 7. 训练（M3：GRPO）

```bash
python scripts/train_grpo.py --config configs/cluster-local.toml --output-dir out/grpo
```

当前 `GRPOTrainer.train()` 是**待实现模板**；实现后：
- 3B 模型单卡 48 GB 可放下（bf16）；
- 8 卡可用 `torchrun` / DeepSpeed；
- 论文设置：G=8、75 步、600 样例、8×A100（L40S 8×48G 足够）。

## 8. Slurm 示例（如果学校集群用 Slurm）

```bash
#!/bin/bash
#SBATCH --job-name=mmkg
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

source .venv/bin/activate
python scripts/check_gpu.py
python scripts/build_hypergraph.py --config configs/cluster-local.toml \
    --data-root data/your_corpus --out out/mmkg.json
```

## 9. 常见问题

- **torch.cuda.is_available() 为 False**：驱动/CUDA 版本与 torch 不匹配，先跑 `nvidia-smi` 看驱动版本。
- **CLIP 下载失败**：设置镜像 `export HF_ENDPOINT=https://hf-mirror.com` 再重试。
- **spaCy 模型下载失败**：可手动下载 wheel 离线安装。
- **显存不足**：3B 模型在 48 GB 上余量很大；若跑更大模型再考虑 DeepSpeed ZeRO。
