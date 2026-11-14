# 影像模态适配（Image Modality）

目标：让 ECG / 超声 / CMR 等影像不只是"caption 文本"，而是拥有**像素级向量**，为真正多模态检索和后续医学视觉模型留好接口。

## 1. 已实现

| 能力 | 位置 |
| --- | --- |
| CLIP 图像编码 | `hypergraph/encoder.py::CLIPEmbedder.encode_images` |
| 图像编码器配置 | `[image]` section（`configs/image_clip.toml` 示例） |
| 构建时自动编码 | `hypergraph/builder.py` |
| 路径解析 | 相对 `data-root` / 当前工作目录查找；找不到则退回 caption |
| 存储字段 | `image_embedding`、`image_embed_dim`、`image_embedder` |
| **CLIP 文本→图像检索** | `Store.search_images_clip` + `TwoTierMemory(cross_modal=True)` |
| **Chinese-CLIP 适配** | `encoder.py::ChineseCLIPEmbedder`（`[image].embedder="chinese-clip"`） |

每个 image unit 现在有两套向量：

```json
{
  "id": "z1_img1",
  "caption": "患者的心电图波形",
  "path": "data/smoke_zh/images/ecg_demo.png",
  "embedding": [0.01, ...],          // caption 在文本空间（hashing/BGE），用于现有文本检索
  "embedder": "hashing",
  "embed_dim": 256,
  "image_embedding": [0.02, ...],    // CLIP 像素向量（512 维）
  "image_embedder": "clip",
  "image_embed_dim": 512
}
```

## 2. 为什么保留两套向量

- **文本检索链路不变**：`TwoTierMemory.retrieve` 用文本 query 与 caption 向量比较；
- **像素向量单独存**：用于 image-to-image、CLIP 文本→图像、后续跨模态对齐；
- 把两套空间硬拼在一起会破坏余弦相似度，所以显式分开，文档写清楚。

> 这是"框架适配"的正确做法：先保证链路稳定，再逐步接入跨模态融合。

## 3. 配置

```toml
[hypergraph]
embedder = "hashing"      # 文本空间；集群上可换 "api" (BGE-M3)
embed_dim = 256

[image]
embedder = "clip"         # none | clip
model = "openai/clip-vit-base-patch32"
dim = 512
device = "cpu"            # 集群可改 "cuda:2"
```

本地快照可用环境变量覆盖（避免把绝对路径写进仓库）：

```bash
export MMGRAPH_CLIP_MODEL=/path/to/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/<hash>
```

## 4. 命令

```bash
# 小规模验证（CPU 可跑，需 torch + transformers + pillow）
python scripts/build_hypergraph.py \
    --config configs/image_clip.toml \
    --data-root data/smoke_zh \
    --out out/smoke_zh_clip.json
```

集群实测（2026-11-13）：

```text
store: 18 entities, 3 images, 10 hyperedges
meta: image_embedder=clip, image_dim=512
z1_img1 | text hashing 256 | image clip 512 | path data/smoke_zh/images/ecg_demo.png
z2_img1 | text hashing 256 | image None      | path None   # 无文件，退回 caption
```

## 5. 与临床 schema 的衔接

`docs/CLINICAL_SCHEMA.md` 里 ECG/echo/CMR 条目带 `path`：

```json
{"modality": "ecg", "time": "...", "path": "ecg/P0001.xml"}
```

只要导出时把文件放在 `data-root` 下、`path` 写相对路径，`build_clinical_corpus.py` 生成的语料就会在构建时自动获得 CLIP 像素向量。

## 6. 跨模态检索（已实现）

`TwoTierMemory` 在 `cross_modal = true` 时，会把 CLIP 文本 query 检索到的图像证据与原有文本/超图证据**融合**（统一进入 L1/L2）：

```toml
[retrieval]
cross_modal = true
```

代码路径：

- `Store.search_images_clip(query_vec)`：在 `image_embedding` 空间检索；
- `TwoTierMemory.retrieve()`：`encode_texts(query)` → CLIP 空间 → 取回图像证据并 `admit` 到 L1；
- caption 空间检索保持不变，两个通道互补。

**合成图像验证**（6 张图，英文查询；同义/抽象描述）：

| 编码器 | CLIP hit@1 | caption 基线 hit@1 |
| --- | --- | --- |
| OpenAI CLIP (ViT-B/32) | **1.00** | 0.667 |
| **Chinese-CLIP** (ViT-B/16) | **1.00** | 0.167 |

> 图像为脚本生成的红方块/蓝圆/绿三角/心电图波形等，仅验证检索机制，不是临床数据。

## 7. Chinese-CLIP（中文医学影像首选）

`[image].embedder = "chinese-clip"` 启用 `ChineseCLIPEmbedder`（transformers `ChineseCLIPModel`），中文文本 query 与图像共享空间，比英文 CLIP 更适合中文临床图像。

```toml
[image]
embedder = "chinese-clip"
model = "OFA-Sys/chinese-clip-vit-base-patch16"
dim = 512
device = "cpu"
```

本地快照（避免重复下载）：

```bash
export MMGRAPH_CLIP_MODEL=/path/to/models/chinese-clip-vit-base-patch16
```

下载（集群实测 hf-mirror 约 4–16 MB/s，719MB 约 1 分钟）：

```bash
huggingface-cli download OFA-Sys/chinese-clip-vit-base-patch16 \
  --local-dir /path/to/models/chinese-clip-vit-base-patch16
```

中文查询验证（6 张合成图）：

```text
红色方块、蓝色圆形、绿色三角形、心电图波形、黄色星星、紫色十字
Chinese-CLIP hit@1 = 1.00，caption 基线 hit@1 = 0.167
```

## 8. 局限与下一步

1. 中文医学影像首选 **Chinese-CLIP**（已适配并验证）；BiomedCLIP 需要 `open_clip` 适配，暂未接入；
2. 跨模态检索已接入 Memory，但尚未做 **重排/融合权重学习**；当前是两路证据合并；
3. 超声/CMR 视频/三维序列需要抽帧/切片，当前只支持单张图片；
4. 影像文件必须留在院内网络，不得上传公网。
