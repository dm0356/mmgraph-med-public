"""Generate a tiny synthetic image-retrieval demo (no patient data).

Creates data/image_demo/{images,corpus.jsonl,queries.jsonl}.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "image_demo"
IMAGES = OUT / "images"


def _base():
    image = Image.new("RGB", (224, 224), "white")
    return image, ImageDraw.Draw(image)


def make_red_square(path):
    image, draw = _base()
    draw.rectangle((52, 52, 172, 172), fill=(220, 30, 30))
    image.save(path)


def make_blue_circle(path):
    image, draw = _base()
    draw.ellipse((52, 52, 172, 172), fill=(30, 60, 220))
    image.save(path)


def make_green_triangle(path):
    image, draw = _base()
    draw.polygon([(112, 40), (52, 180), (172, 180)], fill=(30, 160, 60))
    image.save(path)


def make_ecg_trace(path):
    import math

    image, draw = _base()
    points = []
    for x in range(20, 204):
        t = (x - 20) / 184.0
        phase = (t * 4) % 1.0
        if phase < 0.10:
            y = 120 - 8 * math.sin(phase / 0.10 * math.pi)
        elif phase < 0.18:
            y = 120 + 70 * math.sin((phase - 0.10) / 0.08 * math.pi)
        elif phase < 0.26:
            y = 120 - 40 * math.sin((phase - 0.18) / 0.08 * math.pi)
        else:
            y = 120
        points.append((x, y))
    draw.line(points, fill=(10, 10, 10), width=3)
    image.save(path)


def make_yellow_star(path):
    import math

    image, draw = _base()
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = 78 if index % 2 == 0 else 32
        points.append((112 + radius * math.cos(angle), 112 + radius * math.sin(angle)))
    draw.polygon(points, fill=(230, 200, 20))
    image.save(path)


def make_purple_plus(path):
    image, draw = _base()
    draw.rectangle((96, 44, 128, 180), fill=(150, 40, 180))
    draw.rectangle((44, 96, 180, 128), fill=(150, 40, 180))
    image.save(path)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    specs = [
        ("img1", "A red square on a white background.", make_red_square, "a red box"),
        ("img2", "A blue circle on a white background.", make_blue_circle, "a round blue object"),
        ("img3", "A green triangle on a white background.", make_green_triangle, "a green three-sided shape"),
        ("img4", "An ECG waveform trace on a white background.", make_ecg_trace, "a heart electrical signal"),
        ("img5", "A yellow star on a white background.", make_yellow_star, "a star-shaped yellow item"),
        ("img6", "A purple plus sign on a white background.", make_purple_plus, "a medical cross symbol"),
    ]
    (OUT / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": image_id,
                    "text": caption,
                    "images": [{"image_id": image_id, "caption": caption,
                                "path": f"images/{image_id}.png"}],
                    "tables": [],
                    "diagrams": [],
                },
                ensure_ascii=False,
            )
            for image_id, caption, maker, _query in specs
            for _ in [maker(IMAGES / f"{image_id}.png")]
        ),
        encoding="utf-8",
    )
    (OUT / "queries.jsonl").write_text(
        "\n".join(
            json.dumps({"query": query, "gold_image_id": image_id}, ensure_ascii=False)
            for image_id, _caption, _maker, query in specs
        ),
        encoding="utf-8",
    )
    (OUT / "qa.jsonl").write_text("", encoding="utf-8")
    chinese = {
        "img1": "红色方块",
        "img2": "蓝色圆形",
        "img3": "绿色三角形",
        "img4": "心电图波形",
        "img5": "黄色星星",
        "img6": "紫色十字",
    }
    (OUT / "queries_zh.jsonl").write_text(
        "\n".join(
            json.dumps({"query": chinese[image_id], "gold_image_id": image_id}, ensure_ascii=False)
            for image_id, _caption, _maker, _query in specs
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} images -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
