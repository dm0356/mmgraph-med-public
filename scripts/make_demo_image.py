"""Generate a synthetic demo ECG image for the smoke corpus.

This is **not patient data**; it only exists so the optional Qwen-VL captioner
can be exercised end-to-end without committing a real medical image.
"""

from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 640, 320
OUT = Path(__file__).resolve().parents[1] / "data" / "smoke_zh" / "images" / "ecg_demo.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    for x in range(0, WIDTH, 40):
        draw.line([(x, 0), (x, HEIGHT)], fill=(235, 235, 235))
    for y in range(0, HEIGHT, 40):
        draw.line([(0, y), (WIDTH, y)], fill=(235, 235, 235))

    points = []
    import math

    for x in range(WIDTH):
        y = HEIGHT / 2
        y -= 60 * math.exp(-((x - 120) ** 2) / 200)
        y += 70 * math.exp(-((x - 150) ** 2) / 120)
        y -= 40 * math.exp(-((x - 170) ** 2) / 200)
        y += 50 * math.exp(-((x - 420) ** 2) / 200)
        y += 90 * math.exp(-((x - 450) ** 2) / 120)
        y -= 30 * math.exp(-((x - 470) ** 2) / 200)
        points.append((x, y))

    draw.line(points, fill=(180, 30, 30), width=2)
    draw.text((10, 10), "Synthetic ECG demo - not patient data", fill=(90, 90, 90))
    image.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
