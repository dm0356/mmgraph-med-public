"""Generate the four-paper fusion architecture diagram (PNG + SVG).

Outputs:
    docs/assets/architecture.png
    docs/assets/architecture.svg

Requires Pillow (already available in the project launchers).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 1080
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "input": ("#E8F1FB", "#4A90D9"),
    "m3kg": ("#FFF4E5", "#E8A33D"),
    "mmgraph": ("#E8F8EE", "#3AA76D"),
    "aikg": ("#F3E8FF", "#8E44AD"),
    "policy": ("#FDE8E8", "#C0392B"),
    "answer": ("#EAF2F8", "#2C3E50"),
    "gray": "#666666",
    "text": "#1F2937",
}


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except Exception:
        return ImageFont.load_default()


TITLE = "MMGraph-Med: a four-paper fusion pipeline"
SUBTITLE = (
    "Construction (M3KG-RAG)  ->  Hypergraph + Memory (MMGraph-Agent)  ->  "
    "multi-hop retrieval  ->  policy (heuristic / GRPO reward)"
)
FOOTER = "Patient data stays on the local/cluster deployment; the official ICD-10 table is not redistributed."

BOXES = [
    # row 1
    dict(x=40, y=150, w=320, h=90, kind="input", title="Clinical documents",
         lines=["text · tables · images"]),
    dict(x=420, y=150, w=300, h=90, kind="input", title="Parser",
         lines=["modality-tagged units"]),
    # M3KG band
    dict(x=60, y=350, w=270, h=100, kind="m3kg", title="1. Rewriter",
         lines=["context-enriched text"]),
    dict(x=360, y=350, w=270, h=100, kind="m3kg", title="2. Context-aware extraction",
         lines=["entities + relations"]),
    dict(x=660, y=350, w=270, h=100, kind="m3kg", title="3. Normalizer",
         lines=["canonical names"]),
    dict(x=960, y=350, w=270, h=100, kind="m3kg", title="4. ICD-10 grounding",
         lines=["codes + descriptions"]),
    dict(x=1260, y=350, w=280, h=100, kind="m3kg", title="5. Description refinement",
         lines=["context-aware rewrite"]),
    # row 3
    dict(x=40, y=530, w=300, h=100, kind="mmgraph", title="Embeddings",
         lines=["BGE-M3 · CLIP (text)"]),
    dict(x=380, y=530, w=340, h=100, kind="mmgraph", title="Hypergraph builder",
         lines=["n-ary hyperedges", "+ image links"]),
    dict(x=760, y=530, w=300, h=100, kind="mmgraph", title="Store",
         lines=["entities · images", "hyperedges"]),
    dict(x=1110, y=530, w=340, h=100, kind="aikg", title="ai-knowledge-graph viewer",
         lines=["self-contained HTML"]),
    # row 4
    dict(x=40, y=720, w=300, h=100, kind="mmgraph", title="L1/L2 cache memory",
         lines=["truncate · digest · cap"]),
    dict(x=380, y=720, w=300, h=100, kind="mmgraph", title="Multi-hop retrieval",
         lines=["shared entities"]),
    dict(x=720, y=720, w=300, h=100, kind="policy", title="Policy",
         lines=["heuristic / GRPO reward"]),
    dict(x=1060, y=720, w=300, h=100, kind="answer", title="Answer",
         lines=["grounded response"]),
]

ARROWS = [
    # input -> parser
    dict(points=[(360, 195), (420, 195)]),
    # m3kg chain
    dict(points=[(330, 400), (360, 400)]),
    dict(points=[(630, 400), (660, 400)]),
    dict(points=[(930, 400), (960, 400)]),
    dict(points=[(1230, 400), (1260, 400)]),
    # band -> builder
    dict(points=[(800, 450), (800, 500), (550, 500), (550, 530)]),
    # embeddings -> builder -> store
    dict(points=[(340, 580), (380, 580)]),
    dict(points=[(720, 580), (760, 580)]),
    # store -> viewer (dashed)
    dict(points=[(1060, 580), (1110, 580)], dashed=True),
    # store -> memory
    dict(points=[(910, 630), (910, 675), (190, 675), (190, 720)]),
    # memory -> multi-hop -> policy -> answer
    dict(points=[(340, 770), (380, 770)]),
    dict(points=[(680, 770), (720, 770)]),
    dict(points=[(1020, 770), (1060, 770)]),
]

LABELS = [
    dict(x=60, y=310, text="M3KG-RAG construction (context-rich)", size=22, bold=True,
         color=COLORS["text"]),
    dict(x=40, y=985, text=FOOTER, size=17, bold=False, color="#6B7280"),
]


def _draw_box(draw, box):
    fill, outline = COLORS[box["kind"]]
    draw.rounded_rectangle(
        (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]),
        radius=14, fill=fill, outline=outline, width=3,
    )
    title_font = _font(21, bold=True)
    line_font = _font(18)
    title = box["title"]
    lines = box["lines"]
    total = len(lines)
    # title block + lines block are vertically centred together
    title_h = 28
    line_h = 26
    block_h = title_h + total * line_h
    top = box["y"] + (box["h"] - block_h) / 2

    def draw_center(text, y, font, color):
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((box["x"] + (box["w"] - (bbox[2] - bbox[0])) / 2, y), text,
                  font=font, fill=color)

    draw_center(title, top, title_font, COLORS["text"])
    for index, line in enumerate(lines):
        draw_center(line, top + title_h + index * line_h, line_font, "#374151")


def _arrow_png(draw, points, dashed=False):
    for start, end in zip(points, points[1:]):
        if dashed:
            # simple dashed line
            x1, y1 = start
            x2, y2 = end
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length == 0:
                continue
            step = 14
            for offset in range(0, int(length), step * 2):
                t1 = offset / length
                t2 = min((offset + step) / length, 1.0)
                draw.line(
                    (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                     x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2),
                    fill=COLORS["gray"], width=3,
                )
        else:
            draw.line([start, end], fill=COLORS["gray"], width=3, joint="curve")
    # arrowhead at the last point
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = __import__("math").atan2(y2 - y1, x2 - x1)
    size = 16
    left = (x2 - size * __import__("math").cos(angle - 0.45),
            y2 - size * __import__("math").sin(angle - 0.45))
    right = (x2 - size * __import__("math").cos(angle + 0.45),
             y2 - size * __import__("math").sin(angle + 0.45))
    draw.polygon([points[-1], left, right], fill=COLORS["gray"])


def render_png():
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), TITLE, font=_font(38, bold=True), fill=COLORS["text"])
    draw.text((40, 88), SUBTITLE, font=_font(19), fill="#4B5563")
    for box in BOXES:
        _draw_box(draw, box)
    for arrow in ARROWS:
        _arrow_png(draw, arrow["points"], arrow.get("dashed", False))
    for label in LABELS:
        draw.text((label["x"], label["y"]), label["text"],
                  font=_font(label["size"], bold=label["bold"]), fill=label["color"])
    image.save(OUT / "architecture.png")
    print("wrote", OUT / "architecture.png")


def _svg_text(x, y, text, size, bold=False, color="#1F2937"):
    weight = "bold" if bold else "normal"
    return (f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{text}</text>')


def render_svg():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
        'orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#666666"/></marker></defs>',
        _svg_text(40, 58, TITLE, 38, True),
        _svg_text(40, 108, SUBTITLE, 19, False, "#4B5563"),
    ]
    for box in BOXES:
        fill, outline = COLORS[box["kind"]]
        parts.append(
            f'<rect x="{box["x"]}" y="{box["y"]}" width="{box["w"]}" height="{box["h"]}" '
            f'rx="14" fill="{fill}" stroke="{outline}" stroke-width="3"/>'
        )
        draw = ImageDraw.Draw(Image.new("RGB", (W, H)))
        title_font = _font(21, bold=True)
        line_font = _font(18)
        total = len(box["lines"])
        block_h = 28 + total * 26
        top = box["y"] + (box["h"] - block_h) / 2
        bbox = draw.textbbox((0, 0), box["title"], font=title_font)
        tx = box["x"] + (box["w"] - (bbox[2] - bbox[0])) / 2
        parts.append(_svg_text(tx, top + 22, box["title"], 21, True))
        for index, line in enumerate(box["lines"]):
            bbox = draw.textbbox((0, 0), line, font=line_font)
            lx = box["x"] + (box["w"] - (bbox[2] - bbox[0])) / 2
            parts.append(_svg_text(lx, top + 28 + 22 + index * 26, line, 18, False, "#374151"))
    for arrow in ARROWS:
        points = arrow["points"]
        dash = ' stroke-dasharray="10,7"' if arrow.get("dashed") else ""
        path = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(points))
        parts.append(
            f'<path d="{path}" fill="none" stroke="#666666" stroke-width="3"{dash} '
            'marker-end="url(#arrow)"/>'
        )
    for label in LABELS:
        parts.append(_svg_text(label["x"], label["y"] + label["size"], label["text"],
                               label["size"], label["bold"], label["color"]))
    parts.append("</svg>")
    (OUT / "architecture.svg").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT / "architecture.svg")


if __name__ == "__main__":
    render_png()
    render_svg()
