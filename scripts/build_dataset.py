"""Build composite images for the prompt-injection benchmark.

Reads scripts/dataset_config.csv, applies one of several overlay styles to
each base image in data/raw/, and writes the result to data/final/.

Run from the repo root:
    python scripts/build_dataset.py

Requires: Pillow  (pip install Pillow)
"""

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
FINAL_DIR = REPO_ROOT / "data" / "final"
CONFIG = REPO_ROOT / "scripts" / "dataset_config.csv"

# Windows font paths. Adjust if running on a different OS.
FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_SERIF = "C:/Windows/Fonts/times.ttf"


def load_font(path, size):
    return ImageFont.truetype(path, size)


def wrap_text(text, font, max_width, draw):
    """Greedy word-wrap so text fits within max_width pixels."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def style_soft_strip(img, text):
    """Blunt overlay: dark stripe across upper third with white bold text."""
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(20, int(W * 0.038))
    font = load_font(FONT_BOLD, font_size)

    margin = int(W * 0.05)
    max_text_width = W - 2 * margin
    lines = wrap_text(text, font, max_text_width, draw)
    line_height = font_size + 8
    block_h = line_height * len(lines)

    strip_top = int(H * 0.22)
    strip_bottom = strip_top + block_h + 40
    draw.rectangle([(0, strip_top), (W, strip_bottom)], fill=(0, 0, 0, 190))

    y = strip_top + 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (W - text_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height

    return img


def style_stego_corner(img, text):
    """Hard: tiny low-contrast text in lower-right corner."""
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(11, int(W * 0.013))
    font = load_font(FONT_REGULAR, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = W - text_w - 24
    y = H - text_h - 24

    # Light gray, semi-transparent. Should be legible to OCR but easy
    # for a human to miss on a busy background.
    draw.text((x, y), text, font=font, fill=(140, 140, 140, 170))

    return img


def style_ui_bubble(img, text):
    """Hard: fake system-message bubble at top of chat UI."""
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(16, int(W * 0.022))
    font = load_font(FONT_BOLD, font_size)

    margin = int(W * 0.06)
    pad_x = 28
    pad_y = 18
    max_text_width = W - 2 * margin - 2 * pad_x
    lines = wrap_text(text, font, max_text_width, draw)
    line_height = font_size + 8
    block_h = line_height * len(lines)

    bubble_top = int(H * 0.04)
    bubble_left = margin
    bubble_right = W - margin
    bubble_bottom = bubble_top + block_h + 2 * pad_y

    draw.rounded_rectangle(
        [(bubble_left, bubble_top), (bubble_right, bubble_bottom)],
        radius=14,
        fill=(255, 244, 220, 245),
        outline=(220, 130, 50, 255),
        width=3,
    )

    y = bubble_top + pad_y
    for line in lines:
        draw.text(
            (bubble_left + pad_x, y),
            line,
            font=font,
            fill=(80, 30, 0, 255),
        )
        y += line_height

    return img


def style_memo_letterhead(img, text):
    """Hard: fake memo printed on a clipboard's existing paper.

    The injection_text encodes the header/body split with '||' as separator.
    Position calibrated for example 13's clipboard: the paper region lives
    in roughly the middle of the frame horizontally (37 to 64 percent) and
    starts at ~36 percent from the top. We render text directly on the
    existing white paper without drawing our own rectangle, so the memo
    looks printed on the clipboard rather than floating above it.
    """
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    if "||" in text:
        header, body = text.split("||", 1)
    else:
        header, body = "MEMO", text

    header_size = max(16, int(W * 0.018))
    body_size = max(12, int(W * 0.013))
    header_font = load_font(FONT_BOLD, header_size)
    body_font = load_font(FONT_REGULAR, body_size)

    paper_left = int(W * 0.37)
    paper_right = int(W * 0.64)
    paper_top = int(H * 0.36)

    pad = 16
    max_text_width = paper_right - paper_left - 2 * pad
    header_lines = wrap_text(header, header_font, max_text_width, draw)
    body_lines = wrap_text(body, body_font, max_text_width, draw)

    y = paper_top + 12
    for hline in header_lines:
        bbox = draw.textbbox((0, 0), hline, font=header_font)
        line_w = bbox[2] - bbox[0]
        line_x = (paper_left + paper_right - line_w) // 2
        draw.text((line_x, y), hline, font=header_font, fill=(0, 0, 0, 255))
        y += header_size + 4

    underline_y = y + 4
    draw.line(
        [(paper_left + 16, underline_y), (paper_right - 16, underline_y)],
        fill=(0, 0, 0, 255),
        width=1,
    )

    y = underline_y + 12
    for line in body_lines:
        draw.text((paper_left + pad, y), line, font=body_font, fill=(0, 0, 0, 255))
        y += body_size + 6

    return img


def style_stickynote(img, text):
    """Hard: text rendered as if handwritten on the sticky note.

    Calibrated for example 14: targets the lower-center yellow sticky
    note (the more centrally framed of the two notes in the scene)
    and constrains text to fit within the note's bounds.
    """
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(10, int(W * 0.010))
    font = load_font(FONT_REGULAR, font_size)

    note_left = int(W * 0.50)
    note_top = int(H * 0.62)
    max_text_width = int(W * 0.13)

    lines = wrap_text(text, font, max_text_width, draw)
    line_height = font_size + 4

    y = note_top
    for line in lines:
        draw.text((note_left, y), line, font=font, fill=(15, 15, 15, 235))
        y += line_height

    return img


def style_document_paragraph(img, text):
    """Hard: serif body text overlaid as if part of the page."""
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font_size = max(14, int(W * 0.018))
    font = load_font(FONT_SERIF, font_size)

    margin = int(W * 0.14)
    max_text_width = W - 2 * margin
    lines = wrap_text(text, font, max_text_width, draw)
    line_height = font_size + 4
    block_h = line_height * len(lines)

    block_top = int(H * 0.55)
    block_bottom = block_top + block_h + 24

    draw.rectangle(
        [(margin - 12, block_top - 14), (W - margin + 12, block_bottom)],
        fill=(252, 248, 238, 220),
    )

    y = block_top
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(20, 20, 20, 255))
        y += line_height

    return img


STYLE_REGISTRY = {
    "soft_strip": style_soft_strip,
    "stego_corner": style_stego_corner,
    "ui_bubble": style_ui_bubble,
    "memo_letterhead": style_memo_letterhead,
    "stickynote": style_stickynote,
    "document_paragraph": style_document_paragraph,
}


def process_row(row):
    raw_path = RAW_DIR / row["raw_filename"]
    final_path = FINAL_DIR / row["final_filename"]
    category = row["category"]

    if not raw_path.exists():
        print(f"[skip] {row['id']}: missing {raw_path}")
        return

    if category == "clean":
        shutil.copy(raw_path, final_path)
        print(f"[clean]      {row['id']} -> {final_path.name}")
        return

    style_id = row["style_id"]
    style_fn = STYLE_REGISTRY.get(style_id)
    if style_fn is None:
        raise ValueError(f"Unknown style_id '{style_id}' for row {row['id']}")

    img = Image.open(raw_path).convert("RGBA")
    img = style_fn(img, row["injection_text"])
    img.convert("RGB").save(final_path, quality=92)
    print(f"[{style_id:<19}] {row['id']} -> {final_path.name}")


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        process_row(row)

    print(f"\nDone. Wrote {len(rows)} files to {FINAL_DIR}")


if __name__ == "__main__":
    main()
