# Stitch all per-run AlphaZero learning-curve PNGs into one image for the
# project report.  Each PNG is cropped to its top panel (win rates) so the
# result reads as a single timeline rather than 4 stacked full reports.
# Run from the project root:
#   python scripts/stitch_plots.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image, ImageDraw, ImageFont

from src.paths import PLOTS_DIR

# Chronological order — edit labels/filenames if you add more runs.
PHASES = [
    ("Phase 1 – Initial training",  "alphazero_learning_curve.png"),
    ("Phase 2 – Continuation 1",    "alphazero_learning_curve_continue_447932.png"),
    ("Phase 3 – Continuation 2",    "alphazero_learning_curve_continue_505814.png"),
    ("Phase 4 – Continuation 3",    "alphazero_learning_curve_continue_545140.png"),
]

OUTPUT_PATH = PLOTS_DIR / "alphazero_full_history.png"

# Matplotlib saves the figure with tight_layout.  The three subplots each
# occupy roughly one-third of the figure height.  We keep only the top third
# (win-rate panel) plus a small bottom margin so the x-axis label shows.
# Adjust PANEL_FRACTION if your plots look different.
PANEL_FRACTION = 0.38   # fraction of image height to keep per phase

LABEL_HEIGHT = 30
LABEL_BG     = (220, 230, 220)   # soft green tint to distinguish from plot bg
LABEL_FG     = (30, 30, 30)
FONT_SIZE    = 18


def load_font(size):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def make_label_strip(width, text, font):
    img  = Image.new("RGB", (width, LABEL_HEIGHT), color=LABEL_BG)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (LABEL_HEIGHT - th) // 2), text,
              fill=LABEL_FG, font=font)
    return img


def crop_top_panel(img, fraction):
    """Keep only the top `fraction` of the image (the win-rate subplot)."""
    h = int(img.height * fraction)
    return img.crop((0, 0, img.width, h))


def main():
    font = load_font(FONT_SIZE)

    panels = []
    for label, filename in PHASES:
        path = PLOTS_DIR / filename
        if not path.exists():
            print(f"  WARNING: {filename} not found — skipping")
            continue
        img = Image.open(path).convert("RGB")
        panel = crop_top_panel(img, PANEL_FRACTION)
        panels.append((label, panel))

    if not panels:
        print("No plot files found. Nothing to stitch.")
        return

    width = max(p.width for _, p in panels)

    strips = []
    for label, panel in panels:
        strips.append(make_label_strip(width, label, font))
        if panel.width != width:
            scale = width / panel.width
            panel = panel.resize((width, int(panel.height * scale)), Image.LANCZOS)
        strips.append(panel)

    total_height = sum(s.height for s in strips)
    canvas = Image.new("RGB", (width, total_height), color=(255, 255, 255))
    y = 0
    for strip in strips:
        canvas.paste(strip, (0, y))
        y += strip.height

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(OUTPUT_PATH), dpi=(100, 100))
    print(f"Saved → {OUTPUT_PATH}  ({len(panels)} phases, top-panel only)")
    print()
    print("Tip: once you've done another training run, use plot_history.py")
    print("     for a proper single continuous chart from the accumulated JSON.")


if __name__ == "__main__":
    main()
