"""Render the Q10 financial MAS construction stages as a PNG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.q9 import render_evolution_graph as base  # noqa: E402


def render_q10(record: dict, output: Path) -> None:
    panels = base._build_panels(record)
    image = Image.new("RGB", (base.CANVAS_W, base.CANVAS_H), "#e2e8f0")
    draw = ImageDraw.Draw(image)
    fonts = base._fonts()
    for index, panel in enumerate(panels):
        x = 24 + index * (base.PANEL_W + 24)
        base._draw_panel(draw, x, 48, panel, fonts)
        if index < len(panels) - 1:
            base._draw_transition(draw, x + base.PANEL_W + 4, 245, fonts["large"])
    dataset = record.get("dataset", "dataset")
    draw.text((24, 15), f"Q10 Financial MAS Graph Evolution | {dataset}", fill="#0f172a", font=fonts["title"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
    print(f"wrote {output} {image.size[0]}x{image.size[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Q10 Input/Requirement/Blueprint/Executable MAS stages.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_q10(json.loads(args.input.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
