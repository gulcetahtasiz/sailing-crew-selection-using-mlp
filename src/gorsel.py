# overlay_lineup.py
# Usage:
#   python overlay_lineup.py
# Inputs:
#   - A background boat image (e.g., "boat.png")
#   - A CSV "team_selection.csv" with columns: role, athlete_id (and optional: score, athlete_name)
# Output:
#   - "lineup_overlay.png" with names overlaid on the image

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# ---------- Role positions (normalized 0..1) ----------
ROLE_COORDS = {
    "basustu":   (0.82, 0.50),
    "direkdibi": (0.60, 0.50),
    "piyano":    (0.40, 0.50),
    "trim1":     (0.20, 0.20),
    "trim2":     (0.20, 0.80),
    "anayelken": (0.20, 0.55),
    "dumen":     (0.10, 0.45),
}

# ---------- Font config (macOS paths shown; edit if needed) ----------
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"  # set to a TTF that supports Turkish glyphs

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a TTF font; fall back gracefully."""
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size=size)
        except Exception:
            pass
    # common macOS fallback
    fallback = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    if os.path.exists(fallback):
        try:
            return ImageFont.truetype(fallback, size=size)
        except Exception:
            pass
    # common Linux fallback
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(dejavu):
        try:
            return ImageFont.truetype(dejavu, size=size)
        except Exception:
            pass
    # last resort
    return ImageFont.load_default()

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """Return (tw, th) for possibly multiline text, robust across Pillow versions."""
    try:
        # Preferred: direct multiline bbox without anchor
        l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        return (r - l), (b - t)
    except Exception:
        # Fallback: measure line by line
        lines = str(text).split("\n")
        widths = [draw.textlength(line, font=font) for line in lines] or [0]
        tw = max(widths)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        th = line_h * max(1, len(lines))
        return tw, th

def draw_label(draw: ImageDraw.ImageDraw, xy, text, font, pad=8,
               show_box=False,                 # default: no background box
               box_fill=(0, 0, 0, 160),
               text_fill=(255, 255, 255, 255),
               stroke_width=1,
               stroke_color=(0, 0, 0, 180)):
    """
    Draw centered (optionally multiline) text at (x,y).
    Optionally draw a semi-transparent rounded box behind it.
    """
    x, y = xy
    tw, th = _measure_text(draw, text, font)
    bw, bh = tw + 2 * pad, th + 2 * pad
    left, top = int(x - bw / 2), int(y - bh / 2)
    right, bottom = int(x + bw / 2), int(y + bh / 2)

    if show_box:
        try:
            draw.rounded_rectangle([left, top, right, bottom], radius=10, fill=box_fill)
        except Exception:
            draw.rectangle([left, top, right, bottom], fill=box_fill)

    tx, ty = int(x - tw / 2), int(y - th / 2)
    try:
        draw.multiline_text((tx, ty), text, font=font, fill=text_fill,
                            align="center", stroke_width=stroke_width, stroke_fill=stroke_color)
    except Exception:
        draw.multiline_text((tx, ty), text, font=font, fill=text_fill, align="center")

def annotate_lineup(image_path: str,
                    team_csv: str,
                    output_path: str,
                    role_coords: dict = ROLE_COORDS,
                    show_score: bool = True,
                    font_scale: float = 0.55,   # smaller text (relative to image width)
                    show_box: bool = False,     # turn background box off
                    stroke_width: int = 1):
    """
    Overlay crew names on the boat image according to role positions.
    """
    im = Image.open(image_path).convert("RGBA")
    W, H = im.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    team = pd.read_csv(team_csv)

    base_font_size = max(12, int(W * 0.028 * font_scale))
    font = load_font(base_font_size)

    for _, row in team.iterrows():
        role = str(row["role"])
        # prefer 'athlete_id'; if absent, fall back to 'athlete_name'
        display = str(row["athlete_id"]) if "athlete_id" in row and pd.notna(row["athlete_id"]) \
                  else str(row.get("athlete_name", ""))

        if show_score and "score" in row and pd.notna(row["score"]):
            try:
                display = f"{display}\n({float(row['score']):.2f})"
            except Exception:
                display = f"{display}\n({row['score']})"

        if role not in role_coords:
            # role missing: show a small warning bottom-left (optional)
            warn_font = load_font(max(10, int(W * 0.018)))
            draw_label(draw, (int(W * 0.16), int(H * 0.92)),
                       f"Missing coords for '{role}'", warn_font,
                       show_box=True, box_fill=(220, 80, 80, 180), stroke_width=0)
            continue

        x_norm, y_norm = role_coords[role]
        x_px, y_px = int(x_norm * W), int(y_norm * H)

        draw_label(draw, (x_px, y_px), display, font,
                   pad=4, show_box=show_box, stroke_width=stroke_width)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    out.save(output_path, format="PNG")
    print(f"Saved annotated lineup to: {output_path}")

# --------- Demo call ---------
if __name__ == "__main__":
    IMAGE_PATH = "boat.png"               # your boat diagram or photo
    TEAM_CSV   = "team_selection.csv"     # produced by the Hungarian step
    OUTPUT     = "lineup_overlay.png"

    annotate_lineup(
        IMAGE_PATH,
        TEAM_CSV,
        OUTPUT,
        ROLE_COORDS,
        show_score=False,      # set False if you don't want scores
        font_scale=0.55,      # smaller text; try 0.45 or 0.65 to adjust
        show_box=False,       # no gray background box
        stroke_width=1        # thin outline; set 0 to remove outline
    )
