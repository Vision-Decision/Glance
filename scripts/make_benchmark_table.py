"""Render assets/benchmarks.svg and assets/benchmarks.png (PNG via headless Chromium / Playwright).

    pip install playwright && python -m playwright install chromium
    python scripts/make_benchmark_table.py

All numbers and their sources are listed in README.md#benchmarks. Footnote letters below match the README.
"""
import html
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

MODELS = [  # (name, subtitle)
    ("Glance-4B-preview", "ours · one prefill"),
    ("Qwen3.5-4B", "base · generative"),
    ("Qwen3-VL-8B", "generative"),
    ("UI-Venus-1.5-8B", "GUI specialist"),
    ("KV-Ground-4B", "GUI specialist"),
    ("RC-GRPO-II-4B", "gRefCOCO specialist"),
]
NA = None
# (category, benchmark, metric note, higher_is_better, [(display, value, footnote) or None per model])
ROWS = [
    ("Referring grounding", "RefCOCO / + / g, average", "Acc@0.5", True,
     [("90.8", 90.8, "a"), ("88.1", 88.1, "b"), ("89.1", 89.1, "c"), NA, NA, NA]),
    (None, "RefCOCO val", "Acc@0.5", True, [("93.2", 93.2, ""), ("91.4", 91.4, "d"), NA, NA, NA, NA]),
    (None, "RefCOCO+ testB", "Acc@0.5", True, [("84.6", 84.6, ""), NA, NA, NA, NA, NA]),
    (None, "RefCOCOg test", "Acc@0.5", True, [("90.4", 90.4, ""), NA, NA, NA, NA, NA]),
    ("Absence / refusal", "gRefCOCO val, no-target", "N-acc", True,
     [("87.4", 87.4, ""), ("65.0", 65.0, "d"), NA, NA, NA, ("85.3", 85.3, "e")]),
    (None, "VenusBench-GD refusal", "refusal acc.", True,
     [("81.8", 81.8, "f"), NA, ("63.8", 63.8, "g"), ("61.6", 61.6, "h"), NA, NA]),
    ("GUI grounding", "ScreenSpot-v2", "click acc.", True,
     [("89.4", 89.4, ""), ("93.4", 93.4, "d"), ("92.1", 92.1, "h"), ("95.9", 95.9, "h"), ("94.1", 94.1, "i"), NA]),
    (None, "ScreenSpot-Pro", "click acc.", True,
     [("29.3", 29.3, ""), ("60.3", 60.3, "b"), ("54.6", 54.6, "c"), ("68.4", 68.4, "h"), ("67.0", 67.0, "i"), NA]),
    (None, "VenusBench-GD overall", "acc.", True,
     [("40.1", 40.1, "f"), NA, ("62.3", 62.3, "g"), ("72.3", 72.3, "h"), NA, NA]),
    ("Typed decisions", "OpenDecisions", "capability", True,
     [("67.6", 67.6, "j"), ("63.6", 63.6, "j"), NA, NA, NA, NA]),
    ("Calibration", "Box-confidence ECE", "gRefCOCO val, ↓", False, [("3.8%", 3.8, "k"), NA, NA, NA, NA, NA]),
    (None, "P(present) ECE", "gRefCOCO val, ↓", False, [("2.4%", 2.4, "k"), NA, NA, NA, NA, NA]),
    ("Efficiency", "Latency, 1 question", "ms, ↓", False, [("85", 85.5, "l"), ("400", 399.6, "d"), NA, NA, NA, NA]),
    (None, "Latency, 8 questions / image", "ms, ↓", False,
     [("86", 85.6, "l"), ("3,161", 3161.2, "d"), NA, NA, NA, NA]),
]
FOOTER = "Methodology: github.com/Vision-Decision/Glance#benchmarks"
NOTE = ("— = not reported; a row is highlighted only where at least two models report it. Letters mark footnotes (sources, protocol differences) in the README. "
        "Latency: batch 1, RTX PRO 6000 Blackwell.")

# ---------------------------------------------------------------- layout
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
PAD = 36
W_CAT, W_BENCH = 178, 330
W_COL = 158
ROW_H, HEAD_H, CAT_GAP = 40, 70, 10
BLUE, BLUE_FILL, GREY_FILL = "#2563eb", "#dbe7fe", "#eceef2"
INK, MUTED, RULE, BG = "#111827", "#8a909c", "#e3e6eb", "#fbfbfc"


def esc(s):
    return html.escape(s, quote=True)


def best_index(row):
    vals = [(i, c[1]) for i, c in enumerate(row[4]) if c]
    if len(vals) < 2:  # nothing to compare against: no highlight
        return None
    f = max if row[3] else min
    return f(vals, key=lambda t: t[1])[0]


def build_svg():
    n = len(MODELS)
    width = PAD * 2 + W_CAT + W_BENCH + n * W_COL
    y = PAD + HEAD_H
    ys = []
    for k, r in enumerate(ROWS):
        if r[0] and k:
            y += CAT_GAP
        ys.append(y)
        y += ROW_H
    table_bottom = y
    height = table_bottom + 74
    x0 = PAD + W_CAT + W_BENCH
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="{esc(FONT)}">',
         f'<rect width="{width}" height="{height}" fill="{BG}"/>']
    # best-cell fills
    for r, yy in zip(ROWS, ys):
        b = best_index(r)
        if b is not None:
            fill = BLUE_FILL if b == 0 else GREY_FILL
            o.append(f'<rect x="{x0 + b * W_COL + 6}" y="{yy + 4}" width="{W_COL - 12}" height="{ROW_H - 8}" rx="6" fill="{fill}"/>')
    # header
    for i, (name, sub) in enumerate(MODELS):
        cx = x0 + i * W_COL + W_COL / 2
        wt = 700 if i == 0 else 600
        col = BLUE if i == 0 else INK
        o.append(f'<text x="{cx}" y="{PAD + 30}" text-anchor="middle" font-size="15" font-weight="{wt}" fill="{col}">{esc(name)}</text>')
        o.append(f'<text x="{cx}" y="{PAD + 50}" text-anchor="middle" font-size="12" fill="{MUTED}">{esc(sub)}</text>')
    o.append(f'<line x1="{PAD}" x2="{width - PAD}" y1="{PAD + HEAD_H - 0.5}" y2="{PAD + HEAD_H - 0.5}" stroke="{INK}" stroke-width="1.2"/>')
    # rows
    for k, (r, yy) in enumerate(zip(ROWS, ys)):
        cat, bench, metric, _, cells = r
        mid = yy + ROW_H / 2 + 5
        if cat and k:
            o.append(f'<line x1="{PAD}" x2="{width - PAD}" y1="{yy - CAT_GAP / 2}" y2="{yy - CAT_GAP / 2}" stroke="{RULE}" stroke-width="1"/>')
        if cat:
            o.append(f'<text x="{PAD}" y="{mid}" font-size="14" font-weight="650" fill="{INK}">{esc(cat)}</text>')
        bx = PAD + W_CAT
        o.append(f'<text x="{bx}" y="{mid}" font-size="14" fill="{INK}">{esc(bench)}'
                 f'<tspan dx="8" font-size="12" fill="{MUTED}">{esc(metric)}</tspan></text>')
        b = best_index(r)
        for i, c in enumerate(cells):
            cx = x0 + i * W_COL + W_COL / 2
            if not c:
                o.append(f'<text x="{cx}" y="{mid}" text-anchor="middle" font-size="14" fill="#b8bcc6">—</text>')
                continue
            disp, _, fn = c
            wt = 700 if i == b else 400
            col = BLUE if (i == b and i == 0) else INK
            sup = f'<tspan dx="1" dy="-6" font-size="10" font-weight="400" fill="{MUTED}">{esc(fn)}</tspan>' if fn else ""
            o.append(f'<text x="{cx}" y="{mid}" text-anchor="middle" font-size="15" font-weight="{wt}" fill="{col}" '
                     f'style="font-variant-numeric: tabular-nums">{esc(disp)}{sup}</text>')
        if k + 1 < len(ROWS) and not ROWS[k + 1][0]:
            o.append(f'<line x1="{PAD + W_CAT}" x2="{width - PAD}" y1="{yy + ROW_H}" y2="{yy + ROW_H}" stroke="{RULE}" stroke-width="0.6" stroke-dasharray="0"/>')
    o.append(f'<line x1="{PAD}" x2="{width - PAD}" y1="{table_bottom + 0.5}" y2="{table_bottom + 0.5}" stroke="{INK}" stroke-width="1.2"/>')
    # blue outline around our column
    o.append(f'<rect x="{x0 + 2}" y="{PAD + 6}" width="{W_COL - 4}" height="{table_bottom - PAD - 2}" rx="10" fill="none" '
             f'stroke="{BLUE}" stroke-width="2"/>')
    o.append(f'<text x="{PAD}" y="{table_bottom + 30}" font-size="12" fill="{MUTED}">{esc(NOTE)}</text>')
    o.append(f'<text x="{PAD}" y="{table_bottom + 52}" font-size="13" fill="{INK}">Methodology: '
             f'<tspan fill="{BLUE}">{esc(FOOTER.split(": ", 1)[1])}</tspan></text>')
    o.append("</svg>")
    return "\n".join(o), width, height


def main():
    os.makedirs(OUT, exist_ok=True)
    svg, w, h = build_svg()
    svg_path = os.path.join(OUT, "benchmarks.svg")
    open(svg_path, "w").write(svg)
    from playwright.sync_api import sync_playwright
    page_html = (f"<html><head><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' "
                 f"rel='stylesheet'><style>body{{margin:0}}</style></head><body>{svg}</body></html>")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        try:  # web font if reachable, else the system sans-serif fallback
            pg.set_content(page_html, wait_until="networkidle", timeout=15000)
        except Exception:
            pg.set_content(page_html.replace("<link", "<meta"), wait_until="load")
        pg.locator("svg").screenshot(path=os.path.join(OUT, "benchmarks.png"))
        b.close()
    print("wrote", svg_path, os.path.join(OUT, "benchmarks.png"))


if __name__ == "__main__":
    main()
