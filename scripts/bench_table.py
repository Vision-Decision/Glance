"""Launch-style benchmark table (rows grouped by category, our model first and outlined in blue).
Cell colour: blue = our model is best in the row; grey = a baseline is best. A row is coloured only when at least
two models report it; ties at the displayed precision colour every tied cell. '—' = not reported/measured.
A row may set "span": [0, 1] to show one value across our first two columns (e.g. two checkpoints with the same
architecture share one speed measurement), with an optional "span_note" under the value.
Usage: python bench_table.py SPEC.json OUTNAME   (writes OUTNAME-{light,dark}.{svg,png})."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

T = {
    "light": dict(bg="#ffffff", ink="#1f1f1f", ink2="#5f6368", rule="#dadce0", strong="#3c4043",
                  ours_fill="#a8d1ff", ours_edge="#4c8df6", other_fill="#e3e5e8", head_ours="#ffffff"),
    "dark": dict(bg="#0d1117", ink="#e8eaed", ink2="#9aa0a6", rule="#30363d", strong="#9aa0a6",
                 ours_fill="#1c4a80", ours_edge="#5a9cf8", other_fill="#2d333b", head_ours="#0d1117"),
}

def fmt(v, f):
    return "—" if v is None else f.format(v)

def winners(vals, f, better):
    shown = [(i, fmt(v, f)) for i, v in enumerate(vals) if v is not None]
    if len(shown) < 2:
        return set()
    key = lambda s: float(s.replace(",", "").rstrip("%"))
    best = (max if better == "high" else min)(key(s) for _, s in shown)
    return {i for i, s in shown if key(s) == best}

def draw(spec, mode):
    t = T[mode]
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "svg.fonttype": "path"})
    cols = spec["columns"]; rows = spec["rows"]
    W_cat, W_bench, W_note, W_col = 1.55, 2.75, 1.55, 1.75
    W = W_cat + W_bench + W_note + W_col * len(cols)
    rh, hh, foot = 0.46, 0.62, 0.35 + 0.22 * len(spec["footer"])
    H = hh + rh * len(rows) + foot + 0.15
    fig = plt.figure(figsize=(W + 0.4, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor(t["bg"] if spec.get("opaque") else "none")
    x0 = W_cat + W_bench + W_note
    cx = [x0 + W_col * (j + 0.5) for j in range(len(cols))]
    # header
    ax.text(W_cat + 0.05, hh / 2, "Benchmark", va="center", fontsize=12, color=t["ink"])
    for j, c in enumerate(cols):
        ax.text(cx[j], hh / 2, c, va="center", ha="center", fontsize=11.5 if j else 12,
                weight="bold" if j == 0 else "normal", color=t["ink"])
    ax.plot([0, W], [hh, hh], color=t["strong"], lw=1.0)
    y = hh
    prev_cat = None
    for r, row in enumerate(rows):
        cat = row.get("cat")
        if cat and cat != prev_cat and r > 0:
            ax.plot([0, W], [y, y], color=t["strong"], lw=0.9)
        elif r > 0:
            ax.plot([W_cat, W], [y, y], color=t["rule"], lw=0.8)
        if cat and cat != prev_cat:
            ax.text(0.05, y + 0.12, cat, va="top", fontsize=10.5, color=t["ink2"], linespacing=1.2)
        prev_cat = cat or prev_cat
        f = row.get("fmt", "{:.1f}")
        span = row.get("span")
        vals = list(row["vals"])
        if span:  # the spanned columns count once, as the first of them
            for j in range(span[0] + 1, span[1] + 1):
                vals[j] = None
        win = winners(vals, f, row.get("better", "high"))
        if span:
            a_, b_ = span
            xa, xb = x0 + W_col * a_, x0 + W_col * (b_ + 1)
            if a_ in win:
                ax.add_patch(Rectangle((xa, y), xb - xa, rh, fc=t["ours_fill"] if a_ == 0 else t["other_fill"], ec="none", zorder=0))
            s = fmt(row["vals"][a_], f)
            dy = 0.07 if row.get("span_note") else 0
            ax.text((xa + xb) / 2, y + rh / 2 - dy, s, va="center", ha="center", fontsize=11.5,
                    weight="bold" if a_ in win else "normal", color=t["ink"])
            if row.get("span_note"):
                ax.text((xa + xb) / 2, y + rh / 2 + 0.13, row["span_note"], va="center", ha="center", fontsize=8,
                        color=t["ink2"])
        for j, v in enumerate(row["vals"]):
            if span and span[0] <= j <= span[1]:
                continue
            if j in win:
                ax.add_patch(Rectangle((x0 + W_col * j, y), W_col, rh, fc=t["ours_fill"] if j == 0 else t["other_fill"],
                                       ec="none", zorder=0))
            mark = (row.get("marks") or [""] * len(cols))[j] or ""
            s = fmt(v, f)
            ax.text(cx[j], y + rh / 2, s, va="center", ha="center", fontsize=11.5,
                    weight="bold" if j in win else "normal", color=t["ink"])
            if mark and v is not None:
                ax.text(cx[j] + 0.045 * len(s) + 0.07, y + rh / 2 - 0.09, mark, va="center", ha="left", fontsize=8,
                        color=t["ink2"])
        ax.text(W_cat + 0.05, y + rh / 2, row["bench"], va="center", fontsize=11, color=t["ink"])
        if row.get("note"):
            ax.text(x0 - 0.08, y + rh / 2, row["note"], va="center", ha="right", fontsize=8.8, color=t["ink2"],
                    linespacing=1.1)
        y += rh
    ax.plot([0, W], [y, y], color=t["strong"], lw=1.0)
    # outline of our column; on rows that span into the next column, the right edge goes around the merged cell
    yy, edge = hh, [(0.04, hh, x0 + W_col)]
    for row in rows:
        sp = row.get("span")
        edge.append((yy, yy + rh, x0 + W_col * (sp[1] + 1 if sp and sp[0] == 0 else 1)))
        yy += rh
    kw = dict(color=t["ours_edge"], lw=2.0, zorder=5, solid_capstyle="projecting")
    ax.plot([x0, x0], [0.04, y], **kw)
    ax.plot([x0, x0 + W_col], [0.04, 0.04], **kw)
    for k, (ya, yb, xr) in enumerate(edge):
        ax.plot([xr, xr], [ya, yb], **kw)
        if k + 1 < len(edge) and edge[k + 1][2] != xr:
            ax.plot([xr, edge[k + 1][2]], [yb, yb], **kw)
    ax.plot([x0, edge[-1][2]], [y, y], **kw)
    for k, line in enumerate(spec["footer"]):
        ax.text(0.0, y + 0.3 + 0.22 * k, line, va="center", fontsize=8.3, color=t["ink2"])
    return fig

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    for mode in ("light", "dark"):
        fig = draw(spec, mode)
        fig.savefig(f"{out}-{mode}.svg", transparent=not spec.get("opaque"), metadata={"Date": None})
        fig.savefig(f"{out}-{mode}.png", dpi=150, facecolor=T[mode]["bg"])
        plt.close(fig)
