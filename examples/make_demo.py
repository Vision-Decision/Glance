"""Build the README demo (assets/demo.gif + assets/demo.mp4) from REAL Glance outputs.

    pip install -e ".[demo]" playwright && python -m playwright install chromium   # playwright only for the GUI mock
    python examples/make_demo.py                      # run the model, save assets/demo_predictions.json, render
    python examples/make_demo.py --from-json          # re-render from the saved predictions (no model needed)
    python examples/make_demo.py --model Vision-Decision/Glance-4B-preview --device mps

Every box, probability and bar in the animation is read from the model's output for that exact image / text and
question; nothing is edited by hand. The predictions are saved next to the GIF so they can be audited.
Requires ffmpeg on PATH for the GIF palette and the MP4.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets")
EX = os.path.join(A, "examples")

# Measured on an RTX PRO 6000 Blackwell (batch 1, COCO images at 640 visual tokens, bf16 + fla kernels):
# median 85.5 ms for 1 box question and 85.6 ms for 8 box questions per image (outputs/v1 eval, see README).
GPU_MS_8Q = 85.6

TICKET = ("Support ticket #4471\n\nI was charged twice for my March invoice and the second payment is still "
          "pending on my card. Please refund the duplicate charge before Friday, otherwise my account goes overdrawn.")

SCENES = [
    dict(name="natural", title="Referring expressions, incl. things that are not there",
         image="park_bench.jpg", max_vis_tokens=640, show_gpu_ms=True,
         questions=[dict(type="box", text="the woman sitting on the bench"),
                    dict(type="box", text="the man in the blue suit"),
                    dict(type="box", text="the clear umbrella"),
                    dict(type="box", text="a dog"),
                    dict(type="box", text="a red bicycle"),
                    dict(type="yesno", question="Is it raining?")]),
    dict(name="gui", title="GUI state check, next action, target, risk",
         image="editor.png", max_vis_tokens=640, show_gpu_ms=False,
         questions=[dict(type="yesno", question="Is a save dialog open?"),
                    dict(type="choice", question="Task: export the photo as a PNG file. What is the next action?",
                         options=["click", "type", "scroll", "done"], label="Next action (task: export as PNG)"),
                    dict(type="box", text="the Export button"),
                    dict(type="box", text="the Save button in the dialog"),
                    dict(type="score", question="How risky would it be to click 'Delete all' now?",
                         levels=["low", "medium", "high", "critical"], label="Risk of clicking 'Delete all'")]),
    dict(name="text", title="Text-only typed decisions (ticket routing)",
         text=TICKET, show_gpu_ms=False,
         questions=[dict(type="choice", question="Which team should handle this ticket?",
                         options={"billing": "payments, invoices, refunds", "technical": "bugs, errors, outages",
                                  "sales": "pricing, upgrades, new plans", "account": "login, profile, security"},
                         label="Route to team"),
                    dict(type="yesno", question="Does the customer ask for money back?"),
                    dict(type="score", question="How urgent is this ticket?", levels=["low", "medium", "high", "critical"],
                         label="Urgency"),
                    dict(type="choice", question="What is the customer's sentiment?",
                         options=["positive", "neutral", "frustrated", "angry"], label="Sentiment")]),
]


# ------------------------------------------------------------------------------------------ predictions
def ensure_gui_screenshot():
    p = os.path.join(EX, "editor.png")
    if os.path.exists(p):
        return
    import pathlib
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 800})
        pg.goto(pathlib.Path(os.path.join(ROOT, "examples", "gui_mock.html")).as_uri())
        pg.screenshot(path=p)
        b.close()


def predict(model_id, device):
    from glance import Glance, to_dict
    ensure_gui_screenshot()
    m = Glance.from_pretrained(model_id, device=device)
    out = dict(model=model_id, device=m.device, dtype=str(m.dtype), scenes=[])
    for sc in SCENES:
        qs = [{k: v for k, v in q.items() if k != "label"} for q in sc["questions"]]
        state = os.path.join(EX, sc["image"]) if sc.get("image") else sc["text"]
        kw = dict(max_vis_tokens=sc["max_vis_tokens"]) if sc.get("image") else {}
        m.ask(state, qs, **kw)  # warm-up
        t = time.perf_counter()
        res = m.ask(state, qs, **kw)
        dt = (time.perf_counter() - t) * 1000
        out["scenes"].append(dict(name=sc["name"], local_ms=round(dt, 1), results=[to_dict(r) for r in res]))
        print(sc["name"], f"{dt:.0f} ms on {m.device}", *res, sep="\n  ")
    return out


# ------------------------------------------------------------------------------------------ rendering
W, H = 960, 560
BG, INK, MUTED, RULE = (251, 251, 252), (17, 24, 39), (120, 126, 138), (227, 230, 235)
BLUE, BLUE_L, RED, GREEN = (37, 99, 235), (219, 231, 254), (220, 38, 38), (22, 163, 74)
PALETTE = [(37, 99, 235), (234, 88, 12), (22, 163, 74), (147, 51, 234), (219, 39, 119)]
FPS = 12


def font(size, bold=False):
    cands = [("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
             ("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf", 0),
             ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", 0),
             ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", 0)]
    for path, idx in cands:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        return ImageFont.truetype(os.path.join(d, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


F = dict(h=font(19, True), s=font(14), sb=font(14, True), xs=font(12), q=font(14), ans=font(15, True), big=font(15, True))


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def wrap(d, text, f, width):
    lines = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def q_label(q):
    if q.get("label"):
        return q["label"]
    return q["text"] if q["type"] == "box" else q["question"]


def answer_view(q, r):
    """-> (main answer text, probability for the bar, detail text, color)."""
    if q["type"] == "box":
        if r["present"]:
            return ("found", r["present_prob"], f"P(present) {r['present_prob']:.2f} · box conf. {r['confidence']:.2f}", None)
        return ("not present", r["present_prob"], f"P(present) {r['present_prob']:.3f}", RED)
    if q["type"] == "yesno":
        yes = r["p_yes"] >= 0.5
        return ("yes" if yes else "no", r["p_yes"] if yes else 1 - r["p_yes"], f"P(yes) {r['p_yes']:.3f}", None)
    probs = r["probs"]
    top = sorted(probs.items(), key=lambda kv: -kv[1])
    det = " · ".join(f"{k} {v:.2f}" for k, v in top[:4])
    return (r["answer"], top[0][1], det, None)


def draw_frame(sc, pred, t, img_cache):
    """t: seconds since scene start."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    qs, rs = sc["questions"], pred["results"]
    # header
    d.text((24, 18), "Glance-4B-preview", font=F["h"], fill=BLUE)
    d.text((24 + d.textlength("Glance-4B-preview", font=F["h"]) + 14, 22), sc["title"], font=F["s"], fill=MUTED)
    d.text((W - 24, 22), "real model outputs", font=F["xs"], fill=MUTED, anchor="ra")
    d.line([(24, 52), (W - 24, 52)], fill=RULE, width=1)
    # timeline: 0-1.0 questions appear; 1.0-1.5 forward pass; 1.5-2.2 answers grow; hold
    t_pass, t_ans = 1.0, 1.5
    reveal = ease((t - t_ans) / 0.7)
    # left: state
    LX, LY, LW, LH = 24, 68, 548, 420
    box_qs = [(i, q, r) for i, (q, r) in enumerate(zip(qs, rs)) if q["type"] == "box"]
    colors = {}
    for k, (i, q, r) in enumerate(box_qs):
        colors[i] = PALETTE[k % len(PALETTE)]
    if sc.get("image"):
        src = img_cache[sc["image"]]
        s = min(LW / src.width, LH / src.height)
        sw, sh = int(src.width * s), int(src.height * s)
        ox, oy = LX + (LW - sw) // 2, LY + (LH - sh) // 2
        im.paste(src.resize((sw, sh), Image.LANCZOS), (ox, oy))
        if reveal > 0:
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            for i, q, r in box_qs:
                if not r["present"]:
                    continue
                c = colors[i]
                x1, y1, x2, y2 = [v * s for v in r["box"]]
                x1, y1, x2, y2 = ox + x1, oy + y1, ox + x2, oy + y2
                a = int(255 * reveal)
                od.rectangle([x1, y1, x2, y2], outline=c + (a,), width=3)
                lab = f"{q['text']}  {r['confidence']:.2f}"
                tw = od.textlength(lab, font=F["xs"]) + 10
                ty = y1 - 20 if y1 - 20 > oy else y1 + 2
                od.rectangle([x1, ty, x1 + tw, ty + 18], fill=c + (int(225 * reveal),))
                od.text((x1 + 5, ty + 2), lab, font=F["xs"], fill=(255, 255, 255, a))
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(im)
    else:  # text state card
        d.rounded_rectangle([LX, LY, LX + LW, LY + LH], radius=12, fill=(255, 255, 255), outline=RULE)
        y = LY + 22
        for ln in wrap(d, sc["text"], font(20), LW - 44):
            d.text((LX + 22, y), ln, font=font(20), fill=INK)
            y += 31
    # forward-pass pulse on the state
    if t_pass <= t < t_ans + 0.3:
        p = ease((t - t_pass) / 0.5)
        yline = LY + int((LH) * p)
        d.line([(LX, yline), (LX + LW, yline)], fill=BLUE, width=3)
    # right: questions + answers
    RX, RW = 596, W - 596 - 24
    n = len(qs)
    row_h = min(70, (LH + 10) // n)
    for i, (q, r) in enumerate(zip(qs, rs)):
        y = LY + i * row_h
        appear = ease((t - 0.12 * i) / 0.35)
        if appear <= 0:
            continue
        tag = {"box": "box", "yesno": "yes/no", "choice": "choice", "score": "score"}[q["type"]]
        c = colors.get(i, (90, 96, 110))
        d.rounded_rectangle([RX, y + 2, RX + 52, y + 20], radius=5, fill=c if q["type"] == "box" else (229, 231, 236))
        d.text((RX + 26, y + 11), tag, font=F["xs"], fill=(255, 255, 255) if q["type"] == "box" else INK, anchor="mm")
        lab = q_label(q)
        while d.textlength(lab, font=F["q"]) > RW - 62 and len(lab) > 8:
            lab = lab[:-2]
        if lab != q_label(q):
            lab = lab.rstrip() + "…"
        d.text((RX + 60, y + 3), lab, font=F["q"], fill=INK)
        if reveal > 0:
            ans, p, det, col = answer_view(q, r)
            col = col or (BLUE if q["type"] != "box" else c)
            d.text((RX, y + 27), ans, font=F["ans"], fill=col)
            bx = RX + 112
            bw = RW - 112 - 46
            d.rounded_rectangle([bx, y + 30, bx + bw, y + 40], radius=5, fill=(234, 236, 240))
            fw = max(0, int(bw * p * reveal))
            if fw > 3:
                d.rounded_rectangle([bx, y + 30, bx + fw, y + 40], radius=5, fill=col)
            d.text((RX + RW, y + 26), f"{p * reveal:.2f}", font=F["sb"], fill=INK, anchor="ra")
            dt = det
            while d.textlength(dt, font=F["xs"]) > RW and len(dt) > 8:
                dt = dt[:-2]
            d.text((RX, y + 47), dt, font=F["xs"], fill=MUTED)
    # footer
    d.line([(24, H - 52), (W - 24, H - 52)], fill=RULE, width=1)
    msg = f"1 forward pass · {n} questions"
    if sc.get("show_gpu_ms"):
        msg += f" · ~{GPU_MS_8Q:.0f} ms on RTX PRO 6000"
    d.text((24, H - 44), msg, font=F["big"], fill=INK if t >= t_pass else MUTED)
    note = "no token-by-token decoding; probabilities are calibrated"
    if sc.get("show_gpu_ms"):
        note = "GPU time: measured median for 8 questions per COCO image, batch 1, 640 visual tokens; " + note
    d.text((24, H - 22), note, font=F["xs"], fill=MUTED)
    return im


def render(preds, out_gif, out_mp4, hold=4.6):
    img_cache = {sc["image"]: Image.open(os.path.join(EX, sc["image"])).convert("RGB") for sc in SCENES if sc.get("image")}
    tmp = tempfile.mkdtemp()
    k = 0
    for sc, pr in zip(SCENES, preds["scenes"]):
        assert sc["name"] == pr["name"]
        dur = 2.2 + hold
        for f in range(int(dur * FPS)):
            draw_frame(sc, pr, f / FPS, img_cache).save(os.path.join(tmp, f"f{k:05d}.png"))
            k += 1
    pat = os.path.join(tmp, "f%05d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat, "-vf",
                    "palettegen=max_colors=128:stats_mode=diff", os.path.join(tmp, "pal.png")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat, "-i", os.path.join(tmp, "pal.png"),
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", out_gif], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat, "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", "-movflags", "+faststart", out_mp4], check=True)
    shutil.rmtree(tmp)
    print(f"{k} frames ({k / FPS:.1f} s) -> {out_gif} ({os.path.getsize(out_gif) / 1e6:.1f} MB), {out_mp4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview")
    ap.add_argument("--device", default=None)
    ap.add_argument("--from-json", action="store_true", help="re-render from assets/demo_predictions.json")
    a = ap.parse_args()
    pj = os.path.join(A, "demo_predictions.json")
    if a.from_json:
        preds = json.load(open(pj))
    else:
        preds = predict(a.model, a.device)
        preds["scenes_spec"] = SCENES
        json.dump(preds, open(pj, "w"), indent=1)
    render(preds, os.path.join(A, "demo.gif"), os.path.join(A, "demo.mp4"))


if __name__ == "__main__":
    sys.exit(main())
