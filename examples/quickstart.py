"""Run every API call shown in the README and print the results.

    python examples/quickstart.py [--model Vision-Decision/Glance-4B-preview] [--device cuda|mps|cpu] [--draw]

--draw also writes assets/example_boxes.png (predicted boxes; absent targets listed with their P(present)).
"""
import argparse
import os
import time

from glance import Glance, Box, Choice, YesNo, Score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "assets", "examples")


def timed(label, fn):
    t = time.perf_counter()
    r = fn()
    shown = f"{len(r)} results" if isinstance(r, list) else r
    print(f"[{label}] {(time.perf_counter() - t) * 1000:.0f} ms\n  {shown}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview")
    ap.add_argument("--device", default=None)
    ap.add_argument("--draw", action="store_true")
    a = ap.parse_args()

    t = time.perf_counter()
    m = Glance.from_pretrained(a.model, device=a.device)
    print(f"loaded on {m.device} ({m.dtype}) in {time.perf_counter() - t:.1f} s")
    park, veg, ui = (os.path.join(EX, f) for f in ("park_bench.jpg", "vegetables.jpg", "editor.png"))
    ticket = ("I was charged twice for my March invoice and the second payment is still pending. "
              "Please refund the duplicate charge before Friday.")

    m.box(park, "the clear umbrella")  # warm-up
    timed("box", lambda: m.box(park, "the clear umbrella"))
    timed("box, absent", lambda: m.box(park, "a red bicycle"))
    timed("choice, image", lambda: m.choice(park, "Where are they?", ["park", "beach", "office", "train station"]))
    timed("yesno, image", lambda: m.yesno(park, "Is it raining?"))
    timed("yesno, screenshot", lambda: m.yesno(ui, "Is a save dialog open?"))
    timed("choice, text", lambda: m.choice(ticket, "Which team should handle this ticket?",
                                           {"billing": "payments, invoices, refunds", "technical": "bugs, errors, outages",
                                            "sales": "pricing, upgrades, new plans"}))
    timed("score, text", lambda: m.score(ticket, "How urgent is this ticket?", levels=["low", "medium", "high", "critical"]))
    rs = timed("ask, 8 questions in one pass", lambda: m.ask(veg, [
        "the broccoli", "the tomatoes", "the carrots", "a banana", "a knife",
        YesNo("Is this food cooked?"),
        Choice("What is the main food group?", ["vegetables", "meat", "dairy", "grains"]),
        Score("How healthy is this food?", ["unhealthy", "neutral", "healthy", "very healthy"])]))
    for r in rs:
        print("   ", r)

    if a.draw:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(veg).convert("RGB")
        d = ImageDraw.Draw(im)
        try:
            f = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 15)
        except OSError:
            f = ImageFont.load_default()
        cols = [(37, 99, 235), (234, 88, 12), (22, 163, 74)]
        y_abs = im.height - 8
        for r in rs:
            if getattr(r, "kind", "") != "box":
                continue
            if r.present:
                c = cols.pop(0)
                d.rectangle(r.box, outline=c, width=3)
                lab = f"{r.text}  conf {r.confidence:.2f}"
                tw = d.textlength(lab, font=f) + 8
                d.rectangle([r.box[0], r.box[1], r.box[0] + tw, r.box[1] + 20], fill=c)
                d.text((r.box[0] + 4, r.box[1] + 2), lab, font=f, fill="white")
            else:
                lab = f"\"{r.text}\": not present, P(present) = {r.present_prob:.3f}"
                tw = d.textlength(lab, font=f) + 10
                d.rectangle([6, y_abs - 22, 6 + tw, y_abs], fill=(220, 38, 38))
                d.text((11, y_abs - 20), lab, font=f, fill="white")
                y_abs -= 26
        out = os.path.join(ROOT, "assets", "example_boxes.png")
        im.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
