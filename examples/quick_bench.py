"""Quick benchmark: a few minutes on one GPU, small fixed subsets of public benchmarks.

    pip install "glance @ git+https://github.com/Vision-Decision/Glance" pandas
    python examples/quick_bench.py --model Vision-Decision/Glance-4B-v1.3

Subsets (item ids in examples/quick_bench_items.json, drawn once with seed 0 by --make-items):

  refcoco_val      200 RefCOCO (UNC) val expressions        Acc@0.5 of the box (standard REC, presence gate ignored)
  grefcoco_nt      200 gRefCOCO val no-target expressions   N-acc: P(present) < 0.5
  grefcoco_multi   100 gRefCOCO val multi-target expressions set F1 at IoU 0.5 (Boxes question; models without
                                                             multi-box heads answer one box)
  screenspot_v2    100 ScreenSpot-v2 instructions            click accuracy (box centre inside the target),
                                                             <= 2,048 visual tokens
  latency          30 RefCOCO val images                     median ms of one ask() with 1 and with 8 box questions

These are smoke numbers on small subsets: they check that a checkpoint loads and behaves, and they are noisy
(one standard error is about 2 points at 90% accuracy on 200 items, 3 on 100). Full-benchmark results are on each model card.

Data is downloaded from public sources: annotations from the Hugging Face datasets jxu124/refcoco, FudanCVL/gRefCOCO
and OS-Copilot/ScreenSpot-v2 (only the 100 screenshots are read out of the zip), COCO train2014 images from
images.cocodataset.org. Nothing is redistributed by this repository except the item ids.

On Google Colab, uninstall the preinstalled torchao first (`pip uninstall -y torchao`): peft refuses torchao older
than 0.16 when it injects the LoRA.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "quick_bench_items.json")
COCO_URL = "http://images.cocodataset.org/train2014/{}"
GREF_FILE = "grefs(unc).json"


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(".").strip()


# ------------------------------------------------------------------------------------------- data
def refcoco_val(cache):
    import pandas as pd
    from huggingface_hub import list_repo_files, hf_hub_download
    f = [x for x in list_repo_files("jxu124/refcoco", repo_type="dataset") if x.startswith("data/validation-")][0]
    df = pd.read_parquet(hf_hub_download("jxu124/refcoco", f, repo_type="dataset", cache_dir=cache))
    out = {}
    for r in df.itertuples():
        info = json.loads(r.raw_image_info)
        for s in r.sentences:
            out[int(s["sent_id"])] = dict(file=info["file_name"], text=norm_text(s["sent"]),
                                          box=[float(x) for x in r.bbox])
    return out


def grefcoco_val(cache, need_boxes):
    from huggingface_hub import hf_hub_download
    grefs = json.load(open(hf_hub_download("FudanCVL/gRefCOCO", GREF_FILE, repo_type="dataset", cache_dir=cache)))
    ann_box, files = {}, {}
    if need_boxes:
        inst = json.load(open(hf_hub_download("FudanCVL/gRefCOCO", "instances.json", repo_type="dataset",
                                              cache_dir=cache)))
        for a in inst["annotations"]:
            x, y, w, h = a["bbox"]
            ann_box[a["id"]] = [x, y, x + w, y + h]
        files = {im["id"]: im["file_name"] for im in inst["images"]}
    nt, multi = {}, {}
    for r in grefs:
        if r["split"] != "val":
            continue
        ids = [x for x in r["ann_id"] if x != -1] if isinstance(r["ann_id"], list) else [r["ann_id"]]
        file = files.get(r["image_id"]) or f"COCO_train2014_{r['image_id']:012d}.jpg"
        for s in r["sentences"]:
            sid = int(s["sent_id"])
            if r["no_target"]:
                nt[sid] = dict(file=file, text=norm_text(s["sent"]))
            elif len(ids) > 1:
                multi[sid] = dict(file=file, text=norm_text(s["sent"]), boxes=[ann_box.get(i) for i in ids])
    return nt, multi


def screenspot_v2(cache, keys=None):
    """-> {"<platform>_<index>": dict(img=PIL image or None, text, box)}; images only for ``keys``."""
    from huggingface_hub import hf_hub_download, HfFileSystem
    from PIL import Image
    items = {}
    for plat in ["mobile", "desktop", "web"]:
        js = json.load(open(hf_hub_download("OS-Copilot/ScreenSpot-v2", f"screenspot_{plat}_v2.json",
                                            repo_type="dataset", cache_dir=cache)))
        for i, a in enumerate(js):
            x, y, w, h = a["bbox"]
            items[f"{plat}_{i}"] = dict(file=a["img_filename"], text=a["instruction"], box=[x, y, x + w, y + h])
    if keys is None:
        return items
    d = os.path.join(cache, "ss2_img")
    os.makedirs(d, exist_ok=True)
    need = sorted({items[k]["file"] for k in keys if not os.path.exists(os.path.join(d, items[k]["file"]))})
    if need:  # random access into the 1.3 GB zip: only the needed members are read
        fs = HfFileSystem()
        with fs.open("datasets/OS-Copilot/ScreenSpot-v2/screenspotv2_image.zip", "rb", block_size=2 ** 20) as fh:
            z = zipfile.ZipFile(fh)
            names = {os.path.basename(n): n for n in z.namelist()}
            for f in need:
                with open(os.path.join(d, f), "wb") as o:
                    o.write(z.read(names[f]))
    for k in keys:
        items[k]["img"] = Image.open(os.path.join(d, items[k]["file"])).convert("RGB")
    return items


def coco_images(files, cache):
    from PIL import Image
    d = os.path.join(cache, "coco_train2014")
    os.makedirs(d, exist_ok=True)

    def get(f):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            for attempt in range(4):
                try:
                    data = urllib.request.urlopen(COCO_URL.format(f), timeout=30).read()
                    open(p + ".part", "wb").write(data)
                    os.replace(p + ".part", p)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
        return f, Image.open(p).convert("RGB")

    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(get, sorted(set(files))))


# ------------------------------------------------------------------------------------------- metrics
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def set_f1(P, G, thr=0.5):
    """F1 = 2TP / (|P| + |G|), one-to-one greedy matching on IoU (as in gRefCOCO GREC scoring)."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(P) for j, g in enumerate(G)), reverse=True)
    up, ug, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in up and j not in ug:
            up.add(i); ug.add(j); tp += 1
    return 1.0 if not P and not G else 2 * tp / (len(P) + len(G))


# ------------------------------------------------------------------------------------------- items
def make_items(cache, seed=0):
    rng = random.Random(seed)
    rc = refcoco_val(cache)
    nt, multi = grefcoco_val(cache, need_boxes=False)
    ss = screenspot_v2(cache)
    items = dict(
        seed=seed,
        note="Drawn with random.Random(seed).sample over the sorted ids of each source; see examples/quick_bench.py.",
        refcoco_val=sorted(rng.sample(sorted(rc), 200)),
        grefcoco_nt=sorted(rng.sample(sorted(nt), 200)),
        grefcoco_multi=sorted(rng.sample(sorted(multi), 100)),
        screenspot_v2=sorted(rng.sample(sorted(ss), 100)),
    )
    json.dump(items, open(ITEMS, "w"), indent=1)
    print("wrote", ITEMS, {k: len(v) for k, v in items.items() if isinstance(v, list)})


# ------------------------------------------------------------------------------------------- run
def run(a):
    import torch
    from glance import Glance, Box, Boxes

    items = json.load(open(ITEMS))
    t0 = time.time()
    rc = refcoco_val(a.cache)
    nt, multi = grefcoco_val(a.cache, need_boxes=True)
    ss = screenspot_v2(a.cache, items["screenspot_v2"])
    rc_items = [rc[i] for i in items["refcoco_val"]]
    nt_items = [nt[i] for i in items["grefcoco_nt"]]
    mt_items = [multi[i] for i in items["grefcoco_multi"]]
    imgs = coco_images([x["file"] for x in rc_items + nt_items + mt_items], a.cache)
    t_data = time.time() - t0

    t0 = time.time()
    m = Glance.from_pretrained(a.model, device=a.device, revision=a.revision)
    t_load = time.time() - t0
    m.box(next(iter(imgs.values())), "the person on the left")  # warm-up

    def per_image(its, make_q):
        """Group items by image and ask all questions about one image in one pass. -> results in input order."""
        by = defaultdict(list)
        for k, it in enumerate(its):
            by[it["file"]].append(k)
        out = [None] * len(its)
        for f, ks in by.items():
            rs = m.ask(imgs[f], [make_q(its[k]) for k in ks])
            for k, r in zip(ks, rs):
                out[k] = r
        return out

    t0 = time.time()
    res = {}
    r = per_image(rc_items, lambda it: Box(it["text"]))
    res["refcoco_val"] = dict(n=len(r), acc50=100 * sum(iou(x.box, it["box"]) >= 0.5 for x, it in zip(r, rc_items)) / len(r),
                              gated_acc50=100 * sum(x.present and iou(x.box, it["box"]) >= 0.5
                                                    for x, it in zip(r, rc_items)) / len(r))
    r = per_image(nt_items, lambda it: Box(it["text"]))
    res["grefcoco_nt"] = dict(n=len(r), n_acc=100 * sum(not x.present for x in r) / len(r))
    r = per_image(mt_items, lambda it: Boxes(it["text"]))
    res["grefcoco_multi"] = dict(n=len(r), set_f1=100 * statistics.mean(set_f1(list(x.boxes), it["boxes"])
                                                                        for x, it in zip(r, mt_items)),
                                 mean_pred_boxes=statistics.mean(len(x.boxes) for x in r),
                                 multibox_heads=m.multibox is not None)
    hit = 0
    for k in items["screenspot_v2"]:
        it = ss[k]
        x = m.ask(it["img"], [Box(it["text"])], max_vis_tokens=2048)[0]
        cx, cy = (x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2
        g = it["box"]
        hit += g[0] <= cx <= g[2] and g[1] <= cy <= g[3]
    res["screenspot_v2"] = dict(n=len(items["screenspot_v2"]), click_acc=100 * hit / len(items["screenspot_v2"]))
    t_eval = time.time() - t0

    # latency: 30 RefCOCO images, 1 and 8 box questions about the same image; image already decoded in memory
    sync = torch.cuda.synchronize if m.device.startswith("cuda") else (lambda: None)
    files = list(dict.fromkeys(it["file"] for it in rc_items))[:30]
    texts = [it["text"] for it in rc_items]
    lat = {}
    for nq in (1, 8):
        ts = []
        for i, f in enumerate(files):
            qs = [texts[(i * 8 + j) % len(texts)] for j in range(nq)]
            if i < 3:
                m.ask(imgs[f], qs)
            sync(); t = time.perf_counter()
            m.ask(imgs[f], qs)
            sync(); ts.append(1000 * (time.perf_counter() - t))
        lat[f"{nq}q_ms"] = statistics.median(ts)
    res["latency"] = dict(n_images=len(files), **lat)

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        fla = True
    except Exception:
        fla = False
    import transformers
    env = dict(gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else m.device, dtype=str(m.dtype),
               torch=torch.__version__, transformers=transformers.__version__, fla=fla)
    out = dict(model=a.model, version=m.cfg.get("version"), env=env, results=res,
               seconds=dict(data=round(t_data, 1), load=round(t_load, 1), eval=round(t_eval, 1)))
    rr = res
    print(f"\n| model | RefCOCO val Acc@0.5 (200) | gRefCOCO no-target N-acc (200) | gRefCOCO multi-target "
          f"set F1 (100) | ScreenSpot-v2 click (100) | latency 1 / 8 q (ms) |")
    print("|---|---|---|---|---|---|")
    print(f"| {a.model.split('/')[-1]} | {rr['refcoco_val']['acc50']:.1f} | {rr['grefcoco_nt']['n_acc']:.1f} | "
          f"{rr['grefcoco_multi']['set_f1']:.1f}{'' if m.multibox is not None else ' (one box)'} | "
          f"{rr['screenspot_v2']['click_acc']:.1f} | {lat['1q_ms']:.0f} / {lat['8q_ms']:.0f} |")
    print(f"\n{env['gpu']}, {env['dtype']}, fla={fla}; data {t_data:.0f} s, load {t_load:.0f} s, eval {t_eval:.0f} s")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/glance_quick_bench"))
    ap.add_argument("--out", default=None, help="write the results as JSON")
    ap.add_argument("--make-items", action="store_true", help="redraw the item ids (seed 0); not needed to run")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    if a.make_items:
        make_items(a.cache)
        sys.exit(0)
    run(a)


if __name__ == "__main__":
    main()
