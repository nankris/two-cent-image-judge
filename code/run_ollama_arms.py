"""Run Ollama arms on the 640 test images + controls. Resumable. usage: run_ollama_arms.py ARM [--subset20] [--controls] [--reversed]"""
import sys, csv, argparse
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "_shared")); sys.path.insert(0, str(ROOT / "code"))
import vdec
from labels import QUESTION, STATE, KEYS
ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--subset20", action="store_true"); ap.add_argument("--per-class", type=int, default=0)
ap.add_argument("--controls", action="store_true"); ap.add_argument("--reversed", action="store_true")
a = ap.parse_args()
arm = vdec.ARMS[a.arm]()
man = [r for r in csv.DictReader(open(ROOT / "data/manifest.csv")) if r["split"] == "test"]
def blanks():
    W, H = 384, 512
    out = {"blank_white": Image.new("RGB", (W, H), "white"), "blank_black": Image.new("RGB", (W, H), "black"),
           "blank_gray": Image.new("RGB", (W, H), (128, 128, 128))}
    for s in range(3):
        out[f"blank_noise{s}"] = Image.fromarray(np.random.default_rng(s).integers(0, 256, (H, W, 3), dtype=np.uint8))
    return out
if a.controls:
    items = [{"id": k, "label": None, "control": "blank", "image": vdec.load_png_bytes(im), "state": STATE, "question": QUESTION} for k, im in blanks().items()]
    items.append({"id": "label_only", "label": None, "control": "label_only", "image": None, "state": STATE, "question": QUESTION})
    vdec.run_items(arm, items, ROOT / f"results/raw/{a.arm}_controls.jsonl"); sys.exit()
if a.subset20 or a.per_class:
    N = a.per_class or 20
    by = {}
    for r in man: by.setdefault(r["label"], []).append(r)
    man = [r for k in KEYS for r in sorted(by[k], key=lambda r: r["id"])[:N]]
q = QUESTION
suffix = ""
if a.reversed:
    by = {}
    for r in man: by.setdefault(r["label"], []).append(r)
    man = [r for k in KEYS for r in sorted(by[k], key=lambda r: r["id"])[:4]]
    q = dict(QUESTION, criteria=dict(reversed(list(QUESTION["criteria"].items())))); suffix = "_reversed"
items = ({"id": r["id"], "label": r["label"], "image": vdec.load_png_bytes(ROOT / r["path"]), "state": STATE, "question": q} for r in man)
vdec.run_items(arm, items, ROOT / f"results/raw/{a.arm}{suffix}.jsonl")
