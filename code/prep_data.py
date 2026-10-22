"""Sample 40/class from the RVL-CDIP *test* split (chainyo/rvl-cdip parquet mirror), seed 20261021.
Train/cal pool for probe + fine-tune: jordyvl/rvl_cdip_100_examples_per_class train (800) + validation (400).
Writes data/test/*.png, data/train/*.png, data/cal/*.png and data/manifest.csv with sha256. Checks overlap."""
import glob, io, hashlib, csv, sys, os
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); from labels import KEYS
ROOT = Path(__file__).resolve().parents[1]; CACHE = Path(os.getenv("RVL_CACHE_DIR", ROOT.parent / "_data_cache/rvl"))
OFFICIAL_KEYS = ["letter", "form", "email", "handwritten", "advertisement", "report", "publication", "specification",
                 "file_folder", "news_article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
SEED, PER = 20261021, 40
rng = np.random.default_rng(SEED)
rows = []
for f in sorted(glob.glob(str(CACHE / "test-*.parquet"))):
    t = pq.read_table(f, columns=["label"]); labs = t["label"].to_pylist()
    rows += [(f, i, l) for i, l in enumerate(labs)]
print("test rows", len(rows))
sel = []
for c in range(16):
    cand = [r for r in rows if r[2] == c]
    pick = rng.choice(len(cand), PER, replace=False)
    sel += [cand[i] for i in sorted(pick)]
man = []
def save(img_bytes, split, idx, lab, src, label_order=KEYS):
    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    p = ROOT / "data" / split / f"{split}_{idx:04d}_{KEYS[lab]}.png"; p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, optimize=True)
    raw = hashlib.sha256(im.tobytes()).hexdigest()
    man.append({"id": p.stem, "split": split, "label": label_order[lab], "label_idx": lab, "path": str(p.relative_to(ROOT)),
                "w": im.size[0], "h": im.size[1], "pixel_sha256": raw, "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "source": src})
by_file = {}
for f, i, l in sel: by_file.setdefault(f, []).append((i, l))
k = 0
for f, lst in by_file.items():
    t = pq.read_table(f).take([i for i, _ in lst]).to_pylist()
    for (i, l), r in zip(lst, t):
        assert r["label"] == l
        save(r["image"]["bytes"], "test", k, l, f"chainyo/rvl-cdip:{Path(f).name}#{i}"); k += 1
for split, pat in [("train", "j-train-*"), ("cal", "j-validation-*")]:
    f = glob.glob(str(CACHE / pat))[0]
    for j, r in enumerate(pq.read_table(f).to_pylist()):
        save(r["image"]["bytes"], split, j, r["label"], f"jordyvl/rvl_cdip_100_examples_per_class:{Path(f).name}#{j}", OFFICIAL_KEYS)
test_h = {m["pixel_sha256"] for m in man if m["split"] == "test"}
ov = [m["id"] for m in man if m["split"] != "test" and m["pixel_sha256"] in test_h]
print("overlap train/cal with test:", len(ov), ov[:5])
with open(ROOT / "data/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
print({s: sum(m["split"] == s for m in man) for s in ["test", "train", "cal"]})
