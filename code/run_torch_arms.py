"""SigLIP2 zero-shot, SigLIP2 linear probe, DiT fine-tuned (supervised ceiling). CPU, 2 threads.
Writes results/raw/{siglip-zs,siglip-probe,dit-ft}.jsonl + controls, embeddings cache in results/cache/."""
import sys, csv, json, time, os
from pathlib import Path
import numpy as np, torch
from PIL import Image
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
from labels import LABELS, KEYS
OFFICIAL_KEYS = ["letter", "form", "email", "handwritten", "advertisement", "report", "publication", "specification",
                 "file_folder", "news_article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
from transformers import AutoModel, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification
man = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
CACHE = ROOT / "results/cache"; CACHE.mkdir(parents=True, exist_ok=True); RAW = ROOT / "results/raw"; RAW.mkdir(parents=True, exist_ok=True)
def blanks():
    W, H = 384, 512
    out = {"blank_white": Image.new("RGB", (W, H), "white"), "blank_black": Image.new("RGB", (W, H), "black"),
           "blank_gray": Image.new("RGB", (W, H), (128, 128, 128))}
    for s in range(3):
        out[f"blank_noise{s}"] = Image.fromarray(np.random.default_rng(s).integers(0, 256, (H, W, 3), dtype=np.uint8))
    return out
def img(r): return Image.open(ROOT / r["path"]).convert("RGB")
def write(name, recs):
    with open(RAW / f"{name}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
# ---------------- SigLIP2
MID = "google/siglip2-base-patch16-384"
model = AutoModel.from_pretrained(MID).eval(); proc = AutoProcessor.from_pretrained(MID)
with torch.no_grad():
    texts = [f"a scanned document: {d.lower()}" for _, d in LABELS]
    ti = proc(text=texts, return_tensors="pt", padding="max_length", max_length=64)
    T = model.get_text_features(**ti); T = T if torch.is_tensor(T) else T.pooler_output
    T = torch.nn.functional.normalize(T, dim=-1)
def embed(images):
    with torch.no_grad():
        x = proc(images=images, return_tensors="pt")
        t0 = time.perf_counter(); E = model.get_image_features(**x); dt = (time.perf_counter() - t0) * 1000 / len(images)
        E = E if torch.is_tensor(E) else E.pooler_output
        return torch.nn.functional.normalize(E, dim=-1), dt
emb = {}; lat = {}
for split in ["test", "train", "cal"]:
    f = CACHE / f"siglip2_{split}.npz"
    rows = [r for r in man if r["split"] == split]
    if f.exists():
        z = np.load(f); emb[split] = z["E"]; lat[split] = z["lat"]; continue
    Es, ls = [], []
    for i in range(0, len(rows), 8):
        E, dt = embed([img(r) for r in rows[i:i+8]]); Es.append(E.numpy()); ls += [dt] * len(rows[i:i+8])
    emb[split] = np.concatenate(Es); lat[split] = np.array(ls); np.savez(f, E=emb[split], lat=lat[split])
    print(split, emb[split].shape, flush=True)
scale = model.logit_scale.exp().item(); bias = model.logit_bias.item()
test = [r for r in man if r["split"] == "test"]
def zs_probs(E):
    logits = torch.tensor(E) @ T.T * scale + bias
    return torch.softmax(logits, -1).numpy()   # softmax over classes for a 16-way decision
P = zs_probs(emb["test"])
write("siglip-zs", [{"arm": "siglip-zs", "id": r["id"], "label": r["label"], "probs": dict(zip(KEYS, map(float, p))),
                     "pred": KEYS[int(p.argmax())], "latency_ms": float(l)} for r, p, l in zip(test, P, lat["test"])])
bl = blanks(); Eb, _ = embed(list(bl.values())); Pb = zs_probs(Eb.numpy())
# --------------- linear probe
from sklearn.linear_model import LogisticRegression
ytr = np.array([int(r["label_idx"]) for r in man if r["split"] == "train"]); ycal = np.array([int(r["label_idx"]) for r in man if r["split"] == "cal"])
best = None
for C in [0.1, 0.3, 1, 3, 10, 30, 100]:
    clf = LogisticRegression(C=C, max_iter=5000).fit(emb["train"], ytr)
    acc = (clf.predict(emb["cal"]) == ycal).mean(); nll = -np.log(clf.predict_proba(emb["cal"])[np.arange(len(ycal)), ycal] + 1e-12).mean()
    print("probe C", C, "cal acc", round(acc, 3), "nll", round(nll, 3), flush=True)
    if best is None or nll < best[1]: best = (C, nll, clf)
C, _, clf = best; json.dump({"C": C}, open(CACHE / "probe_C.json", "w"))
Pp = clf.predict_proba(emb["test"])
def probe_probs(p):
    by_name = dict(zip(OFFICIAL_KEYS, p))
    return np.array([by_name[k] for k in KEYS])
Pp = np.array([probe_probs(p) for p in Pp])
write("siglip-probe", [{"arm": "siglip-probe", "id": r["id"], "label": r["label"], "probs": dict(zip(KEYS, map(float, p))),
                        "pred": KEYS[int(p.argmax())], "latency_ms": float(l), "C": C} for r, p, l in zip(test, Pp, lat["test"])])
Pbp = np.array([probe_probs(p) for p in clf.predict_proba(Eb.numpy())])
recs = []
for (k, _), a, b in zip(bl.items(), Pb, Pbp):
    recs.append({"arm": "siglip-zs", "id": k, "control": "blank", "probs": dict(zip(KEYS, map(float, a))), "pred": KEYS[int(a.argmax())]})
    recs.append({"arm": "siglip-probe", "id": k, "control": "blank", "probs": dict(zip(KEYS, map(float, b))), "pred": KEYS[int(b.argmax())]})
write("siglip_controls", recs)
del model
# --------------- DiT fine-tuned on RVL-CDIP
DID = "microsoft/dit-base-finetuned-rvlcdip"
dm = AutoModelForImageClassification.from_pretrained(DID).eval(); dp = AutoImageProcessor.from_pretrained(DID)
id2 = dm.config.id2label; print(id2, flush=True)
# DiT uses the official RVL-CDIP order (0=letter, 1=form, ...); map by NAME to our KEYS (bug fixed 23:10 IST, was zip by index)
NAME2KEY = {"scientific report": "report", "scientific publication": "publication", "file folder": "file_folder", "news article": "news_article"}
DIT_ORDER = [KEYS.index(NAME2KEY.get(id2[i], id2[i]).replace(" ", "_")) for i in range(16)]
def dit_probs(p): q = np.zeros(16); q[DIT_ORDER] = p; return q
recs = []
for i in range(0, len(test), 8):
    rows = test[i:i+8]
    with torch.no_grad():
        x = dp(images=[img(r) for r in rows], return_tensors="pt"); t0 = time.perf_counter()
        pr = np.array([dit_probs(v) for v in torch.softmax(dm(**x).logits, -1).numpy()]); dt = (time.perf_counter() - t0) * 1000 / len(rows)
    for r, p in zip(rows, pr):
        recs.append({"arm": "dit-ft", "id": r["id"], "label": r["label"], "probs": dict(zip(KEYS, map(float, p))),
                     "pred": KEYS[int(p.argmax())], "latency_ms": dt})
write("dit-ft", recs)
with torch.no_grad():
    pb = np.array([dit_probs(v) for v in torch.softmax(dm(**dp(images=list(bl.values()), return_tensors="pt")).logits, -1).numpy()])
write("dit_controls", [{"arm": "dit-ft", "id": k, "control": "blank", "probs": dict(zip(KEYS, map(float, p))), "pred": KEYS[int(p.argmax())]} for k, p in zip(bl, pb)])
print("DONE")
