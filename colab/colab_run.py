"""Run every request in requests.jsonl through local Ollama decision models on a Colab T4. Resumable (jsonl per model),
hard time cap, optional Drive sync. Usage: python colab_run.py --models clef-flash:9b,d1:3b [--smoke] [--cap-min 80] [--out DIR]"""
import argparse, base64, io, json, os, time, shutil
from pathlib import Path
import requests, sys
sys.path.insert(0, str(Path(__file__).parent))
ap = argparse.ArgumentParser(); ap.add_argument("--models", default="clef-flash:9b,d1:3b,qwen3-vl:2b-instruct"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--cap-min", type=float, default=80); ap.add_argument("--out", default="results"); ap.add_argument("--sync", default="")
ap.add_argument("--url", default="http://127.0.0.1:11434"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
os.environ.setdefault("OLLAMA_URL", a.url); os.environ.setdefault("VDEC_MAX_SIDE", "4096")
import vdec
HERE = Path(__file__).parent; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
reqs = [json.loads(l) for l in open(HERE / "requests.jsonl")]
if a.smoke:   # 3 per (exp, tag)
    cnt = {}; sm = []
    for r in reqs:
        k = (r["exp"], r["tag"]); cnt[k] = cnt.get(k, 0) + 1
        if cnt[k] <= 3: sm.append(r)
    reqs = sm
if a.limit: reqs = reqs[:a.limit]
t_start = time.time(); stop = False
for model in a.models.split(","):
    f = OUT / f"{model.replace(':', '_')}.jsonl"; done = set()
    if f.exists():
        for l in f.read_text().splitlines():
            j = json.loads(l)
            if "error" not in j: done.add((j["exp"], j["tag"], j["id"]))
    with f.open("a") as fh:
        for n, r in enumerate(reqs):
            if (r["exp"], r["tag"], r["id"]) in done: continue
            if (time.time() - t_start) / 60 > a.cap_min: stop = True; break
            q = r["question"]
            img = None
            if r["image"]:
                from PIL import Image
                buf = io.BytesIO(); Image.open(HERE / r["image"]).convert("RGB").save(buf, "PNG"); img = buf.getvalue()
            arm = vdec.OllamaChatVLM(model) if "qwen" in model or "vl" in model else vdec.OllamaDecision(model)
            base = {k: v for k, v in r.items() if k not in ("state", "question", "image")}
            try:
                res = arm.decide(img, r["state"], q, timeout=600)
                rec = {**base, "arm": model, **{k: v for k, v in res.items() if k != "raw"}}
            except Exception as e:
                rec = {**base, "arm": model, "error": repr(e)[:300]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if n % 50 == 0:
                print(f"{model} {n}/{len(reqs)} {(time.time()-t_start)/60:.1f} min", flush=True)
                if a.sync: shutil.copy(f, Path(a.sync) / f.name)
    if a.sync and f.exists(): shutil.copy(f, Path(a.sync) / f.name)
    if stop: print("TIME CAP HIT — stopping cleanly"); break
print("done", (time.time() - t_start) / 60, "min")
