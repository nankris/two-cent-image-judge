"""Merge a returned Colab zip into the experiment folders as results/raw/colab/<model>__<tag>.jsonl.
usage: /workspace/tig-env/bin/python merge_results.py returned/results_vision-clef_*.zip"""
import sys, json, zipfile, collections
from pathlib import Path
X = Path("/workspace/experiments-vision-2026-10")
DIRS = {"01": "01-two-cent-image-judge", "02": "02-pixel-flip", "03": "03-screenshot-gatekeeper", "04": "04-typographic-injection"}
z = zipfile.ZipFile(sys.argv[1]); groups = collections.defaultdict(list)
for n in z.namelist():
    if not n.endswith(".jsonl"): continue
    for l in z.read(n).decode().splitlines():
        r = json.loads(l)
        if "error" in r: continue
        groups[(r["exp"], r["arm"], r["tag"])].append(r)
for (e, m, t), rs in groups.items():
    d = X / DIRS[e] / "results/raw/colab"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"{m.replace(':', '_')}__{t}.jsonl"; f.write_text("".join(json.dumps(r) + "\n" for r in rs)); print(f, len(rs))
