"""One-off exact fix (23:10 IST) for outputs written by the buggy run_torch_arms.py: siglip-probe and dit-ft probabilities were
indexed in the official RVL-CDIP order but stored under our KEYS. Re-keys them (pure permutation, no re-inference). Idempotent."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code")); from labels import KEYS
OFF = ["letter","form","email","handwritten","advertisement","report","publication","specification","file_folder","news_article","budget","invoice","presentation","questionnaire","resume","memo"]
FIX = dict(zip(KEYS, OFF))   # stored-under key -> true class
for f, arms in [("siglip-probe", None), ("dit-ft", None), ("siglip_controls", {"siglip-probe"}), ("dit_controls", None)]:
    p = ROOT / f"results/raw/{f}.jsonl"
    if not p.exists(): print("missing", f); continue
    out = []; n = 0
    for l in p.read_text().splitlines():
        r = json.loads(l)
        if (arms is None or r["arm"] in arms) and not r.get("label_order_fixed"):
            pr = {FIX[k]: v for k, v in r["probs"].items()}; r["probs"] = {k: pr[k] for k in KEYS}
            r["pred"] = max(r["probs"], key=r["probs"].get); r["label_order_fixed"] = True; n += 1
        out.append(json.dumps(r))
    p.write_text("\n".join(out) + "\n"); print(f, "fixed", n)
