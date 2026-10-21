import json
from pathlib import Path
def load_jsonl(p):
    """Last non-error record per id (runs are resumable; error records are retried)."""
    p = Path(p); out = {}; errs = 0
    if not p.exists(): return {}, 0
    for l in p.read_text().splitlines():
        try: j = json.loads(l)
        except Exception: continue
        if "error" in j:
            errs += 1; continue
        out[j["id"]] = j
    return out, errs
PRICES = {  # USD per 1M input tokens, OpenRouter /api/v1/models (decisions, image input), fetched 2026-10-21 22:00 IST
    "openai/gpt-6-luna-decisions": 0.10, "cloudflare/clef-flash": 0.038, "cloudflare/clef": 0.24,
    "typesafe/jev-1.14": 0.042, "liquid/d1": 0.02, "liquid/d1-ultra": 0.042}
VM_USD_PER_H = 0.357   # AWS c7i.2xlarge (8 vCPU) on-demand list, us-east-1 — whole VM charged per sequential request
def clef_image_tokens(w, h): return min(1024, (w // 32) * (h // 32) + 3)
def luna_image_tokens(w, h):
    import math
    s = min(1.0, 2048 / max(w, h)); w, h = w * s, h * s
    s2 = min(1.0, 768 / min(w, h)); w, h = w * s2, h * s2
    return 85 + 170 * math.ceil(w / 512) * math.ceil(h / 512)
