"""Shared harness for the vision-decision experiments (Tech in General, Oct 2026).

Arms share one interface:  arm.decide(image_bytes|None, state, question) -> dict
   {probs: {option: p}, pred, latency_ms, usage, raw}
A *question* is {"name","instructions","criteria": {option: description|None}}.

Local arms: Ollama System One decision models (d1, d1-omni), Ollama chat VLM (qwen3-vl) with
enum-constrained output + first-token logprobs.  Paid arms are implemented but HARD-GATED:
they refuse to send anything unless ALLOW_PAID_CALLS == "yes-approved-by-hari" AND the key is set.
"""
from __future__ import annotations
import base64, io, json, math, os, time, hashlib
from pathlib import Path
import requests
from PIL import Image

OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11535")
NUM_THREAD = int(os.getenv("VDEC_THREADS", "4"))
MAX_SIDE = int(os.getenv("VDEC_MAX_SIDE", "512"))
PAID_GATE = "yes-approved-by-hari"


def load_png_bytes(path_or_img, max_side=MAX_SIDE) -> bytes:
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = im.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy(); im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def loadavg():
    try:
        return os.getloadavg()[0]
    except Exception:
        return None


def _norm(d):
    s = sum(d.values())
    return {k: (v / s if s > 0 else 1 / len(d)) for k, v in d.items()}


class OllamaDecision:
    """Ollama /v1/systemone (d1:3b, d1-omni:600m, clef-flash...)."""
    kind = "decision"
    def __init__(self, model):
        self.model = model; self.name = model
    def decide(self, img, state, q, timeout=1800):
        body = {"model": self.model, "state": state,
                "questions": {q["name"]: {"type": "choice", "instructions": q["instructions"],
                                          "criteria": q["criteria"]}},
                "options": {"num_thread": NUM_THREAD}, "keep_alive": "30m"}
        if img is not None:
            body["images"] = [base64.b64encode(img).decode()]
        la = loadavg(); t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/v1/systemone", json=body, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status(); j = r.json()
        a = j["answers"][q["name"]]
        probs = _norm({k: float(a["probabilities"].get(k, 0.0)) for k in q["criteria"]})
        return {"probs": probs, "pred": max(probs, key=probs.get), "latency_ms": dt,
                "usage": j.get("usage"), "loadavg": la, "raw": a}


class OllamaChatVLM:
    """Generic chat VLM via Ollama /api/chat, output constrained to the option enum (JSON string),
    probability = first label-token top_logprobs, mapped to options by unique prefix.
    Options must have distinct first tokens for exact probabilities; 'coverage' records mass found."""
    kind = "chat"
    def __init__(self, model):
        self.model = model; self.name = model
    def prompt(self, state, q):
        opts = "\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in q["criteria"].items())
        return (f"{state}\n\nQuestion: {q['instructions']}\nOptions:\n{opts}\n"
                "Answer with exactly one option name from the list and nothing else.")
    def decide(self, img, state, q, timeout=1800):
        opts = list(q["criteria"])
        msg = {"role": "user", "content": self.prompt(state, q)}
        if img is not None:
            msg["images"] = [base64.b64encode(img).decode()]
        body = {"model": self.model, "messages": [msg], "stream": False,
                "format": {"type": "string", "enum": opts}, "logprobs": True, "top_logprobs": 20,
                "options": {"temperature": 0, "num_thread": NUM_THREAD, "num_predict": 16, "seed": 0},
                "keep_alive": "30m"}
        la = loadavg(); t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status(); j = r.json()
        text = (j.get("message", {}).get("content") or "").strip().strip('"')
        lps = j.get("logprobs") or []
        # first entry is the opening quote of the JSON string; take the first non-quote token
        entry = next((e for e in lps if e["token"].strip() not in ('"', "")), None)
        mass = dict.fromkeys(opts, 0.0)
        if entry:
            for t in entry.get("top_logprobs", []):
                tok = t["token"].strip().lower()
                if not tok:
                    continue
                cands = [o for o in opts if o.lower().startswith(tok)]
                for o in cands:
                    mass[o] += math.exp(t["logprob"]) / len(cands)
        cov = sum(mass.values())
        if cov == 0 and text in mass:
            mass[text] = 1.0
        probs = _norm(mass)
        pred = text if text in probs else max(probs, key=probs.get)
        return {"probs": probs, "pred": pred, "argmax": max(probs, key=probs.get), "text": text,
                "coverage": cov, "latency_ms": dt,
                "usage": {"input_tokens": j.get("prompt_eval_count"), "output_tokens": j.get("eval_count")},
                "loadavg": la}


# ---------------------------------------------------------------- paid arms (gated, zero calls made)
class PaidCallBlocked(RuntimeError):
    pass


def _paid_guard(env_key):
    if os.getenv("ALLOW_PAID_CALLS") != PAID_GATE:
        raise PaidCallBlocked("Paid API calls are not approved (set ALLOW_PAID_CALLS only after Hari approves).")
    if not os.getenv(env_key):
        raise PaidCallBlocked(f"{env_key} missing")
    return os.getenv(env_key)


class OpenRouterDecision:
    """OpenRouter Decisions API (e.g. openai/gpt-6-luna-decisions, liquid/d1, typesafe/jev-1.14).
    Image goes as an image_url part inside `state` (OpenRouter ignores TypeSafe `assets`)."""
    kind = "decision-paid"
    URL = "https://openrouter.ai/api/v1/systemone"
    def __init__(self, model):
        self.model = model; self.name = "openrouter:" + model
    def build(self, img, state, q):
        st = [{"type": "text", "text": state}]
        if img is not None:
            st.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}})
        return {"model": self.model, "state": st,
                "questions": {q["name"]: {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}}}
    def decide(self, img, state, q, timeout=120):
        key = _paid_guard("OPENROUTER_API_KEY")
        t0 = time.perf_counter()
        r = requests.post(self.URL, json=self.build(img, state, q), headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000; r.raise_for_status(); j = r.json()
        a = j["answers"][q["name"]]; probs = _norm({k: float(a["probabilities"].get(k, 0)) for k in q["criteria"]})
        return {"probs": probs, "pred": max(probs, key=probs.get), "latency_ms": dt, "usage": j.get("usage")}


class TypeSafeJev(OpenRouterDecision):
    """TypeSafe direct: images via `assets` map + header TypeSafe-Danger-Zone-Images: true (preview)."""
    URL = "https://api.typesafe.ai/v1/systemone"
    def __init__(self, model="jev-1.14.0"):
        self.model = model; self.name = "typesafe:" + model
    def build(self, img, state, q):
        body = {"model": self.model, "state": {"text": state, "image": "asset:img0"},
                "questions": {q["name"]: {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}}}
        if img is not None:
            body["assets"] = {"img0": {"type": "image/png", "data": base64.b64encode(img).decode()}}
        return body
    def decide(self, img, state, q, timeout=120):
        key = _paid_guard("TYPESAFE_API_KEY")
        r = requests.post(self.URL, json=self.build(img, state, q), timeout=timeout,
                          headers={"Authorization": f"Bearer {key}", "TypeSafe-Danger-Zone-Images": "true"})
        r.raise_for_status(); j = r.json(); a = j["answers"][q["name"]]
        probs = _norm({k: float(a["probabilities"].get(k, 0)) for k in q["criteria"]})
        return {"probs": probs, "pred": max(probs, key=probs.get), "usage": j.get("usage")}


class CloudflareClef(OpenRouterDecision):
    """Workers AI @cf/cloudflare/clef-flash (or clef). Keep base64 < ~300 KB (413 otherwise)."""
    def __init__(self, model="@cf/cloudflare/clef-flash"):
        self.model = model; self.name = "cloudflare:" + model
    def decide(self, img, state, q, timeout=120):
        key = _paid_guard("CLOUDFLARE_API_TOKEN"); acct = _paid_guard("CLOUDFLARE_ACCOUNT_ID")
        url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{self.model}"
        body = {"state": state, "questions": self.build(img, state, q)["questions"]}
        if img is not None:
            body["images"] = [base64.b64encode(img).decode()]
        r = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        r.raise_for_status(); j = r.json().get("result", {}); a = j["answers"][q["name"]]
        probs = _norm({k: float(a["probabilities"].get(k, 0)) for k in q["criteria"]})
        return {"probs": probs, "pred": max(probs, key=probs.get), "usage": j.get("usage")}


ARMS = {"d1": lambda: OllamaDecision("d1:3b"), "d1-omni": lambda: OllamaDecision("d1-omni:600m"),
        "qwen3-vl-2b": lambda: OllamaChatVLM("qwen3-vl:2b-instruct")}


def run_items(arm, items, out_path, log_every=10):
    """items: iterable of dicts {id, image(bytes|None), state, question, **meta}. Resumable jsonl."""
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                j = json.loads(l)
                if "error" not in j: done.add(j["id"])
            except Exception:
                pass
    n = 0
    with out.open("a") as f:
        for it in items:
            if it["id"] in done:
                continue
            meta = {k: v for k, v in it.items() if k not in ("image", "question", "state")}
            try:
                res = arm.decide(it["image"], it["state"], it["question"])
                rec = {"arm": arm.name, **meta, **{k: v for k, v in res.items() if k != "raw"}}
            except Exception as e:
                rec = {"arm": arm.name, **meta, "error": repr(e)[:500]}
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % log_every == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {arm.name} {n} new -> {out.name}", flush=True)
    return n
