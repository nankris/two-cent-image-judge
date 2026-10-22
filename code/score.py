"""Score exp1. Writes results/summary.json, results/summary.md, results/per_item.csv, figures/*.png"""
import sys, json, csv, io
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "_shared")); sys.path.insert(0, str(ROOT / "code"))
from stats import ece, brier, macro_f1, bootstrap, fmt_ci
from common_io import load_jsonl, PRICES, VM_USD_PER_H, clef_image_tokens, luna_image_tokens
from labels import KEYS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
man = [r for r in csv.DictReader(open(ROOT / "data/manifest.csv")) if r["split"] == "test"]
ids = [r["id"] for r in man]; lab = {r["id"]: r["label"] for r in man}
sub20 = set()
core = set()
by = {}
for r in man: by.setdefault(r["label"], []).append(r["id"])
for k in KEYS:
    core |= set(sorted(by[k])[:10])
    sub20 |= set(sorted(by[k])[:20])
ARMS = ["d1", "d1-omni", "qwen3-vl-2b", "siglip-zs", "siglip-probe", "dit-ft"]
R = {}; errs = {}
for a in ARMS:
    R[a], errs[a] = load_jsonl(ROOT / f"results/raw/{a}.jsonl")
def mats(a, idset):
    rows = [R[a][i] for i in ids if i in idset and i in R[a]]
    P = np.array([[r["probs"][k] for k in KEYS] for r in rows]); y = np.array([KEYS.index(r["label"]) for r in rows])
    pred = np.array([KEYS.index(r["pred"]) if r["pred"] in KEYS else int(np.argmax(p)) for r, p in zip(rows, P)])
    lat = np.array([r.get("latency_ms", np.nan) for r in rows]); return rows, P, y, pred, lat
def metrics(P, y, pred):
    conf = P.max(1); corr = (pred == y)
    f = lambda idx: (corr[idx].mean())
    return {"acc": bootstrap(lambda i: corr[i].mean(), len(y), strata=y),
            "macro_f1": bootstrap(lambda i: macro_f1(y[i], pred[i], range(16)), len(y), strata=y),
            "ece": bootstrap(lambda i: ece(conf[i], corr[i]), len(y), strata=y),
            "brier": bootstrap(lambda i: brier(P[i], y[i]), len(y), strata=y)}
summ = {}; full = set(ids)
for a in ARMS:
    if not R[a]: continue
    # Primary analysis is the frozen, balanced 10/class core completed by every
    # arm.  Do not let resumable background runs silently change the estimand.
    idset = core
    rows, P, y, pred, lat = mats(a, idset)
    n = len(rows); m = metrics(P, y, pred)
    tok = [r.get("usage", {}).get("input_tokens") for r in rows if r.get("usage")]
    summ[a] = {"n": n, "historical_error_lines": errs[a], "valid_rate_core": n / len(core), **{k: v for k, v in m.items()},
               "lat_p50_ms": float(np.nanmedian(lat)), "lat_p95_ms": float(np.nanpercentile(lat, 95)),
               "mean_input_tokens": float(np.mean(tok)) if tok else None,
               "usd_per_1k_local_vm": float(np.nanmedian(lat)) / 1000 / 3600 * VM_USD_PER_H * 1000,
               "loadavg_median": float(np.nanmedian([r.get("loadavg") or np.nan for r in rows])) if rows and "loadavg" in rows[0] else None}
    if a != "qwen3-vl-2b" and R.get("qwen3-vl-2b"):
        rows2, P2, y2, pred2, _ = mats(a, sub20)
        if len(rows2): summ[a]["on_sub20"] = {k: v for k, v in metrics(P2, y2, pred2).items()}
# paired comparisons
def paired(a, b, metric, idset):
    common = [i for i in ids if i in idset and i in R[a] and i in R[b]]
    def arr(x):
        P = np.array([[R[x][i]["probs"][k] for k in KEYS] for i in common]); y = np.array([KEYS.index(lab[i]) for i in common])
        pred = np.array([KEYS.index(R[x][i]["pred"]) for i in common]); return P, y, pred
    Pa, y, pa = arr(a); Pb, _, pb = arr(b)
    if metric == "macro_f1": fn = lambda i: macro_f1(y[i], pa[i], range(16)) - macro_f1(y[i], pb[i], range(16))
    elif metric == "acc": fn = lambda i: (pa[i] == y[i]).mean() - (pb[i] == y[i]).mean()
    elif metric == "ece": fn = lambda i: ece(Pa[i].max(1), pa[i] == y[i]) - ece(Pb[i].max(1), pb[i] == y[i])
    return {"n": len(common), "diff": bootstrap(fn, len(common), strata=y)}
comp = {}
for a, b, m, s in [("d1", "siglip-zs", "macro_f1", core), ("qwen3-vl-2b", "d1", "ece", core), ("d1", "siglip-probe", "macro_f1", core),
                   ("d1-omni", "d1", "macro_f1", core), ("d1", "qwen3-vl-2b", "macro_f1", core), ("d1", "dit-ft", "macro_f1", core),
                   ("siglip-probe", "siglip-zs", "macro_f1", core)]:
    if R.get(a) and R.get(b): comp[f"{a} - {b} [{m}]"] = paired(a, b, m, s)
# controls
ctrl = {}
for a in ["d1", "d1-omni", "qwen3-vl-2b"]:
    c, _ = load_jsonl(ROOT / f"results/raw/{a}_controls.jsonl")
    ctrl[a] = {k: {"pred": v["pred"], "p_max": max(v["probs"].values())} for k, v in c.items()}
for f in ["siglip_controls", "dit_controls"]:
    for l in (ROOT / f"results/raw/{f}.jsonl").read_text().splitlines() if (ROOT / f"results/raw/{f}.jsonl").exists() else []:
        j = json.loads(l); ctrl.setdefault(j["arm"], {})[j["id"]] = {"pred": j["pred"], "p_max": max(j["probs"].values())}
rev, _ = load_jsonl(ROOT / "results/raw/d1_reversed.jsonl")
if rev:
    same = [rev[i]["pred"] == R["d1"][i]["pred"] for i in rev if i in R["d1"]]
    acc_r = np.mean([rev[i]["pred"] == lab[i] for i in rev]); acc_o = np.mean([R["d1"][i]["pred"] == lab[i] for i in rev if i in R["d1"]])
    ctrl["d1_option_order"] = {"n": len(rev), "agreement": float(np.mean(same)), "acc_reversed": float(acc_r), "acc_original_same_items": float(acc_o)}
# hosted cost estimates (list price x tokens); text tokens from d1 label-only control
c1, _ = load_jsonl(ROOT / "results/raw/d1_controls.jsonl")
text_tok = c1.get("label_only", {}).get("usage", {}).get("input_tokens") if c1 else None
sizes = []
from PIL import Image
for r in man:
    w, h = int(r["w"]), int(r["h"]); s = min(1, 512 / max(w, h)); sizes.append((int(w * s), int(h * s)))
tt = text_tok or 230
d1_tok = summ.get("d1", {}).get("mean_input_tokens") or 279
est = {"text_tokens_assumed": tt,
       "openai/gpt-6-luna-decisions": np.mean([luna_image_tokens(*s) for s in sizes]) + tt,
       "cloudflare/clef-flash": np.mean([clef_image_tokens(*s) for s in sizes]) + tt,
       "cloudflare/clef": np.mean([clef_image_tokens(*s) for s in sizes]) + tt,
       "typesafe/jev-1.14": np.mean([luna_image_tokens(*s) for s in sizes]) + tt,   # unknown tokenizer: Luna-like assumption
       "liquid/d1": d1_tok}
cost = {m: {"tokens_per_image": float(t), "usd_per_1k": float(t) * PRICES[m] / 1e6 * 1000, "usd_for_this_exp_640+controls": float(t) * PRICES[m] / 1e6 * (640 + 7 + 64)}
        for m, t in est.items() if m in PRICES}
json.dump({"arms": summ, "paired": comp, "controls": ctrl, "hosted_cost_estimate": cost, "text_tokens": tt}, open(ROOT / "results/summary.json", "w"), indent=1)
# markdown table
L = ["| arm | n | accuracy [95% CI] | macro-F1 [95% CI] | ECE | Brier | p50 / p95 latency (s) | $ per 1k (local VM) |", "|---|---|---|---|---|---|---|---|"]
for a, s in summ.items():
    L.append(f"| {a} | {s['n']} | {fmt_ci(s['acc'], pct=True)} | {fmt_ci(s['macro_f1'])} | {fmt_ci(s['ece'])} | {s['brier'][0]:.3f} | {s['lat_p50_ms']/1000:.2f} / {s['lat_p95_ms']/1000:.2f} | {s['usd_per_1k_local_vm']:.3f} |")
L += ["", "| paired difference | n | point [95% CI] |", "|---|---|---|"] + [f"| {k} | {v['n']} | {fmt_ci(v['diff'])} |" for k, v in comp.items()]
(ROOT / "results/summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); print(json.dumps(ctrl, indent=0)[:2500]); print(json.dumps(cost, indent=0))
# per item
with open(ROOT / "results/per_item.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "label"] + [f"{a}_pred" for a in ARMS] + [f"{a}_pmax" for a in ARMS])
    for i in ids: w.writerow([i, lab[i]] + [R[a].get(i, {}).get("pred", "") for a in ARMS] + [f"{max(R[a][i]['probs'].values()):.4f}" if i in R[a] else "" for a in ARMS])
# figures
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
arms_here = [a for a in ARMS if a in summ]
fig, axs = plt.subplots(1, len(arms_here), figsize=(3.2 * len(arms_here), 3.4), sharey=True)
for ax, a in zip(np.atleast_1d(axs), arms_here):
    _, P, y, pred, _ = mats(a, core); conf = P.max(1); corr = pred == y; edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.scatter(xs, ys, s=[10 + n / 2 for n in ns]); ax.plot(xs, ys)
    ax.set_title(f"{a}\nECE {summ[a]['ece'][0]:.3f}", fontsize=9); ax.set_xlabel("confidence")
np.atleast_1d(axs)[0].set_ylabel("accuracy"); fig.tight_layout(); fig.savefig(FIG / "reliability.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.4, 4))
for a in arms_here:
    ax.errorbar(summ[a]["usd_per_1k_local_vm"], summ[a]["macro_f1"][0], yerr=[[summ[a]["macro_f1"][0] - summ[a]["macro_f1"][1]], [summ[a]["macro_f1"][2] - summ[a]["macro_f1"][0]]], fmt="o", capsize=3)
    ax.annotate(a, (summ[a]["usd_per_1k_local_vm"], summ[a]["macro_f1"][0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("$ per 1,000 images (local CPU VM time, log)"); ax.set_ylabel("macro-F1 (95% CI)"); ax.set_title("Cost vs quality, 16-class RVL-CDIP (matched core n=160)")
fig.tight_layout(); fig.savefig(FIG / "cost_vs_f1.png", dpi=150); plt.close(fig)
if "d1" in summ:
    _, P, y, pred, _ = mats("d1", core); C = np.zeros((16, 16), int)
    for t, p in zip(y, pred): C[t, p] += 1
    fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(C, cmap="Blues"); ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels(KEYS, rotation=70, fontsize=7); ax.set_yticklabels(KEYS, fontsize=7)
    for i in range(16):
        for j in range(16):
            if C[i, j]: ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6, color="white" if C[i, j] > 20 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("d1-3B confusion (matched core n=160)"); fig.tight_layout(); fig.savefig(FIG / "confusion_d1.png", dpi=150); plt.close(fig)
# ---------------------------------------------------------------- pre-registered verdicts (added 22:35 IST;
# rules copied verbatim from README; each comparison uses the largest set of items both arms share, n reported)
def rule_margin(ci, margin, need_point=None):
    p, lo, hi = ci
    if lo > margin and (need_point is None or p >= need_point): return "PASS"
    if hi < (need_point if need_point is not None else margin): return "FAIL"
    return "INCONCLUSIVE"
V = {}
def pv(name, a, b, metric, rule):
    if not (R.get(a) and R.get(b)): V[name] = {"verdict": "PENDING (arm missing)"}; return
    r = paired(a, b, metric, core)
    if r["n"] < 32: V[name] = {"verdict": f"PENDING (n={r['n']} shared)", **r}; return
    V[name] = {"verdict": rule(r["diff"]), **r}
pv("H1 d1 - siglip-zs macro-F1 >= +10pt", "d1", "siglip-zs", "macro_f1", lambda c: rule_margin(c, 0.0, 0.10))
pv("H2 ECE(qwen) - ECE(d1) > 0", "qwen3-vl-2b", "d1", "ece", lambda c: "PASS" if c[1] > 0 else "FAIL" if c[2] < 0 else "INCONCLUSIVE")
pv("H3 d1 - siglip-probe macro-F1 > -5pt", "d1", "siglip-probe", "macro_f1", lambda c: "PASS" if c[1] > -0.05 else "FAIL" if c[2] < -0.05 else "INCONCLUSIVE")
pv("H4 d1-omni - d1 macro-F1 > -10pt", "d1-omni", "d1", "macro_f1", lambda c: "PASS" if c[1] > -0.10 else "FAIL" if c[2] < -0.10 else "INCONCLUSIVE")
V["validity"] = {a: {"valid_core": summ[a]["n"], "target_core": len(core),
                       "historical_error_lines": errs[a], "valid_share_of_core": summ[a]["n"] / len(core)} for a in summ}
V["sanity"] = {"probe>=zs": (summ["siglip-probe"]["macro_f1"][0] >= summ["siglip-zs"]["macro_f1"][0]) if "siglip-probe" in summ and "siglip-zs" in summ else None,
               "dit acc>=0.80": (summ["dit-ft"]["acc"][0] >= 0.80) if "dit-ft" in summ else None}
json.dump(V, open(ROOT / "results/verdicts.json", "w"), indent=1)
L = ["| hypothesis | shared n | difference [95% CI] | verdict |", "|---|---|---|---|"]
for k, v in V.items():
    if k.startswith("H"): L.append(f"| {k} | {v.get('n','-')} | {fmt_ci(v['diff']) if 'diff' in v else '-'} | {v['verdict']} |")
(ROOT / "results/verdicts.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); print(V["validity"], V["sanity"])
