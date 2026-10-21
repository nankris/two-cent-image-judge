# vision-clef — Colab T4 job (prepared 2026-10-21 22:30 IST, NOT run)

**Why.** `clef-flash:9b` (11 GB Ollama tag) cannot load on the box (15 GB RAM shared, no GPU). d1:3b runs at ~55 s/image
and qwen3-vl-2b at ~40 s/image on the shared CPU, so the box only finishes pre-registered *subsets* overnight. This job replays
the exact 1,870 requests of experiments 01-04 on a T4 for clef-flash:9b, d1:3b, qwen3-vl:2b-instruct and d1-omni:600m.

**What it proves.** (a) clef-flash numbers for every experiment (descriptive arm; it is not part of any pre-registered
hypothesis, so its results are labelled "added arm, no hypothesis"); (b) full-n d1 / qwen runs so the pre-registered
hypotheses can be re-scored on the full sets (the box subsets remain the primary evidence if Colab is never run; with
both, we report box-subset verdict and full-n verdict side by side, and check GPU-vs-CPU agreement on the shared items
— Q4 GGUF on CUDA vs CPU is not bit-identical, so agreement is itself reported).

**Pre-registered rules:** in each experiment README (unchanged). Requests: `bundle/requests.jsonl` (exp, tag, id, state,
question, image file, label/meta), images are lossless WebP of the exact pixels the box sends.

**Estimate on T4 (unmeasured):** setup+pulls ~15 min (~17 GB download); clef-flash ~1.5 s/req, d1 ~0.6, qwen ~0.8,
d1-omni ~0.3 → ~100 min inference → **~2 h total ≈ 2.3 CU at the ~1.1 CU/h measured elsewhere, budget 4 CU at 2 CU/h.
Hard cap 150 min = 5 CU max.** If clef-flash is slower than 3 s/req, the cap stops it before qwen/d1-omni; results are still zipped.

**How to run.** 1) Open `vision-clef.ipynb` in Colab, T4 runtime. 2) Leave `SMOKE=True`, Run all (~10 min, ~0.3 CU), check cell 4
shows 0 errors. 3) Set `SMOKE=False`, Run all again (bundle/pulls are cached in the same runtime). 4) Download the zip, put it in
`/workspace/colab-jobs/vision-clef/returned/`, then `/workspace/tig-env/bin/python merge_results.py returned/<zip>` and re-run each
experiment's `code/score.py`.

**Outputs.** `<model>.jsonl` per model (one row per request: probs, pred, latency_ms, usage), `ollama.log`, `env.json`.
Rebuild: `./build_bundle.sh` (re-exports requests from the experiment manifests). Validated on the box: notebook JSON
(nbformat), code-cell syntax, and `colab_run.py --limit` against the box Ollama (d1-omni) — see `_agent_logs/`.
