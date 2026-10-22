# The 2-Cent Image Judge — open image *decision models* vs a VLM vs SigLIP vs a fine-tuned ViT on real document scans

Part of the *Tech in General* vision-decision series (Oct 2026). Everything runs on an 8-vCPU / 15 GB / no-GPU box,
shared with other jobs. Paid APIs are wired in but **not called**. At 23:20 IST on 2026-10-21 I decided to run free models only, so no paid baseline was run.

## Pre-registration (written 2026-10-21 21:39 IST, before any test-set call)

**Question.** Can a small open *decision model* (Liquid d1-3B / d1-omni-600M, which score every option in one pass and
return probabilities) route real scanned documents into 16 classes well enough — and honestly enough — to replace either a
chat VLM or a trained classifier, at near-zero cost?

**Data.** RVL-CDIP **test** split (parquet mirror `chainyo/rvl-cdip`), 40 images per class x 16 = **640**, sampled with
`numpy.random.default_rng(20261021)` (`code/prep_data.py`, `data/manifest.csv` with pixel + file sha256).
Training pool for the linear probe: `jordyvl/rvl_cdip_100_examples_per_class` train (800); calibration/selection
pool: its validation split (400). Pixel-hash overlap of train/cal with test: **0** (checked in prep).
All images are downscaled to max side **512 px** PNG before being sent to Ollama arms (same bytes for every Ollama arm).

**Arms (all local, zero cost except electricity/VM time).**
| id | model | how | n |
|---|---|---|---|
| d1 | `d1:3b` (Ollama 0.40.7, Q4 GGUF, 3.7 GB) | `/v1/systemone` choice question, 16 options with one-line descriptions | 640 |
| d1-omni | `d1-omni:600m` (Ollama, 850 MB) | same request | 640 |
| qwen3-vl-2b | `qwen3-vl:2b-instruct` (Ollama Q4, 1.9 GB) | `/api/chat`, JSON-schema enum over the 16 option keys, temperature 0, P = first-label-token top-20 logprobs | 320 (first 20 per class by id; CPU-time budget) — extended to 640 if time allows |
| siglip-zs | `google/siglip2-base-patch16-384` | zero-shot, prompt "a scanned document: {description}" | 640 |
| siglip-probe | same encoder | L2-normalised image embedding -> multinomial logistic regression trained on the 800 train images (C picked on the 400 cal images) | 640 |
| dit-ft | `microsoft/dit-base-finetuned-rvlcdip` | fine-tuned ViT (trained by Microsoft on the full RVL-CDIP train split) = supervised ceiling | 640 |
Paid arms implemented in `_shared/vdec.py` but hard-gated (`ALLOW_PAID_CALLS`): `openai/gpt-6-luna-decisions` (OpenRouter),
`@cf/cloudflare/clef-flash` (Workers AI), `typesafe/jev-1.14` (TypeSafe, header `TypeSafe-Danger-Zone-Images: true`).
`clef-flash:9b` locally is 11 GB (Ollama) — does not fit next to the other jobs in 15 GB RAM -> Colab T4 notebook (`colab/clef-flash-t4.ipynb`).

**Metrics.** accuracy, macro-F1, top-label ECE (15 equal-width bins), multiclass Brier, reliability diagram, CPU latency
p50/p95 (sequential, 4 threads, box shared — load average logged per call), $/1,000 images.
CIs: 95% percentile bootstrap, 2,000 resamples, seed 0, stratified by class; paired bootstrap for differences
(same resampled images for both arms). Local $/1k = p50 latency x $0.357/h (AWS c7i.2xlarge 8 vCPU on-demand list,
charged for the whole VM — conservative). Hosted $/1k = list price x measured/estimated input tokens.

**Controls.** (a) *Blank image*: 6 blank images (white, black, mid-gray, 3 seeded uniform-noise) x each Ollama arm;
(b) *Label-only*: no image, same state + question. Expected: a constant answer = 6.25% (chance). These detect
label-prior answering. (c) Option-order control for d1: 64 test images (4/class) with the option order reversed.

**Hypotheses and decision rules (fixed now).**
- **H1 (decision model beats contrastive zero-shot):** macro-F1(d1) − macro-F1(siglip-zs) ≥ +10 pts, PASS if the paired
  95% CI lower bound of the difference is > 0 **and** the point estimate ≥ 10 pts. FAIL if the upper bound < 10 pts.
  Otherwise INCONCLUSIVE.
- **H2 (calibration):** ECE(d1) < ECE(qwen3-vl-2b) on the shared 320 images; PASS if the paired 95% CI of
  ECE(qwen) − ECE(d1) excludes 0 on the positive side; FAIL if it excludes 0 on the negative side.
- **H3 ("replace a cheap trained classifier"):** macro-F1(d1) ≥ macro-F1(siglip-probe) − 5 pts. PASS if the paired CI
  lower bound of d1 − probe > −5 pts; FAIL if its upper bound < −5 pts; else INCONCLUSIVE.
- **H4 (tiny model):** macro-F1(d1-omni) ≥ macro-F1(d1) − 10 pts, same CI rule as H3 with a 10-pt margin.
- **Validity:** ≥ 98% valid (parsed, non-error) responses per arm; else the arm is reported as invalid.
- **Control check:** blank-image and label-only accuracy ≤ 10% for every arm (trivially true for a constant answer);
  we additionally report the max class probability on blanks (a confident blank answer = prior bias, reported as an issue).
- **Sanity:** siglip-probe ≥ siglip-zs; dit-ft ≥ 80% accuracy (published ~92%). If either fails we look for a pipeline bug.
Whatever the outcome, it is reported. No re-runs with changed prompts on the test set; any prompt change after seeing
results is labelled post-hoc and run separately.

## Layout
`code/` (prep, arms runner, scoring, figures) · `data/` (manifest + images) · `results/` (raw jsonl per arm, summary) ·
`figures/` · `issues-found/`. Results: see `RESULTS.md` (written after the run).

## Deviation log
- **2026-10-21 21:58 IST (after 4 d1 test items, all advertisement; no metric looked at for decisions):** d1-3B runs at
  ~50 s/image on this shared CPU (vision encoder + 3B prefill on 3 threads, load average ~27), so 640 images would take
  ~9 h. d1 now runs the **320-image subset (first 20 per class by id) first**, the same subset qwen3-vl-2b uses; the
  other 320 run afterwards if time allows. If d1 is incomplete, H1/H3/H4 are evaluated on the 320 subset with *all*
  arms restricted to the same items (stated in RESULTS). A Colab T4 notebook (`colab/clef-flash-t4.ipynb`) is prepared to re-run d1 (and clef-flash) on all 640; I did not run it.
- **2026-10-21 22:10 IST (17 d1 items done, still no metric inspected):** d1 measured ~55 s/image, qwen3-vl-2b ~40 s/image under load. To give every arm a matched core set overnight, CPU order is now: d1 and qwen3-vl-2b on **10 per class (160)** first, then 20 per class (320), then the rest. Primary comparisons use the largest set that all compared arms share; that n is stated next to every number. The Colab T4 notebook (`colab/`) is set up to run d1, qwen3-vl-2b and clef-flash on all 640 (not run).
- **2026-10-21 22:27 IST (run order only; no metric used for decisions — I saw only d1-omni's running accuracy in the auto-generated summary.md, which does not affect order):** the full-640 d1-omni run (188 done) was stopped and resumed later in queue A; d1-omni first completes the 10-per-class set, then qwen3-vl-2b runs its controls and 10-per-class set, so all six arms share the same 160 items as early as possible. `score.py` now also writes `results/verdicts.{json,md}` with the pre-registered rules applied to the largest set both compared arms share (n reported per row).
- **2026-10-21 23:10 IST (pipeline bug, before any hypothesis was scored):** train/cal labels and DiT outputs were in the official RVL-CDIP class order while test used alphabetical order; fixed by an exact remap (see `issues-found/05-label-order-bug.md`). No prompt, data or arm changed.

## Final result (reviewed 2026-10-22 03:50 IST)

The primary analysis is frozen to the balanced 10-per-class core (n=160; Qwen n=157 after three unresolved HTTP 500s). This prevents the resumable queue from silently changing the sample after the core completed.

| hypothesis | shared n | difference [95% CI] | verdict |
|---|---:|---:|---|
| H1 d1 − SigLIP zero-shot macro-F1 | 160 | +0.170 [0.090, 0.247] | **PASS** |
| H2 ECE(Qwen) − ECE(d1) | 157 | +0.162 [0.077, 0.222] | **PASS** |
| H3 d1 − SigLIP probe macro-F1 | 160 | −0.129 [−0.221, −0.055] | **FAIL** |
| H4 d1-omni − d1 macro-F1 | 160 | −0.558 [−0.619, −0.478] | **FAIL** |

Headline macro-F1: d1 0.606 [0.527, 0.664], Qwen 0.598 [0.524, 0.661], SigLIP zero-shot 0.436 [0.364, 0.499], SigLIP probe 0.735 [0.669, 0.796], DiT 0.919 [0.874, 0.957], and d1-omni 0.048 [0.023, 0.074].

See [`RESULTS.md`](RESULTS.md) for the full results, figures, limitations, and reproduction command, and [`results/REVIEW.md`](results/REVIEW.md) for the statistical and integrity audit. The review found and fixed two non-blocking reproducibility/reporting problems: stale train/cal label text plus an unfrozen live sample/retry denominator. Neither changed a verdict.
