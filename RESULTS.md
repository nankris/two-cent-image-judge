# Results

## Question and setup

Can small open image decision models classify real scanned business documents well enough—and with calibrated enough confidence—to replace a chat VLM or a trained classifier?

The primary set is 160 RVL-CDIP test images: the first 10 preregistered sampled IDs in each of 16 balanced classes. Every arm uses the same stored images. Qwen has 157 valid responses after three unresolved HTTP 500s; every other arm has all 160. CIs are 95% percentile bootstrap intervals from 2,000 class-stratified resamples (seed 0).

## Final numbers

| arm | n | accuracy [95% CI] | macro-F1 [95% CI] | ECE [95% CI] |
|---|---:|---:|---:|---:|
| d1 3B | 160 | 58.8% [51.9, 65.6] | 0.606 [0.527, 0.664] | 0.154 [0.115, 0.228] |
| d1-omni 600M | 160 | 10.0% [6.9, 13.8] | 0.048 [0.023, 0.074] | 0.294 [0.257, 0.335] |
| Qwen3-VL 2B | 157 | 61.1% [54.8, 67.5] | 0.598 [0.524, 0.661] | 0.311 [0.258, 0.375] |
| SigLIP2 zero-shot | 160 | 46.2% [40.0, 52.5] | 0.436 [0.364, 0.499] | 0.075 [0.074, 0.163] |
| SigLIP2 linear probe | 160 | 74.4% [68.8, 80.6] | 0.735 [0.669, 0.796] | 0.096 [0.083, 0.168] |
| DiT fine-tuned | 160 | 91.9% [87.5, 95.6] | 0.919 [0.874, 0.957] | 0.101 [0.085, 0.141] |

## Preregistered verdicts

| hypothesis | shared n | difference [95% CI] | verdict |
|---|---:|---:|---|
| H1: d1 beats SigLIP zero-shot by ≥10 macro-F1 points | 160 | +0.170 [0.090, 0.247] | **PASS** |
| H2: d1 ECE is lower than Qwen ECE | 157 | +0.162 [0.077, 0.222] for Qwen − d1 | **PASS** |
| H3: d1 is within 5 macro-F1 points of the trained SigLIP probe | 160 | −0.129 [−0.221, −0.055] | **FAIL** |
| H4: d1-omni is within 10 macro-F1 points of d1 | 160 | −0.558 [−0.619, −0.478] | **FAIL** |

The simple trained probe beats d1 by 12.9 macro-F1 points, and the in-domain fine-tuned DiT is far ahead. The 600M d1-omni model is near chance. See `results/REVIEW.md` for the full integrity and failure audit.

## Figures

- `figures/reliability.png` — per-arm reliability curves on the matched core
- `figures/cost_vs_f1.png` — local shared-CPU cost proxy versus macro-F1
- `figures/confusion_d1.png` — d1 confusion matrix on the matched core

## Limitations

- This is a balanced n=160 subset of one old scanned-document benchmark. Intervals are wide, and external validity is limited.
- DiT is an in-domain supervised ceiling trained on RVL-CDIP; it is not a general zero-shot model.
- The probe uses 800 training and 400 calibration images. d1 and Qwen do not.
- Exact pixel-hash overlap is zero, but near-duplicate/common-source leakage was not tested.
- Qwen has three missing responses. Its 98.125% valid rate passes the preregistered threshold; comparisons use the 157 available pairs.
- Shared-box CPU latency and derived VM cost are workload-contaminated estimates, not clean benchmarks.
- Blank controls expose label priors and overconfidence, especially for Qwen and the probe.
- The user declined paid baselines at 23:20 IST on 2026-10-21. Results are free-model-only; hosted prices in generated JSON are estimates, not measured paid runs.

## Reproduce

From the repository root, after obtaining the named Hugging Face/Ollama models and preparing data:

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  python code/score.py
```

To rebuild model outputs from scratch, run `code/prep_data.py`, `code/run_torch_arms.py`, and the Ollama runner as documented in the source. The one-off `code/fix_label_order.py` is retained for old pre-fix JSONL only; new runs map labels correctly at write time.
