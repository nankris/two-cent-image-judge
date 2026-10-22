# Statistical review — 2026-10-22 03:50 IST

## Decision

The evidence is conclusive under the preregistered rules, and no blocking bug remains. The corrected primary analysis is the frozen, balanced 160-item core (10 RVL-CDIP test images per class). H2 has n=157 because three Qwen requests ended in unresolved HTTP 500 errors; its 98.125% valid rate still passes the preregistered ≥98% threshold.

## Independent rerun and verdict check

`code/score.py` was rerun CPU-only with three BLAS/OpenMP threads; it loads stored JSONL only and did not load an Ollama or torch model. Bootstrap CIs use 2,000 class-stratified resamples with seed 0. The primary results are:

| hypothesis | shared n | preregistered comparison | difference [95% CI] | verdict |
|---|---:|---|---:|---|
| H1 | 160 | d1 − SigLIP zero-shot macro-F1; point ≥ +0.10 and CI lower > 0 | +0.170 [0.090, 0.247] | PASS |
| H2 | 157 | ECE(Qwen) − ECE(d1); CI lower > 0 | +0.162 [0.077, 0.222] | PASS |
| H3 | 160 | d1 − SigLIP probe macro-F1; noninferiority margin −0.05 | −0.129 [−0.221, −0.055] | FAIL |
| H4 | 160 | d1-omni − d1 macro-F1; noninferiority margin −0.10 | −0.558 [−0.619, −0.478] | FAIL |

The pre-review scorer used all live shared records and showed n=168 for H1/H3/H4. That was not the frozen balanced core. After correction, all four verdicts remain the same; issue 06 records the change.

## Label-order verification

- All 640 `siglip-probe` records and all 640 `dit-ft` records have `label_order_fixed: true`, complete 16-label probability vectors, and valid predictions.
- The exact permutation is official RVL-CDIP order → the experiment's alphabetical key order. DiT reaches 91.9% accuracy [87.5%, 95.6%] on the core, above the ≥80% sanity threshold. The probe reaches macro-F1 0.735 [0.669, 0.796], above zero-shot's 0.436 [0.364, 0.499].
- The review found stale train/cal label text in the manifest. Numeric `label_idx`—the field actually used to train/select the probe—was correct, so existing results were unaffected. Manifest metadata and the from-scratch prep/runner paths were fixed; see issue 05.

## Leakage and artifact integrity

The manifest contains 640 test, 800 train, and 400 calibration rows. Pixel-SHA256 intersections are zero for train↔test, calibration↔test, and train↔calibration. All 1,840 image files exist and their file hashes match the manifest. This rules out exact-pixel leakage; it does not rule out near-duplicates or common-source documents in RVL-CDIP.

## Failures, retries, and abstentions

| arm | valid core IDs | unresolved failures | validity | notes |
|---|---:|---:|---:|---|
| d1 | 160/160 | 0 | 100% | 11 historical HTTP 500 lines, all later retried successfully |
| d1-omni | 160/160 | 0 | 100% | no error lines |
| qwen3-vl-2b | 157/160 | 3 | 98.125% | three unresolved HTTP 500s; omitted pairwise, not counted correct |
| SigLIP zero-shot | 160/160 | 0 | 100% | — |
| SigLIP probe | 160/160 | 0 | 100% | — |
| DiT fine-tuned | 160/160 | 0 | 100% | — |

There are no malformed JSON lines, invalid predicted labels, incomplete/non-normalized probability vectors, or explicit abstentions. The three Qwen failures are honestly excluded from its paired n and disclosed. Treating them as wrong in an accuracy sensitivity check would only reduce Qwen's accuracy and does not affect H2's stored-confidence ECE comparison, which cannot score absent probabilities.

## Remaining limitations

The test is a small subset; macro-F1 intervals are wide. RVL-CDIP is an old scanned-document benchmark, not a deployment distribution. DiT was trained on the full RVL-CDIP training split and is a supervised in-domain ceiling. Latencies were measured on a heavily shared CPU box and are not clean model benchmarks. Blank controls reveal strong priors (especially Qwen and the probe) but have no true class, so “blank accuracy” is not meaningful. I chose not to run paid baselines (decided 23:20 IST); this is a free-model-only comparison.

