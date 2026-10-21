# Label-order bug in the trained baselines (found 23:07 IST, fixed before any hypothesis was scored)

**What.** The two Hugging Face sources number the 16 classes differently:
- `chainyo/rvl-cdip` (our **test** set): alphabetical (0=advertisement, 1=budget, ...) — matches `code/labels.py`.
- `jordyvl/rvl_cdip_100_examples_per_class` (probe **train/cal**) and `microsoft/dit-base-finetuned-rvlcdip`: the official
  RVL-CDIP order (0=letter, 1=form, 2=email, ...).
`prep_data.py` named the train/cal images with the alphabetical list, and `run_torch_arms.py` stored DiT's probabilities under
the alphabetical names. Both are a fixed permutation, so siglip-probe and dit-ft answers came out under the wrong class names.

**How it was found.** While DiT loaded, its printed `id2label` began `{0: 'letter', 1: 'form', ...}`; the dataset cards confirmed
the two orders. The pre-registered sanity check (probe >= zero-shot, DiT >= 80%) would also have flagged it.

**Impact.** Test labels and every Ollama arm (d1, d1-omni, qwen3-vl-2b) and SigLIP zero-shot were never affected.

**Fix (exact, no re-inference).** `code/fix_label_order.py` re-keyed the already-written
siglip-probe and dit-ft probabilities (records carry `label_order_fixed: true`). `run_torch_arms.py` now maps DiT by class name.
After the fix the probe's accuracy is 76.2% vs 53.7% for zero-shot on the 640 test images (before: near chance).

**Review correction (03:45 IST, 2026-10-22).** The earlier note incorrectly said the manifest's train/cal label text had already been remapped. It had not: `label_idx` remained the correct official-order training target, so the fitted probe and repaired test probabilities were valid, but the unused label text was stale. The manifest metadata is now corrected (pre-review copy: `data/manifest.csv.pre-review-20261022-0345`), and `prep_data.py` plus the probe output path in `run_torch_arms.py` now handle the official order directly. All 640 existing probe and 640 DiT records have `label_order_fixed: true`; no inference was repeated.
