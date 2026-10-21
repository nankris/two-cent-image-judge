# Infrastructure and compute limits

- **`ollama pull` fails on this box.** The registry redirects blob downloads to an R2 host that resolves to 198.18.0.1, which
  Ollama 0.40.7 rejects. Models were pulled manually with `../_tools/ollama_manual_pull.sh` (sha256-verified per blob).
  Anyone reproducing on a normal network can just `ollama pull`.
- **CPU-only and shared.** 8 vCPU / 15 GB RAM, no GPU, shared with another worker's jobs (load average 25-27 on 8 cores
  during the runs, logged per call as `loadavg`). Latency numbers are therefore *upper bounds*, not benchmarks of the models.
  d1:3b took ~55-70 s per image, qwen3-vl-2b ~40 s, d1-omni ~13-15 s.
- **Consequence: subsets.** The pre-registered n=640 for d1 and qwen3-vl-2b could not run on CPU in one night. Deviation log
  (README): 10 per class (160) first, then 20 per class, then the rest. Every comparison states its n.
- **clef-flash:9b does not fit** (11 GB tag next to ~6 GB of other jobs). Covered by the Colab T4 notebook
  `/workspace/colab-jobs/vision-clef/` (prepared, not run).
- **RAM pressure.** ~0.6-1 GB available while two llama-servers and two foreign evaluation jobs run; no extra model
  processes were started while the queues ran.
- **Ollama answers are not bit-reproducible under concurrent load.** The same d1-omni request (same pixels, state and options)
  replayed through `colab_run.py` at 22:35 IST gave P(letter)=0.357 vs 0.339 in the main run (same prediction). The server runs 16
  parallel slots with a unified KV cache, so batch composition changes the floating-point path. Differences are small; any
  repeated item is reported with its agreement (exp 4 `determinism_clean_vs_source`).
