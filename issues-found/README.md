# Issues found — The 2-Cent Image Judge

| file | group | status |
|---|---|---|
| [01-infrastructure-and-compute.md](01-infrastructure-and-compute.md) | ollama pull failure, shared CPU, subsets, clef-flash RAM | documented; Colab notebook prepared |
| [02-d1-omni-prior-bias.md](02-d1-omni-prior-bias.md) | d1-omni answers near chance, blank pages get confident labels | confirmed on primary core |
| [03-scoring-and-analysis.md](03-scoring-and-analysis.md) | unmatched sets / missing verdict code in first scorer | fixed |
| [04-paid-arms-not-run.md](04-paid-arms-not-run.md) | paid baselines gated, cost estimates only | declined; free models only |
| [05-label-order-bug.md](05-label-order-bug.md) | train/cal + DiT used official RVL-CDIP class order, test used alphabetical | fixed; reproduction path hardened |
| [06-frozen-core-and-validity-accounting.md](06-frozen-core-and-validity-accounting.md) | live-run sample drift and retry double-counting | fixed; verdicts unchanged |
