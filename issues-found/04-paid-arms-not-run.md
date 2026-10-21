# Paid arms: implemented, hard-gated, not run (0 paid calls)

`../_shared/vdec.py` refuses any paid request unless `ALLOW_PAID_CALLS=yes-approved-by-hari` and a key is set.
Cost estimates (list price x estimated image+text tokens, 711 requests = 640 test + 7 controls + 64 reversed) are in
`results/summary.json -> hosted_cost_estimate`. Nothing was billed.

At 23:20 IST on 2026-10-21, the user declined paid baselines and chose free models only. The estimates are not measurements and are not included as evaluated arms.
