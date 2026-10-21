# Scoring and analysis issues found and fixed

- The first `score.py` computed per-arm metrics on whatever items each arm had finished, and did not apply the pre-registered
  decision rules. Fixed (22:27 IST): `results/verdicts.{json,md}` applies H1-H4 on the largest set *both* compared arms share,
  with the n in every row; verdicts with fewer than 32 shared items are reported as PENDING.
- Macro-F1 over 16 classes on a 160-item subset (10 per class) has wide intervals; the CI is the honest number, not the point.
- d1 option-order control uses 64 items (4 per class); if d1 has not run those items in the original order, the agreement is
  computed only on the overlap (n reported).
- Review at 03:30 IST found that “largest currently shared set” allowed queue progress to move H1/H3/H4 from the balanced n=160 core to n=168. The primary scorer is now frozen to first 10/class. See issue 06.
