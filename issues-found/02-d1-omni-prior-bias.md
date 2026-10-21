# d1-omni-600M answers largely from a prior, not from the page

- On the first 147 test items (advertisement, budget, email, file_folder) d1-omni was right 4.8% of the time, below the 6.25%
  chance level. It put most images into `publication` (61) or `letter` (55).
- Blank controls: white/black/gray pages -> `news_article` at 0.23-0.36, noise -> `publication` at up to 0.54.
  Label-only (no image) -> `letter` 0.20.
- Each 512-px image costs only ~192 input tokens for d1-omni (414 with image vs 222 without), so the model sees very little of a
  scanned page. This is a likely cause but is not tested here (any resolution change would be a post-hoc, separately labelled run).
- **Final primary core (10/class, n=160):** 10.0% accuracy [6.9%, 13.8%], macro-F1 0.048 [0.023, 0.074]. H4 failed by a wide margin: d1-omni minus d1 macro-F1 = -0.558 [-0.619, -0.478].
