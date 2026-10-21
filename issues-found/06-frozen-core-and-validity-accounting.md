# Live-run drift and retry accounting in the scorer

## Symptom

At 03:30 IST on 2026-10-22, `score.py` reported shared n=168 for H1/H3/H4 although the completed primary design was the balanced 160-item core (10 per class). It also reported d1 validity as 168/(168+11)=93.9%, below the preregistered 98% threshold.

## Evidence and root cause

- Queue C had continued after completing the core, so d1 had eight additional valid records and d1-omni had 140 additional records. The scorer used every currently available shared ID; results could therefore drift while the queue ran.
- The 11 d1 error lines are historical HTTP 500s. Every one has a later valid record for the same ID. Counting records rather than unique target IDs double-counted those retry attempts.
- Qwen has 157 valid core records and three unresolved HTTP 500s, for 157/160 = 98.125%. There are no malformed JSON lines, invalid labels, incomplete probability vectors, or explicit abstentions in any arm.

## Fix and status

Fixed in `code/score.py` at 03:40 IST: the primary analysis is frozen to the first 10 IDs per class, all comparisons use that core, and validity is valid unique core IDs / 160. Historical error-line counts remain visible separately. The corrected verdicts are unchanged: H1 PASS, H2 PASS, H3 FAIL, H4 FAIL. This was a reporting/reproducibility issue, not a blocking inference bug.

