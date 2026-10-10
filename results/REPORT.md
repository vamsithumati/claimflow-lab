# ClaimFlow lab — validation report for post #1

Claims from *“Your AI's Override Rate Is Lying to You”*, tested against synthetic insurance-claims data
where the true label of every document is known.

## Verdicts

| ID | Claim | Verdict | Evidence | Change to the post |
|---|---|---|---|---|
| C1a | Naive override rate overstates the true correction rate when accepting requires a re-pick. | **PASS** | pick-to-accept: override rate 15.5% vs true corrections 10.2% (20 seeds: confirm-as-is 36.3% ± 2.4%). | — |
| C1b | 'A third to half of overrides are confirm-as-is.' | **QUALIFIED** | Lab default 34.4%; across the sensitivity grid 19%–76%. The share depends on hold rate and how often correct items are opened. | Keep 'a third to half' as a production observation; add that the size depends on hold rate and review habits. |
| C1c | One-click accept removes confirm-as-is noise. | **PASS** | one-click-accept: confirm-as-is 0 of 550 overrides. | — |
| C1d | NEW — bulk 'set category' hides real errors, not just adds noise. | **PASS** | true corrections captured (mean over seeds): one-click 10.7% vs pick-to-accept with bulk-set 9.7% — bulk-set stamped wrong labels as human-confirmed. | Add: bulk-set doesn't just inflate overrides, it stamps wrong labels as human-confirmed. |
| C2a | The reviewed subset overstates error (people open what looks wrong). | **PASS** | reviewed-subset error 54.8% vs true error 20.1%. | — |
| C2b | corrections ÷ all drafts is a lower bound. | **PASS** | default 10.2% ≤ true 20.1%; holds in all 54 sensitivity cells. | — |
| C2c | A blind stratified audit (20 per bin per week) recovers the true error. | **PASS** | with adjudication: 95% CI covers truth 95% of runs, bias +0.0pp, ±3.1pp after 4 weeks. Single-label audit: bias +1.8pp, coverage 87%. | Add: audit labeller mistakes bias the estimate upward — send disagreements to a second blind labeller. |
| C3a | On look-alike twins the classifier is confidently wrong. | **QUALIFIED** | match-strength router: 41% of twin errors at ≥0.9; calibrated classifier: 1%. Twin pairs are 88% of errors in both. | Rewrite: true when confidence is a match-strength score; a calibrated or margin-based score sees twins. |
| C3b | Raising the global threshold is an expensive way to catch twin errors. | **QUALIFIED** | Auto-passed error at a 33% human-review budget: router: raise threshold 7.4%; router: + twin verifier 6.1%; router: margin confidence 5.0%; calibrated classifier 4.5%. | Rewrite: a threshold only works if confidence is computed against the runner-up. |
| C3c | A twin-aware verification step beats raising the threshold. | **PASS** | at 33% review budget: verifier 6.1% vs threshold 7.4%; but margin confidence (5.0%) beats both. | Lead with 'fix the confidence first', verification second. |
| C4 | Joining overrides to drafts by filename loses and misattributes ground truth. | **PASS** | default: 32% ambiguous, latest-draft-wins picks the wrong draft 26% of the time; up to 55% ambiguous at 60% generic names / 1000 docs per tenant. ID join: 100%. | Add: the 'latest draft wins' fallback is worse than ambiguity — it is silently wrong. |

## Errors that pass automatically, at a fixed human-review budget

| Strategy | 20% held | 33% held | 50% held |
|---|---|---|---|
| router: raise threshold | 11.5% | 7.4% | 4.3% |
| router: + twin verifier | not reachable¹ | 6.1% | 3.4% |
| router: margin confidence | not reachable¹ | 5.0% | 1.3% |
| calibrated classifier | 8.6% | 4.5% | 1.0% |

¹ This strategy always holds more than this share — e.g. every unverifiable twin goes to a human.

## Audit precision by weeks of sampling (20 docs per confidence bin per week)

| Weeks | 95% CI half-width | CI coverage |
|---|---|---|
| 1 | ±6.3pp | 94% |
| 2 | ±4.5pp | 93% |
| 4 | ±3.1pp | 96% |
| 8 | ±2.1pp | 97% |

## Assumptions (defaults)

| Parameter | Value |
|---|---|
| `seed` | 7 |
| `n_docs` | 5000 |
| `n_train` | 3000 |
| `n_tenants` | 20 |
| `p_distinct_twin` | 0.7 |
| `p_distinct_other` | 0.95 |
| `p_cross_token` | 0.3 |
| `p_garbled` | 0.04 |
| `generic_filename_share` | 0.45 |
| `threshold` | 0.85 |
| `p_open_if_wrong` | 0.35 |
| `p_open_if_right` | 0.05 |
| `reviewer_accuracy` | 0.97 |
| `p_bulk_batch` | 0.2 |
| `bulk_batch_size` | 8 |
| `audit_per_bin_per_week` | 20 |
| `audit_weeks` | 4 |
| `audit_labeller_accuracy` | 0.98 |
| `audit_bins` | (0.0, 0.6, 0.8, 0.9, 0.95, 1.0001) |

## Limitations

- **Synthetic data, simulated people.** The lab validates *mechanisms and estimators* — that an effect
  exists, which way it points and what drives it. It does not tell you how big the effect is in your
  system. Every rate here moves with the assumptions in the table above (see the sensitivity grid).
- **The classifiers are deliberately simple** (keyword router, numpy logistic regression). An LLM
  classifier's confidence (e.g. verbalised confidence or token log-probs) has its own calibration
  behaviour and should be measured the same way before trusting a threshold.
- **The audit assumes blind labellers** who never see the AI's answer. A non-blind audit inherits anchoring bias.
- **Ground truth (`oracle_truth`) exists only in the lab.** Production systems must estimate it — which
  is what the audit is for.

## Reproduce

```bash
python -m claimflow            # all experiments, ~2 min
python -m unittest discover -s tests
```
