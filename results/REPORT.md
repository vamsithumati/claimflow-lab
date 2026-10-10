# ClaimFlow lab — validation report for post #1

**How to read this.** The lab is a simulation with known ground truth. Claims are tagged:
**BY CONSTRUCTION** — the assumptions in `config.py` produce them; the lab shows the mechanism is
coherent and its size *under those assumptions*. **HARNESS** — the comparison method is tested,
but magnitudes come from the assumptions. **MEASURED** — an estimator is graded against ground
truth it cannot see. Only HARNESS and MEASURED claims get a verdict.

## Claims

| ID | Claim | Evidence | Verdict | Detail | How the post should say it |
|---|---|---|---|---|---|
| C1a | When accepting requires a re-pick, the override rate misstates real corrections. | BY CONSTRUCTION | **—** | pick-to-accept, 20 seeds: override rate 14.9%, real corrections 9.4%; 37% of overrides changed nothing (range 33%–42%); re-picks alone were 24% of all overrides. Representative (median) run: seed 11. | Production observation; the lab only illustrates the mechanism. Quote the mean, not one run. |
| C1b | The share of overrides that aren't corrections depends on hold rate, bulk use and review habits. | BY CONSTRUCTION | **—** | Across 108 assumption cells: 12%–70%. | Measure yours. |
| C1c | Bulk 'set category' loses real corrections (stamps AI errors as human-confirmed). | BY CONSTRUCTION | **—** | Paired over 20 seeds: +1.11pp of documents corrected without bulk (t = 16.3). Direction is built into the bulk model; size depends on bulk usage. | Mechanism, not a measured effect size. |
| C2a | corrections ÷ all suggestions is a lower bound on true error (if reviewers are right). | MEASURED | **QUALIFIED** | Holds in 101/108 cells: reviewer accuracy 85–97%, wrong items opened 15–80%, correct items opened 2–40%, threshold 0.80/0.90, bulk use 0/20%. Tightest cell: gap -3.6pp (reviewers 85%, open-if-wrong 80%, open-if-right 40%). Default: 8.6% vs true 19.2%. | Lower bound if reviewers' corrections are right; can fail with weak reviewers at very high review coverage. |
| C2b | Reviewed-subset error is biased; it is not a bound. | BY CONSTRUCTION | **—** | Overstates in 106/108 cells, because every cell triggers review by low confidence or breakage. Review driven by value or random QA can understate. | Never call it an upper bound. |
| C2c | A blind stratified audit recovers the true error rate. | MEASURED | **QUALIFIED** | 98% labellers, double-labelled, reviewed items re-audited blind: bias -0.0 to +0.1pp across AI error rates of ~4/19/37%, coverage 95%–100%. 90% labellers: single label +3.8 to +9.4pp; second label on disagreement -4.4 to +0.6pp (sign flips with the base rate); double-label everything +0.8 to +2.4pp; gold-slice Rogan–Gladen correction -0.3 to +2.3pp, coverage 80%–88% (gold-slice uncertainty not propagated). If labellers drop to 75% on ambiguous look-alikes (errors correlated across labellers): double-labelled bias -0.3pp, coverage 92%; gold-corrected +0.1pp. | Recovers it with good blind labellers. With weaker labellers every shortcut is biased in a base-rate-dependent way: improve labellers and measure them against a gold slice. |
| C3a | A match-strength score is confidently wrong on look-alikes. | BY CONSTRUCTION | **—** | Twin errors at ≥0.9: match-strength 42%, margin 0% (by arithmetic: margin can't exceed 0.57 on look-alike families), classifier 1%. Within look-alike families (pooled) no score separates right from wrong much: AUROC match-strength 0.73, margin 0.76, classifier 0.78. | Don't claim margin 'ranks twins better' — it works by pushing whole look-alike families below the rest. |
| C3b | At equal workload, treating look-alike families differently beats one global threshold. | HARNESS | **QUALIFIED** | Errors passed automatically at 20/33/50% held, exactly in expectation (randomised gate at the boundary), mean of 20 seeds — match-strength, one global threshold: 11.0%/7.1%/4.2%; match-strength, threshold on look-alike families only: 10.5%/6.4%/3.1%; match-strength + look-alike evidence check: n/a/5.9%/3.4%; margin score, one global threshold: 9.0%/5.8%/2.0%; classifier, 300 labels: 12.6%/7.5%/2.9%; classifier, 3,000 labels: 8.6%/4.3%/0.9%. The ranking changes with the budget; magnitudes come from the generator. The rules are built from the generator's own vocabulary (an upper bound on hand-written rules); the classifiers are untuned. | Show all budgets; say the ranking depends on budget; disclose label budgets. |
| C4 | Joining overrides to suggestions by filename loses and misattributes ground truth. | HARNESS | **PASS** | Interleaved timeline, 10 seeds: 28% ambiguous; latest-draft-wins wrong 6% (grid 0%–26%). Magnitudes depend on the assumed filename collision rate; the direction holds whenever names collide. ID join: 100%. | Quote as a range; the size is an assumption. |

## Errors passed automatically at a review budget

Share of ALL documents that are wrong and pass automatically, mean ± sd over 20 seeds. **Exact** = a
randomised gate that holds a random fraction of items at the boundary score so every strategy holds exactly
the budget in expectation. **Deterministic** = best single threshold at or under the budget, with the hold rate it achieves.

| Strategy | 20% exact | ⅓ exact | 50% exact | ⅓ deterministic (hold achieved) |
|---|---|---|---|---|
| match-strength, one global threshold | 11.0% ± 0.4 | 7.1% ± 0.4 | 4.2% ± 0.3 | 9.2% (26% held) |
| match-strength, threshold on look-alike families only | 10.5% ± 0.4 | 6.4% ± 0.4 | 3.1% ± 0.2 | 8.5% (25% held) |
| match-strength + look-alike evidence check | not reachable (0/20 seeds) | 5.9% ± 0.2 | 3.4% ± 0.2 | 6.0% (33% held) |
| margin score, one global threshold | 9.0% ± 0.4 | 5.8% ± 0.2 | 2.0% ± 0.2 | 6.1% (32% held) |
| classifier, 300 labels | 12.6% ± 0.8 | 7.5% ± 0.6 | 2.9% ± 0.3 | 7.5% (33% held) |
| classifier, 3,000 labels | 8.6% ± 0.4 | 4.3% ± 0.3 | 0.9% ± 0.2 | 4.3% (33% held) |

Paired difference vs one global threshold (exact budget, mean ± 95% CI over 20 seeds):

| Strategy | 20% | ⅓ | 50% |
|---|---|---|---|
| match-strength, threshold on look-alike families only | -0.5pp ± 0.1 (-5%) | -0.7pp ± 0.1 (-10%) | -1.1pp ± 0.1 (-26%) |
| match-strength + look-alike evidence check | — | -1.2pp ± 0.1 (-17%) | -0.8pp ± 0.1 (-20%) |
| margin score, one global threshold | -2.0pp ± 0.1 (-18%) | -1.3pp ± 0.1 (-18%) | -2.2pp ± 0.2 (-53%) |
| classifier, 300 labels | +1.6pp ± 0.4 (+15%) | +0.4pp ± 0.3 (+6%) | -1.3pp ± 0.2 (-30%) |
| classifier, 3,000 labels | -2.4pp ± 0.2 (-22%) | -2.8pp ± 0.1 (-40%) | -3.3pp ± 0.1 (-78%) |

| Classifier | Error rate | ECE | Twin errors at ≥0.9 | AUROC, look-alike families pooled |
|---|---|---|---|---|
| prob_classifier_300 | 22.8% | 0.065 | 15% | 0.69 |
| keyword_router | 19.4% | 0.104 | 42% | 0.73 |
| prob_classifier | 19.5% | 0.036 | 1% | 0.78 |
| keyword_router_margin | 19.4% | 0.406 | 0% | 0.76 |

## Blind audit: bias by AI error rate, labeller quality and labelling scheme

Reviewed items are re-audited blind unless marked trusted. 10 seeds × 20 audits per row.

| AI error rate | True error | Labellers | Scheme | Reviewed items | Bias | CI coverage | ±CI |
|---|---|---|---|---|---|---|---|
| low (~4%) | 4.2% | 98% | none | re-audited | +1.7pp | 84% | ±2.6pp |
| low (~4%) | 4.2% | 98% | on_disagreement | re-audited | -0.0pp | 100% | ±2.1pp |
| low (~4%) | 4.2% | 98% | double_all | re-audited | +0.1pp | 100% | ±2.1pp |
| low (~4%) | 4.2% | 98% | gold_corrected | re-audited | +0.7pp | 96% | ±3.0pp |
| low (~4%) | 4.2% | 90% | none | re-audited | +9.4pp | 0% | ±3.8pp |
| low (~4%) | 4.2% | 90% | on_disagreement | re-audited | +0.6pp | 99% | ±2.3pp |
| low (~4%) | 4.2% | 90% | double_all | re-audited | +2.4pp | 66% | ±2.7pp |
| low (~4%) | 4.2% | 90% | gold_corrected | re-audited | +2.3pp | 87% | ±5.6pp |
| default (~19%) | 19.4% | 98% | none | re-audited | +1.3pp | 92% | ±3.9pp |
| default (~19%) | 19.4% | 98% | on_disagreement | re-audited | -0.5pp | 94% | ±3.7pp |
| default (~19%) | 19.4% | 98% | double_all | re-audited | -0.0pp | 97% | ±3.7pp |
| default (~19%) | 19.4% | 98% | gold_corrected | re-audited | -0.3pp | 95% | ±4.1pp |
| default (~19%) | 19.4% | 90% | none | re-audited | +6.6pp | 12% | ±4.4pp |
| default (~19%) | 19.4% | 90% | on_disagreement | re-audited | -1.5pp | 86% | ±3.7pp |
| default (~19%) | 19.4% | 90% | double_all | re-audited | +1.7pp | 90% | ±3.9pp |
| default (~19%) | 19.4% | 90% | gold_corrected | re-audited | +0.2pp | 88% | ±5.4pp |
| high (~37%) | 36.7% | 98% | none | re-audited | +0.8pp | 94% | ±4.7pp |
| high (~37%) | 36.7% | 98% | on_disagreement | re-audited | -1.3pp | 90% | ±4.6pp |
| high (~37%) | 36.7% | 98% | double_all | re-audited | +0.0pp | 95% | ±4.6pp |
| high (~37%) | 36.7% | 98% | gold_corrected | re-audited | +0.2pp | 91% | ±5.0pp |
| high (~37%) | 36.7% | 90% | none | re-audited | +3.8pp | 67% | ±4.8pp |
| high (~37%) | 36.7% | 90% | on_disagreement | re-audited | -4.4pp | 53% | ±4.6pp |
| high (~37%) | 36.7% | 90% | double_all | re-audited | +0.8pp | 96% | ±4.7pp |
| high (~37%) | 36.7% | 90% | gold_corrected | re-audited | -0.3pp | 80% | ±6.0pp |
| default (~19%) | 19.4% | 98% | double_all | trusted | -0.0pp | 96% | ±3.3pp |
| default (~19%) | 19.4% | 98%, 75% on ambiguous | none | re-audited | +0.7pp | 94% | ±3.8pp |
| default (~19%) | 19.4% | 98%, 75% on ambiguous | on_disagreement | re-audited | -4.4pp | 29% | ±3.4pp |
| default (~19%) | 19.4% | 98%, 75% on ambiguous | double_all | re-audited | -0.3pp | 92% | ±3.7pp |
| default (~19%) | 19.4% | 98%, 75% on ambiguous | gold_corrected | re-audited | +0.1pp | 88% | ±4.8pp |

| Audit weeks (20/bin/week, double-labelled) | ±CI | Coverage | Items labelled |
|---|---|---|---|
| 1 | ±8.5pp | 98% | 120 |
| 2 | ±5.4pp | 99% | 237 |
| 4 | ±3.7pp | 95% | 425 |
| 8 | ±2.4pp | 98% | 745 |

## Assumptions (defaults)

| Parameter | Value |
|---|---|
| `seed` | 7 |
| `n_docs` | 5000 |
| `n_train` | 3000 |
| `n_train_small` | 300 |
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
| `reviewer_twin_error_share` | 0.7 |
| `reviewer_anchor_share` | 0.5 |
| `p_bulk_batch` | 0.2 |
| `bulk_batch_size` | 8 |
| `review_lag` | 40 |
| `auto_open_delay` | (5, 300) |
| `audit_per_bin_per_week` | 20 |
| `audit_weeks` | 4 |
| `audit_labeller_accuracy` | 0.98 |
| `audit_labeller_twin_error_share` | 0.7 |
| `audit_labeller_ambiguous_accuracy` | None |
| `audit_bins` | (0.0, 0.6, 0.8, 0.9, 0.95, 1.0001) |

## Limitations

- **Synthetic data, simulated people.** Effect sizes depend on the assumptions table below.
- **The look-alike problem is generated, not discovered**: twins share vocabulary and lose their
  distinguishing evidence by design.
- **Unequal inputs**: the classifiers are trained on 300 or 3,000 labelled documents with untuned hyper-parameters;
  the routers get no labels but are built from the generator's own vocabulary — an upper bound on hand-written rules.
- **The twin check reads the generator's own distinguishing vocabulary** — a production evidence list
  would be less complete.
- **No drift**: audit "weeks" are drawn from one static population.
- **Labeller errors**: by default labellers err independently of each other and of how ambiguous a document is.
  One audit cell drops accuracy to 75% on ambiguous look-alikes for all labellers (correlated errors).
- **Gold slice**: 100 items, assumed labelled without error by an expert; rates are pooled across strata.
- **Margin works by family, not by ranking**: within look-alike families no score separates right from
  wrong much better than the others (AUROC table). The margin score helps because it pushes whole
  look-alike families below everything else — the same effect as a per-family threshold.
- **Audit interval**: stratified Wald-type with an Agresti–Coull plug-in variance; conservative at small
  samples. The gold-slice correction does not propagate the gold slice's own uncertainty.
- **The classifiers are deliberately simple.** An LLM classifier's confidence has its own behaviour
  and should be measured the same way.

## Reproduce

```bash
python -m claimflow            # all experiments
python -m unittest discover -s tests
```
