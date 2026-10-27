# Findings — Rajkumar (Lane 2: Refresh / Content Opportunity Scoring)

All numbers below come from running the notebooks in `work/notebooks/` on the bundled starter CSV
(30,000 pages, 32 clients), seed 42. Receipts: `work/outputs/*.json`. Only the starter data was used; the gated
Hugging Face warehouse was not.

**Re-run:** `pip install -r requirements.txt && python scripts/run_all.py`, then execute each `work/notebooks/*.ipynb`.
Shared code is in `work/scripts/wk.py`.

## ML-02 Research question
Which existing pages should an editor review first? The label is `trend_direction == "down"`. Base rate is 0.542 overall
and 0.594 among pages with at least 1k impressions in 90 days.

## ML-03 Framing
This is a ranking task built on binary classification. The metric is Precision@50 on held-out clients, always shown
next to the base rate, with ROC AUC alongside. No single feature separates the classes well: single-feature AUCs range
from 0.41 to 0.58.

## ML-04 Data contract
- Grain is verified: one row per page.
- Rows per client range from 3 to 7,008 (median 567).
- `word_count` is missing on 7,699 rows (25.7%).
- In every row, last-30d plus prev-30d impressions are no more than the 90d impressions. So the 90-day features overlap
  the label window, and I list this as a limit.

## ML-05 Leakage check (optional)
I used the same random forest and the same client-holdout split each time.

| Feature set | AUC | P@50 |
|---|---|---|
| Safe features | 0.626 | 0.74 |
| Safe features + `trend_pct` | 1.000 | 1.00 |
| Safe features + 30-day windows | 0.895 | 1.00 |

Adding `trend_pct` or the 30-day windows leaks the label, so both are excluded.

## ML-06 Signal audit (optional)
- **Age: OPPOSITE to the naive rule.** Decline rate is 0.60–0.64 for pages up to 236 days old and 0.42 for pages older
  than 333 days.
- **Word count: MIXED.** The shortest quartile declines least (0.497). But among visible pages (at least 1k impressions),
  pages under 2,000 words decline more (0.768, n=849) than longer ones (0.583).
- **Competition level: FALSE as a separator here.** Decline rates are 0.556 (HIGH), 0.563 (LOW) and 0.567 (MEDIUM).
- **Position: MIXED.** Pages in positions 6–22 decline most (0.58–0.61).

## ML-07 Baseline
My hand rule weights age, staleness, visibility and a position 4–20 bonus. On the client-holdout test set it is **worse
than random**: AUC 0.463 and P@50 0.36, against a base rate of 0.511. In its top 20, 8 pages were not declining. This is
an honest negative result: the age assumption is backwards in this data, as the audit above shows.

## ML-08 Model
Same split for every model: 25 train clients and 7 test clients, with no overlap. The test set has 6,163 pages and a base
rate of 0.511.

| Model | AUC | AP | P@50 | P@200 |
|---|---|---|---|---|
| My baseline rule | 0.463 | 0.470 | 0.36 | 0.325 |
| Logistic regression | 0.631 | 0.629 | 0.82 | 0.780 |
| Random forest | 0.626 | 0.617 | 0.74 | 0.680 |
| HistGBM | 0.623 | 0.621 | 0.76 | 0.745 |

Top permutation importances for the random forest are `days_with_impressions`, `avg_position` and `content_age_days`.
The model ranks worse on high-visibility pages (AUC 0.581) than on low-visibility ones (AUC 0.651).

**Reference pipeline (`scripts/run_all.py`), re-run on this box:**
- Its random forest scored AUC 0.747 and P@50 0.68.
- Its built-in baseline scored AUC 0.627 and P@50 0.24.

That pipeline uses its own split and engineered features, so its numbers are not directly comparable to mine.

## ML-09 Validation audit
- **Split effect.** The same random forest scores AUC 0.769 and P@50 0.94 on a random row split, but AUC 0.626 and P@50
  0.74 on a client holdout. A row split overstates performance.
- **Paper Finding #1** ("growing content is longer"). In this starter sample the pooled median word count is 2,848 for
  growing pages and 2,909 for declining pages. Within clients, growing pages are longer in only 18 of 30 clients. So
  this is not reproduced in this sample; the paper used the full portfolio.
- **Claim rewrite.** "On held-out clients, the model's top-50 review list had a higher measured share of declining pages
  than my hand rule. It is decision-support for ordering reviews, not a forecast or a causal claim."

## ML-10 Action playbook
The queue counts are:
- monitor: 11,668
- review: 10,509
- refresh_and_review_ctr: 4,872
- refresh: 2,951

Pages with fewer than 100 impressions (7,994) should be excluded from action. There are human-review and no-go rules,
plus retrain triggers. The queue is written to `work/outputs/action_queue.csv` (gitignored).

**Caveat:** the reason codes come from my weak baseline rule. Better reason codes are a next step.

## ML-11 / ML-12 Capstone
The notebook sections are filled in. **Not done** (needs the user):
- the deployed static paper and its URL in `submission/paper_url.txt`
- the ML-12 demo outline, social post and employer summary
- portal card submissions
