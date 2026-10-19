# DATA 4401 Project 2: project context

A complete rundown of this project from start to its current state, written so a
teammate (or their AI assistant) can pick up any part of the remaining work without
re-deriving anything. Every number here comes from an executed notebook in this repo.

Last updated: October 18, 2026 (presentation materials added).

---

## 1. The assignment

We are data scientists at a bank's mortgage division, predicting whether an applicant
will repay a loan, using the Kaggle **Home Credit Default Risk** data, primarily
`application_train.csv`. Requirements from the rubric:

- **Model:** a feed-forward, fully connected neural network, developed iteratively
  (train, check validation, inspect errors, adjust). Baselines like logistic regression
  or trees are allowed but optional.
- **Splits:** at minimum (1) a random split, (2) a stratified split, (3) a non-random
  split chosen so test/validation represents deployment data. Report how the split
  choice changes validation results and which one gives the most confidence about
  performance on new data. The split is *not* a hyperparameter.
- **Metrics:** ROC-AUC (Kaggle's metric) plus at least accuracy, F1 and one more. One
  single *optimizing* metric; optional *satisficing* metrics as constraints.
- **Error analysis:** manual, following Machine Learning Yearning, with eyeball and
  blackbox validation sets. Maintain a spreadsheet through the iterations and record
  whether decisions led to improvement.
- **Fairness:** evaluate fairness (e.g. gender, age), use fairness metrics as satisficing
  constraints if wanted, and discuss any conflict between fairness and accuracy.

### Deliverables and dates

| deliverable | due | status |
|---|---|---|
| progress review (lab) | Wed Oct 8 | done |
| error analysis spreadsheet (CSV ok if one sheet) | Fri Oct 16 | `mortgage_error_analysis_100.csv` |
| **presentation, 10 minutes, in class** | **Mon Oct 19** | **in progress** |
| written report | Tue Oct 20 | not started |
| code | Tue Oct 20 | done, in this repo |
| peer review survey | a few days after | required for a grade |

**Presentation audience:** a data science team at the client company. Technical detail
is expected about data prep, feature selection, model fitting and assessment, and the
insights from error analysis.

**Report sections required:** feature engineering and data prep; model selection
including error analysis and decisions based on it (plus a reflections section);
fairness criteria and how they were used in evaluation; evaluation of the final model;
ethical considerations.

---

## 2. Team, repo, files

- Repo: `github.com/rheachellani/DATA4401-Project2`. Branches: `main` (scaffold only),
  `owen/eda` (Owen: EDA, model run, this file), `rhea` (Rhea: splits and network),
  `codyEDA` (Cody: eval harness scaffold).
- The `owen/eda` branch contains everyone's notebooks and is the most complete.

| file | what it is |
|---|---|
| `Owen_EDA.ipynb` | Phases 1 to 8: cleaning, pruning, missing values, feature engineering, feature ranking, fairness baselines, preprocessing pipeline. 61 cells. |
| `Rhea_Splits.ipynb` | Owen's EDA plus Phase 9 (splits). Strict subset of the next file. |
| `Rhea_Splits_Model.ipynb` | Plus Phase 10 (PyTorch network, tuning) and Phase 11 (error analysis export). Uses data path `home-credit-default-risk/`. |
| `Cody_Eval.ipynb` | Empty evaluation harness scaffold (placeholders only). |
| **`Owen_Model_Run.ipynb`** | **The final, complete notebook.** Rhea's 110 cells (with fixes) plus Phases 12 to 15 and Conclusions. 126 cells, executed with all outputs, zero errors. **Use this as the source of truth.** |
| `mortgage_error_analysis_100.csv` | The error-analysis deliverable: 111 reviewed errors plus 5 iteration summary rows. |
| `data/` | Raw Kaggle CSVs and `data/processed/*.parquet`. Gitignored (2.5 GB), so not in the repo. |

### Running the notebook

- Python 3.14 venv at the workspace root, `torch` 2.14.1 (MPS on Apple Silicon works).
- Data folder is auto-detected: `data/` or `home-credit-default-risk/`.
- Full run takes about 10 to 15 minutes. Seeds are fixed (42); results reproduce exactly.

---

## 3. Data

- `application_train.csv`: 307,511 applicants x 122 columns, one row per applicant
  (`SK_ID_CURR` unique). `TARGET` 1 = payment difficulties (default), 0 = otherwise.
  **Default rate 8.07%** (imbalance about 1:11).
- `application_test.csv`: 48,744 applicants, same columns minus `TARGET` (Kaggle
  holdout, never labelled). Used only for the adversarial validation.
- Auxiliary tables used: `bureau.csv` (other lenders' credit history, covers 85.7% of
  applicants) and `installments_payments.csv` (repayment history on previous Home Credit
  loans, 94.8%). Not used: `bureau_balance`, `previous_application`, `POS_CASH_balance`,
  `credit_card_balance`.
- **No race or ethnicity column exists in the data.**

---

## 4. EDA and preprocessing (Owen_EDA.ipynb, Phases 1 to 8)

Phases ran in the order 1, 2, 4, 3, 7, 5, 6, 8 (pruning before imputing, engineering
before ranking). Columns: 122 raw -> 93 cleaned and engineered -> 199 after encoding.

**Phase 1, integrity.** No duplicate rows or IDs, train/test columns match, no orphan
keys. `installments_payments` is not unique per installment (743,407 duplicate keys:
partial or repeat payments), so it must be collapsed per installment before aggregating.

**Phase 2, repairs.**
- `DAYS_EMPLOYED == 365243` (about 1000 years) in 18% of rows is a code for "no
  employment record" (all pensioners/unemployed). Set to missing plus `FLAG_NOT_EMPLOYED`.
- Income log-transformed (`AMT_INCOME_LOG`); one row was 117 million vs 18 million next.
- Implausible counts winsorized at the 99.9th percentile (caps learned on train only).
- 6 placeholder rows dropped (`CODE_GENDER == 'XNA'`, `NAME_FAMILY_STATUS == 'Unknown'`).
- `DAYS_*` converted to positive years (`AGE_YEARS` etc.).

**Phase 4, pruning (measured, not assumed).** The 43 building-description columns
(`_AVG/_MODE/_MEDI` triplets) and 4 categorical building columns added nothing in a
3-seed tree ablation (all within seed noise, about 0.752 AUC), so they collapse to one
`HAS_BUILDING_INFO` flag. 18 near-constant columns dropped.

**Phase 3, missing values.**
- `AMT_REQ_CREDIT_BUREAU_*` is missing exactly when the applicant has no bureau file
  (0 of 41,518 appear in `bureau.csv`). Filled with 0 plus `FLAG_NO_BUREAU_FILE`; that
  group defaults at 10.34% vs 7.72%.
- `EXT_SOURCE` mean/min/max/count built before imputation.
- `OCCUPATION_TYPE` missing split into `Pensioner/Unemployed` vs `Unknown` (opposite risk).
- `OWN_CAR_AGE` = -1 for non-owners.

**Phase 7, features.** Six credit ratios (credit/income, annuity/income, credit/goods,
payment rate, credit term, income per person) and bureau aggregates over all history and
the last 12 months. Pearson correlation badly understated these: `CREDIT_TERM` had
correlation -0.032 but a 9.4-point default-rate spread across deciles;
`CREDIT_INCOME_RATIO` has an arch-shaped (non-linear) relationship.

**Phase 5, ranking with Information Value.** `EXT_SOURCE_MEAN` is by far the strongest
feature (IV 0.608). Past-delinquency features triple default risk but score low IV
because they affect under 1% of applicants, so **no IV cutoff was applied**. IV was only
used to pick one member of each near-duplicate pair (Rhea later re-did this per training
split to avoid leakage).

**Phase 6, fairness baselines.** Raw default rates: women 7.0%, men 10.1%; ages 20 to 25
at 12.3% falling to 4.9% for 60 to 70. **Gender can be reconstructed from the other
features at AUC 0.906, age 50+ at 0.970**, so removing a column does not make the model
blind to it.

**Phase 8, pipeline.** scikit-learn `ColumnTransformer`, fit on training rows only:
61 continuous (median impute, standardize), 12 categorical (mode impute, one-hot),
18 binary flags passed through unscaled.

---

## 5. Splits and network (Rhea, Phases 9 to 11)

**Phase 9, three splits**, each 70/15/15 train/validation/test:
- random; stratified on `TARGET`;
- **deployment-like**: a classifier trained to tell training applicants from Kaggle test
  applicants reached **AUC 0.905** (adversarial validation). The 30% of training
  applicants that look most like the Kaggle applicants became validation + test.

**What the 0.905 means:** future applicants differ from past ones. No single feature
explains it (the best alone reaches 0.67); it is many moderate differences, not a data
error:

| | training | Kaggle test |
|---|---|---|
| revolving loans | 9.5% | 0.9% |
| median number of payments | 20 | 16.3 |
| median loan / income | 3.27 | 2.67 |
| EXT_SOURCE_1 missing | 56.4% | 42.1% |

**Phase 10, network.** PyTorch MLP, BCE loss, Adam, best epoch by validation ROC-AUC.
Tuning on the stratified split ran out of room (total spread about 0.004):

| experiment | validation ROC-AUC |
|---|---|
| baseline 128-64 | 0.7599 |
| best regularization (dropout 0.2 + L2 1e-4) | 0.7629 |
| class weights (pos_weight 1 to 11.4) | no gain, best was 1.0 |
| **best architecture: 128-64-32**, dropout 0.2, L2 1e-4, early stopping | **0.7636** |

Decision threshold **0.15**, the best-F1 cutoff on validation (0.5 flags almost nobody
at 8% prevalence). Known caveat: it was chosen on the full validation set, which
includes the blackbox rows, so blackbox F1 at 0.15 is slightly optimistic.

**Optimizing metric: ROC-AUC. Satisficing: fairness** (see Phase 13).

---

## 6. Error analysis (Phase 11)

- Stratified validation set: 46,126 rows, error rate 16.16% at threshold 0.15.
- **Eyeball set: 619 rows**, sized to contain about 100 errors (Machine Learning
  Yearning), giving **111 errors (80 false positives, 31 false negatives)**, all reviewed.
- **Blackbox set: the other 45,507 rows**, never inspected; the only place a change
  prompted by the review may be measured.

Results of the manual review:

| category | FN | FP | total |
|---|---|---|---|
| external scores misled the model | 21 | 39 | **60 (54%)** |
| borderline threshold calls | 2 | 19 | 21 |
| limited credit information | 3 | 15 | 18 |
| other / mixed signals | 1 | 6 | 7 |
| elevated loan burden | 4 | 1 | 5 |

Interpretation: the model over-relies on the external credit scores. Good scores hid
two thirds of missed defaulters; poor scores caused half the false alarms.

Known issue with the CSV: the `notes`/`possible_reason` text is templated (10 distinct
reasons, 6 distinct improvements, one note format), so it reads as generated rather
than hand-written.

---

## 7. Iteration 2 and fairness (Owen, Phases 12 to 14)

**The change: repayment history** from `installments_payments.csv`, 13 new columns
(share of installments late, mean and max days late, share underpaid, paid/owed ratio,
count, each for all history and the last 12 months, plus `FLAG_NO_INSTALLMENT_HISTORY`).

Evidence it targets the error categories, checked before building:
- Among 172,554 applicants with strong external scores: >10% of installments late ->
  5.61% default vs 3.42%; >5% underpaid -> 11.88% vs 3.91%.
- 94.6% of applicants with no bureau file do have installment history.

Results, all on the same 45,507 blackbox rows:

| model | inputs | blackbox ROC-AUC | PR-AUC |
|---|---|---|---|
| iteration 1 | 199 | 0.7635 | 0.2474 |
| iteration 2, + repayment history | 212 | **0.7713** | 0.2554 |
| **iteration 2, gender + family status withheld (FINAL)** | 206 | 0.7684 | **0.2567** |
| iteration 2, also withholding age (rejected) | 205 | 0.7661 | 0.2499 |
| no external scores (diagnostic ablation) | 203 | 0.7348 | 0.2194 |

- Improvement: **+0.0077, 95% bootstrap interval +0.0047 to +0.0107**, iteration 2
  better in 100% of 500 resamples.
- The ablation confirms the review: removing the external scores costs 0.0365.

**Fairness audit** (threshold 0.15). Measures per attribute: approval ratio (least
approved group's rate / most approved; below 0.80 fails the "four-fifths" screen), TPR
gap (equal opportunity, our primary fairness metric), FPR gap (good customers wrongly
flagged, the direct harm), AUC gap.

- **Final model withholds gender and family status** (protected under ECOA/Reg B):
  costs 0.0029 AUC; gender approval ratio 0.891 -> 0.916; gender TPR gap 0.143 -> 0.096
  (-33%); FPR gap 0.082 -> 0.060.
- **Age fails** (ratio about 0.67). Ages 20 to 25 have roughly 3 in 10 good applicants
  wrongly flagged. Partly real (11.5% vs 4.9% default), but the model is also less
  accurate at both ends (AUC about 0.72 vs 0.77 middle-aged). **Removing age made it
  worse** (ratio 0.674 -> 0.657, AUC -0.0023) because age is rebuilt from proxies.
  Per-group thresholds would fix it but mean treating applicants differently by a
  protected attribute, generally unlawful in US lending.
- **Region rating fails** (about 0.79 blackbox, 0.77 test); relevant because geography
  is how redlining operated, and race itself cannot be measured.

---

## 8. Final evaluation (Phase 15)

Held-out test rows scored once, nothing changed afterwards. Final setup trained on each
split; threshold 0.15.

| split | validation ROC-AUC | **test ROC-AUC** | test PR-AUC | test F1 | test recall |
|---|---|---|---|---|---|
| random | 0.7674 | 0.7674 | 0.2533 | 0.3165 | 0.4345 |
| **stratified (primary)** | 0.7689 | **0.7698** | **0.2600** | **0.3181** | **0.4705** |
| deployment-like | 0.7636 | **0.7657** | 0.2372 | 0.3017 | 0.4731 |

Stratified test also: accuracy 0.837, precision 0.240. It catches nearly half of
defaulters; about 1 in 4 flagged applicants actually defaults. Accuracy looks high only
because 92% repay.

**Split conclusion:** random and stratified give near-identical, slightly optimistic
results (307k rows means a random split already lands near 8%). Deployment-like is
consistently lowest and is the most trustworthy for new applicants. **Realistic
performance estimate: about 0.766.** Test landed close to validation everywhere, so no
overfitting to validation.

**Fairness on the test sets (approval ratio):**

| attribute | stratified test | deployment-like test |
|---|---|---|
| gender | 0.908 pass | 0.909 pass |
| age band | **0.675 fail** | **0.608 fail** |
| family status | 0.858 pass | 0.848 pass |
| children | 0.945 pass | 0.934 pass |
| education | 0.844 pass | 0.813 pass |
| region rating | **0.772 fail** | **0.753 fail** |

Age disparity is **worse** on the deployment-like population, i.e. for the applicants
the model would actually see.

**Bottom line for the report and slides:** more accurate than where we started, gender
disparity reduced at almost no cost, but **not deployable as-is**: age and region need a
compliance review, and race cannot be tested without collecting that data under fair
lending rules.

---

## 9. Talking points that land well

- `EXT_SOURCE_MEAN` (averaging whichever external scores exist) became the best feature
  in the dataset and removed most of the missing-data problem.
- Measured pruning: 43 columns deleted because a controlled experiment showed they did
  nothing.
- Pearson vs Information Value: correlation said `CREDIT_TERM` was useless; it was the
  strongest engineered feature.
- IV hides rare-but-severe signals (delinquency triples risk yet scores "useless").
- Model tuning plateaued (about 0.004 total); the gain came from data the model lacked,
  found by error analysis. That is the core iterative story.
- Fairness through unawareness fails (gender recoverable at 0.906). Removing age made age
  disparity worse.
- Adversarial validation: 0.905 shift, explained by loan-product mix.

## 10. Writing conventions for this project

- Plain language, explain jargon on first use (the audience for writeups is assumed
  non-specialist; slides can be more technical).
- **No em dashes** in written deliverables.

## 11. Presentation (agreed structure, in progress)

Four presenters, ten slides, ten minutes. Everything needed to build the deck is in
`presentation/`:

- `presentation/slide_content.md`: bullets for every slide, timings, speaker notes, and
  the presenter split (1: slides 1-3, 2: slides 4-5, 3: slides 6-7, 4: slides 8-10)
- `presentation/figures/`: 12 PNG charts named by slide (e.g.
  `slide08a_approval_ratio_four_fifths.png`)
- `presentation/make_figures.py`: regenerates every figure. Data-derived charts read the
  raw CSVs and processed parquet; model results are copied from the executed notebook
  with the source cell noted

Chart style: light surface, blue `#2a78d6` and orange `#eb6834` (validated for
colorblind safety), gray for de-emphasis, claim-style titles.
