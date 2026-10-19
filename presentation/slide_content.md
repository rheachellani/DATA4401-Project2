# Presentation: slide content

10 minutes, 10 slides, 4 presenters. Audience: a data science team at the client's
company, so technical detail is expected on data prep, feature selection, model fitting
and assessment, and the error analysis.

Figures are in `presentation/figures/`, named by slide. Regenerate them with
`python presentation/make_figures.py` from the project2 folder. Every number below comes
from the executed `Owen_Model_Run.ipynb` or `Owen_EDA.ipynb`; see `PROJECT_CONTEXT.md`
for the full story.

## Presenter split

| presenter | slides | time | why this split |
|---|---|---|---|
| 1 | 1 to 3: intro, data, data prep | 2:30 | EDA work |
| 2 | 4 to 5: splits, network and tuning | 2:00 | Rhea's splits and network |
| 3 | 6 to 7: error analysis, iteration 2 | 2:15 | the error analysis loop |
| 4 | 8 to 10: fairness, final results, conclusions | 3:15 | evaluation and ethics |

Presenter 4 carries the most time. If that is too much, presenter 3 can take slide 9.

---

## Slide 1: Predicting mortgage default responsibly (0:15) · Presenter 1

- Team names
- One line: a neural network that predicts loan repayment, improved through error
  analysis and checked for fairness

*No figure.*

---

## Slide 2: The problem and the data (0:45) · Presenter 1

- Home Credit Default Risk: 307,511 past applicants, 122 columns, plus credit bureau
  history and past repayment records
- Target: payment difficulties. Only **8.07%** default, roughly 1 in 12
- That imbalance shapes everything: accuracy is misleading (92% from predicting "repays"
  for everyone)
- **Optimizing metric: ROC-AUC**, how well the model ranks risky applicants above safe
  ones. **Satisficing: fairness**
- Also reported: PR-AUC, accuracy, F1, precision, recall

*No figure; put **8.07%** large on the slide.*

---

## Slide 3: Data preparation (1:30) · Presenter 1

- **Fake values fixed:** employment length of 365,243 days (1000 years) on 18% of rows
  was a code for "not employed", turned into a flag
- **Pruning was measured, not guessed:** 43 building-description columns removed
  after a controlled test showed they changed nothing
- **Missing data can be informative:** blank credit-check counts meant "no credit file at
  all", a group that defaults at 10.3% against 7.7%
- **Best feature was one we built:** the average of the three external credit scores
  (Information Value 0.608)
- **Correlation misleads:** loan-to-income risk rises then falls, so correlation reads
  near zero; we ranked features by Information Value instead

**Figures:** `slide03a_credit_income_ratio_arch.png` (lead with this),
`slide03b_top_features_information_value.png`. Optional:
`slide03c_building_columns_ablation.png`.

*Speaker note:* five findings in 90 seconds is tight. If short on time, keep the arch
chart and the best-feature point and mention the rest in one line each.

---

## Slide 4: Three ways to split the data (1:00) · Presenter 2

- All three: 70% train, 15% validation, 15% test
- **Random** and **stratified** (keeps the 8% default rate equal in every part)
- **Deployment-like:** trained a classifier to tell our applicants from Kaggle's test
  applicants. It succeeded with **AUC 0.905**, so future applicants differ
- Reserved the 30% of our applicants who look most like Kaggle's as validation and test
- The shift is real, not a data error: fewer revolving loans (0.9% vs 9.5%), shorter
  terms, smaller loans relative to income

**Figure:** `slide04_train_vs_kaggle_shift.png`

---

## Slide 5: The network and tuning (1:00) · Presenter 2

- PyTorch feed-forward network: 199 inputs, hidden layers 128, 64, 32, dropout 0.2,
  L2 1e-4, early stopping on validation ROC-AUC
- Inputs prepared in a pipeline fit on training rows only (median fill, scaling,
  one-hot encoding), refit for every split to avoid leakage
- **13 tuning runs** (regularization, class weights, architectures) moved AUC by only
  **0.004**, from 0.7599 to 0.7636
- Decision threshold 0.15 (best F1), since 0.5 flags almost nobody at 8% prevalence
- Takeaway: the model had stopped improving, so the gain had to come from somewhere else

**Figure:** `slide05_tuning_plateau.png`

---

## Slide 6: Error analysis (1:30) · Presenter 3

- Machine Learning Yearning setup: split validation into an **eyeball set** to inspect
  and a **blackbox set** to judge changes
- Eyeball set sized to hold about 100 errors: 619 applicants, **111 errors** (80 false
  alarms, 31 missed defaults), every one reviewed by hand
- Blackbox: 45,507 applicants never inspected
- **54% of errors trace to the external credit scores:** good scores hid 21 of 31 missed
  defaults, poor scores caused 39 of 80 false alarms
- Next biggest: borderline threshold calls (21) and limited credit history (18)

**Figures:** `slide06a_eyeball_blackbox_split.png`, `slide06b_error_categories.png`

---

## Slide 7: Iteration 2, acting on the errors (0:45) · Presenter 3

- The model needed evidence the external scores do not carry: **how applicants actually
  repaid previous loans** (`installments_payments.csv`)
- Checked first: among strong-score applicants, underpaying more than 5% of installments
  meant **11.9% default vs 3.9%**
- 94.6% of applicants with no credit bureau file do have this repayment history
- Result on the untouched blackbox: **0.7635 to 0.7713 (+0.0077)**, 95% interval
  +0.0047 to +0.0107. Twice what all the tuning achieved
- Removing the external scores entirely costs 0.0365, confirming the dependence

**Figures:** `slide07a_repayment_history_evidence.png`,
`slide07b_iteration_results_blackbox.png`

---

## Slide 8: Fairness (1:30) · Presenter 4

- Removing a column does not hide it: gender can be rebuilt from the other features with
  **AUC 0.906**, age with 0.970
- **Final model withholds gender and family status** (protected under ECOA). Cost: 0.003
  AUC. Gender approval ratio is 0.908 on the test set, above the 0.80 "four-fifths"
  screen (on the blackbox rows, withholding raised it from 0.891 to 0.916)
- **Age fails** (0.675): about 3 in 10 good applicants aged 20 to 25 are refused.
  Removing age made it *worse*, because the model rebuilds it from proxies
- **Region rating fails** (0.772), which matters because geography is how redlining worked
- **Race is not in the data at all**, so we cannot test the most serious form of lending
  discrimination

**Figures:** `slide08a_approval_ratio_four_fifths.png`, `slide08b_age_band_disparity.png`

*Speaker note:* say plainly that we do not claim the model is fair. That honesty is the
point of the slide.

---

## Slide 9: Final results (1:00) · Presenter 4

- Test rows scored **once**, after every decision was made
- Stratified test: **ROC-AUC 0.770**, PR-AUC 0.260, recall 0.47, precision 0.24.
  Catches nearly half of defaulters; about 1 in 4 flagged applicants defaults
- Random and stratified give near-identical, slightly optimistic numbers
- Deployment-like is lowest and most realistic: **about 0.766 for new applicants**
- Test matched validation on every split, so no overfitting to validation

**Figure:** `slide09_split_comparison.png`

---

## Slide 10: Conclusions (0:45) · Presenter 4

- Error analysis beat tuning: one targeted data change outperformed 13 tuning runs
- Final model: 0.770 test, about 0.766 expected on new applicants
- Gender disparity reduced at almost no cost
- **Not deployable as-is:** age and region disparities need a compliance review, and race
  must be collected under fair lending rules before it can be tested
- Next steps: previous-application features, model calibration, and recalibrating
  thresholds once costs of false alarms vs missed defaults are known

*No figure.*
