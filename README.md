# Predicting Loan Repayment Risk with a Neural Network

DATA 4401, Project 2 (Fall 2026)

Owen Loughery, Rhea Chellani, Ben Cox, Cody

We built a feed-forward neural network that predicts whether a loan applicant will have
trouble repaying, using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
data from Kaggle. The model was improved through manual error analysis and evaluated for
fairness across protected groups.

## Results

| | ROC-AUC |
|---|---|
| Final model, stratified held-out test set | **0.770** |
| Final model, deployment-like test set (most realistic estimate for new applicants) | **0.766** |
| Gain from the error-analysis change (repayment history) | **+0.0077** (95% interval +0.0047 to +0.0107) |

- **Optimizing metric:** ROC-AUC. **Fairness** was used as a constraint when choosing the
  final model.
- **Final model:** a 128-64-32 network with dropout 0.2 and L2 1e-4, trained with gender
  and family status withheld as inputs.
- **Fairness:** passes the four-fifths approval ratio screen for gender, family status,
  children and education. It **fails for age and region rating**. Race is not in the
  dataset, so it could not be tested. Full details are in Phases 13 to 15 of
  `Owen_Model_Run.ipynb`.

The target is the dataset's "payment difficulties" flag (a late payment early in the
loan), which we refer to as default for short. 8.07% of applicants have it.

## Repository contents

| File | What it is |
|---|---|
| `Owen_EDA.ipynb` | Exploratory data analysis and data preparation on their own (Phases 1 to 8). A lighter read for anyone interested only in the data work; everything in it is also included in `Owen_Model_Run.ipynb`. |
| `Owen_Model_Run.ipynb` | **The complete pipeline, and the notebook to run.** Phases 1 to 8 are the same EDA, followed by the three splits, the PyTorch network and tuning, error analysis, the second iteration, the fairness audit, and the final test-set evaluation (Phases 9 to 15). |
| `mortgage_error_analysis_100.csv` | Error analysis spreadsheet: 111 manually reviewed errors, plus one summary row per model iteration recording what changed and whether it helped. |

Each notebook explains its decisions in markdown cells alongside the code.

## How to run

### 1. Get the data

Download the competition files from
[Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) (a free
account is required) and put all of the CSVs in a `data/` folder next to the notebooks:

```
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── installments_payments.csv
├── previous_application.csv
├── POS_CASH_balance.csv
├── credit_card_balance.csv
├── sample_submission.csv
└── HomeCredit_columns_description.csv
```

A folder named `home-credit-default-risk/` (Kaggle's default download name) also works;
the notebooks check for either. The data is about 2.5 GB and is not included in this
repository.

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn torch pyarrow matplotlib jupyter
```

Tested with Python 3.14.5, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.0, torch 2.14.1,
pyarrow 25.0.1 and matplotlib 3.11.1. The network runs on CPU, Apple Silicon (MPS) or an
NVIDIA GPU, whichever is available.

### 3. Run the notebooks

- `Owen_Model_Run.ipynb` reproduces every result on its own. A full run takes about 10 to 15 minutes.
- `Owen_EDA.ipynb` is optional: its contents are the first 8 phases of the model notebook.

Run the cells from top to bottom. Random seeds are fixed (42), so the results match the
saved outputs.

**Files the notebooks write:** `data/processed/*.parquet` (the cleaned data),
`mortgage_error_analysis.csv` (a blank error-review template), and iteration summary rows
appended to `mortgage_error_analysis_100.csv`. Re-running replaces those summary rows
rather than duplicating them.

## Approach in brief

1. **Data preparation:** fixed placeholder values, removed 65 redundant or near constant
   columns after testing that they added nothing, and built features including credit
   ratios, credit bureau history, and an average of the three external credit scores
   (the strongest single feature).
2. **Splits:** random, stratified, and deployment like. For the last one, adversarial
   validation (AUC 0.905) showed that Kaggle's test applicants differ from the training
   applicants, so the most similar training applicants were held out for validation and
   testing.
3. **Model:** a PyTorch feed-forward network. Thirteen tuning runs moved validation AUC by
   only 0.004.
4. **Error analysis:** following *Machine Learning Yearning*, validation was split into an
   eyeball set (619 applicants, 111 errors reviewed by hand) and a blackbox set (45,507
   applicants). 54% of errors came from over reliance on the external credit scores.
5. **Iteration 2:** added each applicant's repayment history on previous loans, which
   improved blackbox AUC from 0.7635 to 0.7713.
6. **Fairness and evaluation:** audited approval rates and error rates across protected
   groups, then scored the held-out test sets once at the end.
