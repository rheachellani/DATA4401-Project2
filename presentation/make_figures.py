"""Build every figure used in the Project 2 presentation.

Run from the project2 folder:  ../../../.venv/bin/python presentation/make_figures.py
Writes labelled PNGs to presentation/figures/, named by the slide they belong to.

Figures computed from data read the raw Kaggle CSVs and data/processed/*.parquet.
Model results (AUCs, fairness audits) are copied from the executed
Owen_Model_Run.ipynb, since recomputing them means retraining the networks; the
cell each number comes from is noted beside it.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path("presentation/figures")
OUT.mkdir(parents=True, exist_ok=True)
DATA = next(p for p in [Path("data"), Path("home-credit-default-risk")] if p.exists())

# Palette (validated: blue/orange pass every CVD and contrast check on light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, GRAY = "#e1e0d9", "#c3c2b7", "#bdbbb3"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def finish(fig, ax, name, title, subtitle, extra_top=0.0):
    """Title and subtitle at fixed distances (in inches) from the top, so short and
    tall figures get the same spacing and the two lines can never collide."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.30 / h, title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.72 / h, subtitle, ha="left", va="top", fontsize=11.5, color=INK2)
    fig.subplots_adjust(top=1 - (1.25 + extra_top) / h)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def hbar_labels(ax, bars, fmt, pad):
    for b in bars:
        ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2, fmt(b.get_width()),
                va="center", fontsize=11, color=INK2)


app = pd.read_parquet(DATA / "processed/application_train_processed.parquet")
kag = pd.read_parquet(DATA / "processed/application_test_processed.parquet")


# ---------- Slide 3a: the arch Pearson misses ----------
dec = pd.qcut(app.CREDIT_INCOME_RATIO, 10)
rates = app.groupby(dec, observed=True).TARGET.mean() * 100
r = app.CREDIT_INCOME_RATIO.corr(app.TARGET)
fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(1, 11)
ax.plot(x, rates.values, color=BLUE, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2)
ax.axhline(app.TARGET.mean() * 100, color=MUTED, lw=1)
ax.text(10.15, app.TARGET.mean() * 100, f"overall\n{app.TARGET.mean():.1%}", va="center", fontsize=10, color=MUTED)
peak = int(np.argmax(rates.values))
ax.annotate(f"riskiest: {rates.values[peak]:.1f}%", (x[peak], rates.values[peak]),
            xytext=(0, 14), textcoords="offset points", ha="center", fontsize=11, color=INK)
ax.set_xticks(x, [f"{iv.left:.1f}–{iv.right:.1f}" if i not in (0,) else f"<{iv.right:.1f}"
                  for i, iv in enumerate(rates.index)], rotation=0, fontsize=9.5)
ax.set_xlabel("credit-to-income ratio, split into ten equal-size groups")
ax.set_ylabel("default rate (%)")
ax.set_ylim(6, 10.2); ax.set_xlim(0.5, 10.9)
ax.grid(axis="x", visible=False)
finish(fig, ax, "slide03a_credit_income_ratio_arch.png",
       "Risk rises then falls with loan size, a pattern correlation misses",
       f"Default rate by credit-to-income ratio  ·  Pearson correlation is only {r:+.3f}, "
       f"yet the groups range from {rates.min():.1f}% to {rates.max():.1f}%")


# ---------- Slide 3b: Information Value ranking ----------
eda = json.load(open("Owen_EDA.ipynb"))["cells"]
g = {"np": np, "pd": pd}
exec(next("".join(c["source"]) for c in eda if "def bin_feature" in "".join(c["source"])), g)
ivs = {c: g["information_value"](app, c) for c in app.columns if c not in ("SK_ID_CURR", "TARGET")}
top = pd.Series(ivs).sort_values(ascending=False).head(12)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
colors = [BLUE if n.startswith("EXT_SOURCE") else GRAY for n in top.index]
bars = ax.barh(top.index, top.values, height=0.62, color=colors)
hbar_labels(ax, bars, lambda v: f"{v:.3f}", 0.006)
for cut, lab in [(0.1, "medium"), (0.3, "strong")]:
    ax.axvline(cut, color=AXIS, lw=1)
    ax.text(cut, 11.75, f" {lab} ({cut})", fontsize=10, color=MUTED, va="bottom")
ax.set_xlabel("Information Value (how strongly the feature separates defaulters)")
ax.set_xlim(0, 0.7); ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=10.5)
finish(fig, ax, "slide03b_top_features_information_value.png",
       "The external credit scores dominate, led by the average we built",
       "Top 12 features by Information Value  ·  blue = external score features "
       "(EXT_SOURCE_MEAN is our engineered average of the three)")


# ---------- Slide 3c: the 43 building columns did nothing ----------
abl = pd.DataFrame({   # Owen_EDA.ipynb Phase 4, HistGradientBoosting, 3 seeds
    "config": ["no building columns", "a single yes/no flag", "flag + 14 average columns", "flag + all 43 columns"],
    "cols": [62, 63, 77, 106],
    "auc": [0.75246, 0.75219, 0.75182, 0.75200], "sd": [0.00102, 0.00160, 0.00128, 0.00194]})[::-1]
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(abl))
ax.errorbar(abl.auc, y, xerr=abl.sd, fmt="o", color=BLUE, ms=9, mec=SURFACE, mew=2,
            ecolor=MUTED, elinewidth=1.5, capsize=4)
for yi, (_, row) in zip(y, abl.iterrows()):
    ax.text(0.7565, yi, f"{row.auc:.4f}  ({row.cols} columns)", va="center", fontsize=11, color=INK2)
ax.set_yticks(y, abl.config); ax.set_xlim(0.746, 0.762)
ax.set_xlabel("ROC-AUC (dot = mean of 3 runs, line = spread across runs)")
ax.grid(axis="y", visible=False)
finish(fig, ax, "slide03c_building_columns_ablation.png",
       "Deleting 43 building columns cost nothing",
       "Same model trained with and without the building-description columns  ·  "
       "every result is within run-to-run noise")


# ---------- Slide 4: who applies has shifted ----------
panels = [
    ("revolving loans", (app.NAME_CONTRACT_TYPE == "Revolving loans").mean() * 100,
     (kag.NAME_CONTRACT_TYPE == "Revolving loans").mean() * 100, "{:.1f}%"),
    ("median number of payments", app.CREDIT_TERM.median(), kag.CREDIT_TERM.median(), "{:.1f}"),
    ("median loan / income", app.CREDIT_INCOME_RATIO.median(), kag.CREDIT_INCOME_RATIO.median(), "{:.2f}"),
    ("first external score missing", app.EXT_SOURCE_1.isna().mean() * 100,
     kag.EXT_SOURCE_1.isna().mean() * 100, "{:.1f}%"),
]
fig, axes = plt.subplots(1, 4, figsize=(13, 4.4))
for ax, (lab, a, b, fmt) in zip(axes, panels):
    bars = ax.bar([0, 1], [a, b], width=0.6, color=[BLUE, ORANGE])
    for bar, v in zip(bars, [a, b]):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, fmt.format(v),
                ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_title(lab, fontsize=12, color=INK, pad=10)
    ax.set_xticks([0, 1], ["training", "Kaggle\ntest"], fontsize=10.5)
    ax.set_ylim(0, max(a, b) * 1.25); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(False)
finish(fig, axes[0], "slide04_train_vs_kaggle_shift.png",
       "Future applicants look different from past ones (adversarial AUC 0.905)",
       "A classifier separates training applicants from Kaggle test applicants with AUC 0.905; "
       "no single feature explains it, these do together", extra_top=0.35)


# ---------- Slide 5: tuning plateau ----------
tune = pd.DataFrame([   # Owen_Model_Run.ipynb cells 95, 98, 102 (stratified validation)
    ("baseline 128-64", "regularization", 0.759942), ("dropout 0.2", "regularization", 0.760252),
    ("L2 1e-5", "regularization", 0.760488), ("dropout 0.3", "regularization", 0.760670),
    ("L2 1e-4", "regularization", 0.760831), ("dropout 0.2 + L2 1e-4", "regularization", 0.762853),
    ("class weight 11.4", "class weights", 0.761163), ("class weight 2.0", "class weights", 0.762103),
    ("class weight 4.0", "class weights", 0.762107), ("class weight 3.4", "class weights", 0.762437),
    ("smaller 64-32", "architecture", 0.762371), ("wider 256-128", "architecture", 0.763411),
    ("deeper 128-64-32", "architecture", 0.763632)], columns=["run", "family", "auc"])
fig, ax = plt.subplots(figsize=(10, 6.2))
yy = np.arange(len(tune))[::-1]
best = tune.auc.idxmax()
ax.scatter(tune.auc, yy, s=70, color=[BLUE if i == best else GRAY for i in tune.index],
           edgecolor=SURFACE, linewidth=2, zorder=3)
ax.set_yticks(yy, [f"{r.run}" for r in tune.itertuples()], fontsize=10.5)
for fam in tune.family.unique():
    rows = yy[tune.family == fam]
    ax.text(0.7533, rows.mean(), fam, rotation=90, va="center", ha="center", fontsize=10, color=MUTED)
ax.axvspan(tune.auc.min(), tune.auc.max(), color=BLUE, alpha=0.06, zorder=0)
ax.text(tune.auc.max() + 0.0003, yy[best], f"best {tune.auc.max():.4f}", va="center", fontsize=11, color=INK)
ax.set_xlim(0.7525, 0.7665)
ax.set_xlabel("validation ROC-AUC")
ax.grid(axis="y", visible=False)
finish(fig, ax, "slide05_tuning_plateau.png",
       "Thirteen tuning runs moved AUC by only 0.004",
       f"Regularization, class weights and architecture all land between "
       f"{tune.auc.min():.4f} and {tune.auc.max():.4f} (shaded)")


# ---------- Slide 6a: eyeball / blackbox design ----------
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=SURFACE, ec=edge, lw=2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.9, body, ha="center", va="top", fontsize=11.5, color=INK2, linespacing=1.5)
box(0.2, 1.5, 3.0, 2.2, "Validation set", "46,126 applicants\nerror rate 16.2%", AXIS)
box(4.4, 2.75, 6.3, 2.35, "Eyeball set: 619 applicants",
    "sized to contain about 100 mistakes\n111 errors (80 false alarms, 31 missed defaults)\nevery one reviewed by hand", BLUE)
box(4.4, 0.1, 6.3, 2.35, "Blackbox set: 45,507 applicants",
    "never looked at\nthe only place a change made because\nof the review is allowed to be scored", ORANGE)
for ty in (3.9, 1.3):
    ax.annotate("", xy=(4.35, ty), xytext=(3.25, 2.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
finish(fig, ax, "slide06a_eyeball_blackbox_split.png",
       "Inspect one part, judge changes on the other",
       "Machine Learning Yearning setup: reviewing the blackbox rows would let the review leak into the scores")


# ---------- Slide 6b: error categories ----------
rev = pd.read_csv("mortgage_error_analysis_100.csv")
rev = rev[rev.prediction_type != "ITERATION_SUMMARY"].copy()
rev["cat"] = rev.manual_category.str.replace(r"^F[NP]: ", "", regex=True)
ct = rev.groupby(["cat", "prediction_type"]).size().unstack(fill_value=0)
ct["total"] = ct.sum(axis=1); ct = ct.sort_values("total")
fig, ax = plt.subplots(figsize=(10, 5))
fn = ax.barh(ct.index, ct.false_negative, height=0.6, color=BLUE, label="missed default (false negative)")
fp = ax.barh(ct.index, ct.false_positive, height=0.6, left=ct.false_negative + np.where(ct.false_negative > 0, 0.4, 0),
             color=ORANGE, label="false alarm (false positive)")
for yi, (name, row) in enumerate(ct.iterrows()):
    ax.text(row.total + 1.4, yi, f"{row.total:.0f}  ({row.total/len(rev):.0%})", va="center", fontsize=11, color=INK)
ax.set_xlim(0, 72); ax.set_xlabel("number of reviewed errors (111 total)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
finish(fig, ax, "slide06b_error_categories.png",
       "Over half the mistakes trace back to the external credit scores",
       "Manual review of 111 eyeball-set errors  ·  good scores hid 21 of 31 missed defaults; "
       "poor scores caused 39 of 80 false alarms")


# ---------- Slide 7a: repayment history separates risk the scores hide ----------
a = pd.read_csv(DATA / "application_train.csv", usecols=["SK_ID_CURR", "TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
a["ext"] = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
ins = pd.read_csv(DATA / "installments_payments.csv")
per = ins.groupby(["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]).agg(
    due=("DAYS_INSTALMENT", "first"), owed=("AMT_INSTALMENT", "first"),
    paid_day=("DAYS_ENTRY_PAYMENT", "max"), paid=("AMT_PAYMENT", "sum")).reset_index()
per["late"] = per.paid_day > per.due
per["short"] = per.paid < per.owed * 0.99
cur = per.groupby("SK_ID_CURR").agg(late=("late", "mean"), short=("short", "mean"))
s = a[a.ext >= 0.5].merge(cur, left_on="SK_ID_CURR", right_index=True, how="left")
rows = [
    ("paid 10% or fewer installments late", s.loc[~(s.late > 0.10), "TARGET"], GRAY),
    ("paid more than 10% late", s.loc[s.late > 0.10, "TARGET"], BLUE),
    ("underpaid 5% or fewer installments", s.loc[~(s.short > 0.05), "TARGET"], GRAY),
    ("underpaid more than 5%", s.loc[s.short > 0.05, "TARGET"], BLUE),
][::-1]
fig, ax = plt.subplots(figsize=(10, 4.6))
yy = [0, 1, 2.6, 3.6]
bars = ax.barh(yy, [r[1].mean() * 100 for r in rows], height=0.62, color=[r[2] for r in rows])
for b, (lab, ser, _) in zip(bars, rows):
    ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
            f"{ser.mean():.1%}  (n = {len(ser):,})", va="center", fontsize=11, color=INK2)
ax.set_yticks(yy, [r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 15.5); ax.set_xlabel("default rate (%)")
ax.grid(axis="y", visible=False)
finish(fig, ax, "slide07a_repayment_history_evidence.png",
       "Repayment history finds the risk hidden behind good scores",
       f"Only applicants with strong external scores ({len(s):,} people, {s.TARGET.mean():.1%} default overall)  ·  "
       "from installments_payments.csv, checked before building the features")


# ---------- Slide 7b: iteration results on the blackbox ----------
it = pd.DataFrame([   # Owen_Model_Run.ipynb Phase 12 comparison cell
    ("iteration 1", 0.76353, GRAY), ("iteration 2: + repayment history", 0.77129, BLUE),
    ("final: iteration 2, gender + family status withheld", 0.76840, BLUE),
    ("also withholding age (rejected)", 0.76614, GRAY),
    ("no external scores (diagnostic)", 0.73483, GRAY)], columns=["model", "auc", "c"])[::-1]
fig, ax = plt.subplots(figsize=(10.5, 4.8))
yy = np.arange(len(it))
ax.scatter(it.auc, yy, s=110, color=it.c, edgecolor=SURFACE, linewidth=2, zorder=3)
for yi, r in zip(yy, it.itertuples()):
    ax.text(r.auc + 0.0012, yi, f"{r.auc:.4f}", va="center", fontsize=11.5, color=INK)
ax.set_yticks(yy, it.model, fontsize=11)
ax.set_xlim(0.725, 0.782); ax.set_xlabel("ROC-AUC on the 45,507 blackbox rows")
ax.grid(axis="y", visible=False)
finish(fig, ax, "slide07b_iteration_results_blackbox.png",
       "Repayment history added +0.0077 AUC, more than all tuning combined",
       "95% bootstrap interval +0.0047 to +0.0107; iteration 2 won in 100% of 500 resamples  ·  "
       "removing the external scores costs 0.0365")


# ---------- Slide 8a: four-fifths screen ----------
ff = pd.DataFrame({   # Owen_Model_Run.ipynb Phase 15 fairness cell, final model, test sets
    "attribute": ["gender", "children", "family status", "education", "region rating", "age band"],
    "stratified": [0.9078, 0.9445, 0.8584, 0.8443, 0.7716, 0.6753],
    "deployment": [0.9087, 0.9335, 0.8479, 0.8125, 0.7528, 0.6081]}).sort_values("stratified")
fig, ax = plt.subplots(figsize=(10, 5))
yy = np.arange(len(ff))
ax.hlines(yy, ff[["stratified", "deployment"]].min(axis=1), ff[["stratified", "deployment"]].max(axis=1), color=AXIS, lw=2)
ax.scatter(ff.stratified, yy + 0.09, s=100, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="stratified test set")
ax.scatter(ff.deployment, yy - 0.09, s=100, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="deployment-like test set")
ax.axvline(0.80, color=INK2, lw=1.4)
ax.text(0.797, 0.5, "0.80 screen ", ha="right", va="center", fontsize=10.5, color=INK2)
for yi, r in zip(yy, ff.itertuples()):
    verdict = "fails" if min(r.stratified, r.deployment) < 0.80 else "passes"
    ax.text(1.005, yi, verdict, va="center", fontsize=11, color=INK,
            fontweight="bold" if verdict == "fails" else "normal")
ax.set_yticks(yy, ff.attribute, fontsize=11.5)
ax.set_xlim(0.55, 1.05); ax.set_xlabel("approval ratio: least-approved group ÷ most-approved group")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
finish(fig, ax, "slide08a_approval_ratio_four_fifths.png",
       "Gender passes the four-fifths screen; age and region rating do not",
       "Final model on held-out test applicants  ·  below 0.80 suggests possible disparate impact  ·  "
       "race is not in the data and cannot be checked")


# ---------- Slide 8b: age band detail ----------
age = pd.DataFrame({   # Owen_Model_Run.ipynb Phase 13 detail cell, final model, blackbox rows
    "band": ["20–25", "25–30", "30–40", "40–50", "50–60", "60–70"],
    "approval": [0.6486, 0.7106, 0.8008, 0.8591, 0.9149, 0.9627],
    "fpr": [0.3161, 0.2497, 0.1644, 0.1161, 0.0697, 0.0305],
    "default": [0.1153, 0.1044, 0.0977, 0.0754, 0.0636, 0.0494]})
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(age))
ax.plot(x, age.approval * 100, color=BLUE, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2, label="approved")
ax.plot(x, age.fpr * 100, color=ORANGE, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2,
        label="good customers wrongly refused")
for col, c in [("approval", BLUE), ("fpr", ORANGE)]:
    for i in (0, len(age) - 1):
        v = age[col].iloc[i] * 100
        ax.text(x[i] + (-0.12 if i == 0 else 0.12), v, f"{v:.0f}%", ha="right" if i == 0 else "left",
                va="center", fontsize=11.5, color=INK)
ax.set_xticks(x, age.band); ax.set_xlabel("applicant age")
ax.set_ylabel("share of applicants (%)"); ax.set_ylim(0, 105); ax.set_xlim(-0.6, 5.6)
ax.grid(axis="x", visible=False)
ax.legend(loc="center right", frameon=False, fontsize=11)
finish(fig, ax, "slide08b_age_band_disparity.png",
       "Young applicants are refused far more often, including good ones",
       "Final model, blackbox validation rows  ·  real default rates also differ "
       "(11.5% at 20–25 vs 4.9% at 60–70), but not by this much")


# ---------- Slide 9: split comparison ----------
sp = pd.DataFrame({   # Owen_Model_Run.ipynb Phase 15 results cell
    "split": ["random", "stratified", "deployment-like"],
    "val": [0.7674, 0.7689, 0.7636], "test": [0.7674, 0.7698, 0.7657]})[::-1]
fig, ax = plt.subplots(figsize=(10, 4.4))
yy = np.arange(len(sp))
ax.hlines(yy, sp.val, sp.test, color=AXIS, lw=2)
ax.scatter(sp.val, yy, s=110, color=GRAY, edgecolor=SURFACE, linewidth=2, zorder=3, label="validation")
ax.scatter(sp.test, yy, s=110, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="held-out test")
for yi, r in zip(yy, sp.itertuples()):
    same = " (same as validation)" if abs(r.val - r.test) < 5e-5 else ""
    ax.text(max(r.val, r.test) + 0.0004, yi, f"test {r.test:.4f}{same}", va="center", fontsize=11.5, color=INK)
ax.set_yticks(yy, sp.split, fontsize=12)
ax.set_xlim(0.762, 0.7725); ax.set_xlabel("ROC-AUC")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
finish(fig, ax, "slide09_split_comparison.png",
       "Test scores matched validation on every split",
       "Final model trained on each split, test rows scored once  ·  deployment-like is the realistic "
       "estimate for new applicants (about 0.766)")
