# Project Echo: Socio-Economic Income Predictor

> AI-based **regression** model that predicts an individual's annual wage income
> from socio-economic, demographic, and geographic factors in the
> **2024 ACS PUMS (American Community Survey, Public Use Microdata Sample)**,
> sourced live from the **U.S. Census Data API**.

_Last updated: June 2026_

**Repository:** <https://github.com/iceyatom/Project_Echo2>

---

## What Is This?

Project Echo is a machine-learning project that predicts an individual's **wages/salary
income (`WAGP`)** from 18 socio-economic, demographic, and geographic features drawn from
the 2024 ACS PUMS person records.

The work is organized as a pair of Jupyter notebooks:

1. **Extraction** (`acs2024v1.1_income_extraction.ipynb`) — pulls the raw 2024 ACS PUMS
   records from the Census Data API, filters to the target population, and writes a clean
   dataset (`acs2024v1.1_income.csv`).
2. **Modeling** (`acs2024v1.1_exp4_model.ipynb`) — loads that CSV and benchmarks **ten
   regressors** end-to-end, finishing with a stacking ensemble.

### Final solution — `acs2024v1.1_exp4_model.ipynb` ("Experiment 4")

Experiment 4 is the **final/best configuration** for this project. It keeps all 18 raw
features (no feature dropping) and layers on engineered signal designed for the models that
can't discover it on their own:

- **Interaction / Mincer terms:** `ANNUAL_HOURS = WKHP × WKWN`, `AGE2 = AGEP²` (the concave
  age–earnings curve), and `SCHLxAGEP` (education × experience).
- **Leak-safe encodings** (fit on the training split only, out-of-fold on train): smoothed
  **target encoding** and **frequency encoding** of the high-cardinality codes
  `OCCP`/`INDP`/`POBP`/`ST`, an `OCCP`×`INDP` **pair** target-encoding ("what does this job
  pay *in this sector*"), and `OCCP_te`/`INDP_te` × hours interactions.
- **Ten regressors** grouped by family — linear (Ridge, Linear, Lasso), boosted trees
  (XGBoost, LightGBM, CatBoost), bagging/other (RandomForest, ExtraTrees,
  HistGradientBoosting), and a **stacking blend** of the six tree learners with a Ridge
  meta-learner.

All models train on `log1p(WAGP)` and invert with `expm1`. The high-cardinality categoricals
are kept as integer codes and handed to the native-categorical boosters (no one-hot for the
tree path).

### Results (executed on the full v1.1 extract, 1,359,710 rows)

| Model | Test R² | Test MAE |
|---|---|---|
| **Stacking (GBMs)** | **0.5154** | **\$27,529** |
| LightGBM | 0.511 | \$27,899 |
| XGBoost | 0.508 | \$27,875 |
| CatBoost | 0.507 | \$27,942 |
| HistGradientBoosting | 0.504 | \$28,190 |
| ExtraTrees | 0.502 | \$27,967 |
| RandomForest | 0.494 | \$28,441 |
| Ridge / Linear | 0.43 | ~\$31,600 |
| Lasso | 0.42 | \$32,045 |

The wage signal is **feature-bound**: the boosted trees cluster near Test R² ≈ 0.51 and the
stacking blend tops out at **0.5154**. Further gains require richer *raw* predictors
(sub-state geography/PUMA, finer occupation detail, employer/job tenure, survey weights),
not more re-encoding of the existing 18 columns.

> **Note on units:** the extractor stores `WAGP` scaled by `1e-6` (≈ `[0, 0.92]`, i.e.
> *millions* of constant dollars), so the modeling notebook multiplies predictions back by
> `1e6` to report MAE/RMSE in dollars. R² and Explained Variance are scale-free either way.

---

## The Final Ensemble — How the Stacking Blend Works

The winning model (`Stacking(GBMs)`, Test R² **0.5154**) is a **holdout-blending stacked
ensemble** (`StackedGBM`) built on top of the other fitted tree models. It is the project's
"capstone" model — it doesn't learn from the raw features directly, it learns how to best
*combine* the predictions of the models that do.

### Which models make up the ensemble

The blend stacks **six tree-based base learners** — every tree model in the benchmark, across
three sub-families:

| Base learner | Family | Role |
|---|---|---|
| **XGBoost** | Gradient-boosted trees | Strong native-categorical booster |
| **LightGBM** | Gradient-boosted trees | Strong native-categorical booster (best single model) |
| **CatBoost** | Gradient-boosted trees | Strong native-categorical booster |
| **ExtraTrees** | Bagging (extremely randomized) | Decorrelated, higher-variance learner |
| **HistGradientBoosting** | Histogram boosting | Ordinal-code booster (no native categoricals) |
| **RandomForest** | Bagging | Decorrelated, higher-variance learner |

The **three linear models (Ridge / Linear / Lasso) are deliberately *not* in the blend.** At
Test R² ≈ 0.43 they are far weaker than the trees (≈ 0.50–0.51) and largely redundant with
them, so including them would mostly inject noise into the meta-learner for no gain.

### How the weights are decided

The ensemble uses **holdout blending** with a **regularized linear meta-learner**:

1. **Base models are fit on the training split** (`X_train`) — exactly as benchmarked
   individually, no refitting for the stack.
2. **Each base model predicts on the held-out validation split** (`X_val`). Stacking those
   six prediction vectors column-wise gives a small `(n_val × 6)` matrix — one column of
   `log1p(WAGP)` predictions per model.
3. **A `Ridge(alpha=1.0)` regression is fit** with that 6-column matrix as its inputs and the
   true validation `log1p(WAGP)` as its target. **The Ridge coefficients *are* the blend
   weights** — they tell the ensemble how much to trust each base model.
4. **At inference**, all six base models predict, and the Ridge meta-learner combines those
   predictions with its learned weights (the result is then `expm1`-inverted, and ×1e6 for
   dollars).

Because the weights are ordinary regression coefficients (with an intercept), they are **not
constrained to be positive or to sum to 1** — a model can receive a zero or negative weight if
the meta-learner finds it redundant or useful only as a correction term. The weights learned in
the executed run:

| Base learner | Meta weight (Ridge coef) |
|---|---|
| LightGBM | **+0.32** |
| CatBoost | **+0.31** |
| XGBoost | **+0.28** |
| ExtraTrees | +0.25 |
| RandomForest | −0.00 |
| HistGradientBoosting | −0.15 |

The blend leans on the **three gradient boosters plus ExtraTrees**, ignores RandomForest
(≈ 0), and gives HistGradientBoosting a small **negative** weight — i.e. it is used as a
decorrelating correction rather than as a positive contributor.

### Why it was done this way

- **Holdout blend instead of full cross-validated stacking.** Proper CV stacking would refit
  the heavy base models across every CV fold on the ~870k training rows — expensive for a
  marginal accuracy gain. Blending on a single held-out split captures almost all of the benefit at a
  fraction of the cost (the stack trains in seconds, versus minutes for the boosters).
- **A Ridge meta-learner instead of a plain average or unregularized `LinearRegression`.** The
  six base predictions are **highly correlated** (they predict the same target from the same
  features), which makes ordinary least squares unstable — it produces wild, overfit
  coefficients on collinear inputs. **Ridge's L2 penalty (`alpha=1.0`) shrinks the
  coefficients**, stabilizing them so the blend generalizes from validation to test instead of
  memorizing the validation split. (The earlier exp2 stack used a plain linear meta over just
  three boosters; exp4 widened it to six trees and switched to Ridge specifically to tame the
  extra collinearity.)
- **Let the data choose the weights, including down-weighting redundancy.** Rather than
  hand-pick which models to average, the meta-learner *learns* that the three boosters carry
  most of the signal, that ExtraTrees adds a bit of useful diversity, and that RandomForest /
  HistGradientBoosting are largely redundant — automatically routing weight toward the models
  that help and away from those that don't.

The net effect is modest — Test R² **0.5154** versus the best single model (LightGBM at
**0.5109**) — because the base learners are already near the data's feature-bound ceiling
(≈ 0.51). The blend's value is squeezing out the best available test score and being more
robust than betting on any one model.

---

## Feature Correlation & Why No Features Were Dropped

Stage 5c computes a **Pearson correlation matrix** over the numeric features plus the
`log1p(WAGP)` target. (The high-cardinality categoricals — `OCCP`, `INDP`, `POBP`, `ST`, etc. —
are left out of this matrix: their integer values are **nominal codes**, so a linear
correlation on them is meaningless.)

That matrix shows several **strongly correlated feature pairs**:

| Feature pair | Pearson r | Why they're correlated |
|---|---|---|
| `AGEP` ↔ `AGE2` | **+0.99** | `AGE2 = AGEP²` — it *is* age, squared |
| `WKHP` ↔ `ANNUAL_HOURS` | **+0.90** | `ANNUAL_HOURS = WKHP × WKWN` — built from hours |
| `AGEP` ↔ `SCHLxAGEP` | **+0.85** | `SCHLxAGEP = SCHL × AGEP` — built from age |
| `AGE2` ↔ `SCHLxAGEP` | **+0.83** | both are driven primarily by `AGEP` |
| `WKWN` ↔ `ANNUAL_HOURS` | **+0.63** | `ANNUAL_HOURS` also contains weeks worked |

The important pattern: **every highly correlated pair is one of the engineered terms
(`AGE2`, `SCHLxAGEP`, `ANNUAL_HOURS`) lined up against the raw feature it was derived from.**
The correlation is high *by construction* — it is not a surprise discovered in the data, it is
a direct consequence of how those Mincer/interaction features are defined.

### Why these features were kept anyway

Experiment 4 deliberately keeps **all** features (no Select-K-Best, no correlation-based
pruning). The high correlations are intentional, and dropping the redundant-looking columns
would *hurt* the model, for several reasons:

1. **The redundancy is the entire point of the engineered terms.** `AGE2`, `SCHLxAGEP`, and
   `ANNUAL_HOURS` exist precisely to hand the **linear models** the curvature (the concave
   age–earnings profile) and the interactions (education × experience, hours × weeks) that a
   linear model **cannot form on its own**. Dropping `AGE2` because it correlates with `AGEP`
   would throw away the exact nonlinear signal it was added to provide.
2. **Tree models are insensitive to multicollinearity.** Gradient-boosted and bagged trees
   (which make up the entire final ensemble) split on one feature at a time; correlated inputs
   do **not** destabilize them the way they destabilize an ordinary least-squares fit. Since
   the headline models are all trees, the collinearity simply doesn't hurt them.
3. **For the linear models, collinearity is handled by regularization, not deletion.** Where
   multicollinearity *does* inflate coefficient variance — the linear family — the project
   addresses it with **Ridge (L2)** and **Lasso (L1)** penalties, which are designed to cope
   with correlated predictors. (And the linear models aren't in the final stacking blend
   anyway, so any residual instability never reaches the headline result.)
4. **High correlation is not perfect correlation — each term still adds information.** None of
   the pairs hit 1.0; values of 0.63–0.99 mean each engineered feature still carries distinct
   signal (e.g. `ANNUAL_HOURS` differs from `WKHP` through `WKWN`; `AGE2` separates from `AGEP`
   in the tails). The **mutual-information ranking** confirms they are individually predictive
   of wages — `ANNUAL_HOURS` is the **2nd-strongest** feature (MI ≈ 0.31), with `SCHLxAGEP`
   (≈ 0.16) and `AGE2` (≈ 0.12) also ranking respectably — so they are real predictors, not
   noise to be pruned.
5. **Prior experiments showed pruning loses signal.** An earlier experiment that *did* apply
   feature selection (`Select-K-Best`) performed worse than keeping everything, so the project's
   settled approach is to retain all features and let the models (and the meta-learner) decide
   how much weight each one deserves.

---

## Data Sourcing & Census API Key (required)

The dataset is **not committed to the repo** (`*.csv` and the `data/` Parquet cache are
git-ignored). To reproduce it you run the extraction notebook, which pulls live from the
Census Data API — and that requires a **free API key**.

### 1. Get a free API key

Request one (takes a minute, emailed instantly):
<https://api.census.gov/data/key_signup.html>

### 2. Make the key available to the notebook

The extraction notebook looks for the key in **either** of these places (in order):

1. The **environment variable** `CENSUS_API_KEY`. In PowerShell:
   ```powershell
   $env:CENSUS_API_KEY = "your-key-here"
   ```
   > Set it **before** launching Jupyter/VS Code, because the kernel is a separate process.
   > If you set it in a terminal but the kernel can't see it, use option 2 instead.
2. A file named **`census_api_key.txt`** next to the notebook, containing only the key.
   This file is **git-ignored** (listed under "Secrets" in `.gitignore`) so it is never
   committed — keep it that way; **never paste your key into a tracked file or notebook cell.**

### 3. What the extraction does

- Requests **only the ~21 PUMS variables** the project uses (via the API `get=` parameter),
  one call per state — tens of MB total, not the full ~285-column PUMS file.
- Assembles all 50 states + DC into one frame and **caches it to `./data/...parquet`**, so
  re-runs skip the network.
- Applies the **population filter** — civilian employed (`ESR ∈ {1,2}`), working age
  (`AGEP` 18–64), positive wages (`WAGP > 0`), `WKHP ≥ 1` — taking the ~3.42M raw 2024
  records down to **1,359,710 rows × 19 columns** (18 features + `WAGP`).
- Inflation-adjusts wages via `ADJINC` and writes `acs2024v1.1_income.csv`.

> `DENSITY` (default `1.0`) optionally subsamples rows for a lighter run; set
> `STATES = ['CA','TX','NY','FL']` in Stage 2 for a quick partial pull.

### Why each population filter was applied

The population filter is the **single highest-impact preprocessing step** — it cuts the raw
~3.42M records to ~1.36M and is the main reason this dataset breaks past the old
`new_dataset.csv` ceiling (R² ≈ 0.30). The goal is a **homogeneous population of actual wage
earners**, because the features (occupation, industry, education, hours, etc.) explain *how
much a worker earns* — they cannot explain *why a non-worker earns $0*, which is a separate
labor-force-participation question. Mixing the two pollutes the target and inflates error.
Each condition removes a specific group that would otherwise add unexplainable noise:

| Filter | Keeps | Why it was applied |
|---|---|---|
| `ESR ∈ {1, 2}` | Civilian employed — "at work" (1) or "with a job, not at work" (2) | `ESR` is the employment-status recode. Dropping the unemployed (3), armed forces (4–5), and not-in-labor-force (6 — retirees, students, non-working disabled) removes people who have **no wage to predict**. Only the employed have a meaningful `WAGP`. |
| `AGEP` 18–64 | Prime working-age adults | Under 18 are mostly part-time/seasonal teens with sporadic earnings (and often still in school); 65+ are retirement-age, where wages no longer reflect labor-market value (partial retirement, pensions/Social Security). Restricting to 18–64 keeps the wage-determination relationship consistent instead of mixing in life-stage transitions. |
| `WKHP ≥ 1` | People working at least 1 usual hour/week | Removes rows coded as employed but reporting **zero usual hours** (extended leave, marginal/edge cases). This guarantees every row represents someone actually working, so `WKHP` is a real predictor and the wage corresponds to genuine labor. |
| `WAGP > 0` | People with positive wage/salary income | The target is wages. Zero-wage rows (e.g., self-employed reporting business income but no W-2 wages, or unpaid family workers) are **structural zeros** — a degenerate target for a wage model. Since training is on `log1p(WAGP)`, keeping `WAGP > 0` makes the task cleanly "how much do wage earners earn." |

In short: the filter strips out **retirees, teens, the non-employed, and zero-earners** — the
groups whose income the socio-economic features genuinely cannot explain — so the model learns
the wage signal among workers rather than being dragged down by non-earners.

### Feature columns

| Column | Meaning | Kind |
|---|---|---|
| `AGEP` | Age in years | continuous |
| `WKHP` | Usual hours worked per week | continuous |
| `WKWN` | Weeks worked in past 12 months | continuous |
| `COW` | Class of worker (employment sector) | categorical |
| `SCHL` | Education level (ordinal code) | categorical |
| `MAR` | Marital status | categorical |
| `OCCP` | Occupation code (high-cardinality) | categorical |
| `INDP` | Industry code (high-cardinality) | categorical |
| `POBP` | Place of birth (high-cardinality) | categorical |
| `RELSHIPP` | Relationship to householder | categorical |
| `SEX` | Sex | categorical |
| `RAC1P` | Race group | categorical |
| `HISP` | Hispanic origin | categorical |
| `CIT` | Citizenship status | categorical |
| `NATIVITY` | Nativity (native / foreign-born) | categorical |
| `ENG` | English-speaking ability | categorical |
| `DIS` | Disability | categorical |
| `ST` | State (FIPS code) | categorical |
| **`WAGP`** | **Target — wages/salary income (scaled ÷1e6)** | continuous |

---

## Files

| File | Description |
|------|-------------|
| `acs2024v1.1_income_extraction.ipynb` | **Final** extraction — pulls 2024 ACS PUMS from the Census API and writes the CSV |
| `acs2024v1.1_exp4_model.ipynb` | **Final** modeling notebook — Experiment 4 (10 regressors + stacking) |
| `acs2024v1.1_income.csv` | Cleaned dataset (1,359,710 × 19). **Git-ignored — regenerate via the extraction notebook** |
| `acs2024_*` / `acs2024v2_*` | Earlier benchmark family (base, exp1–exp6, v2/PUMA) kept for reference |
| `requirements.txt` | Python package dependencies |
| `census_api_key.txt` | Your Census API key (**git-ignored — create locally, never commit**) |
| `.gitignore` | Excludes `.venv/`, secrets, generated data/models from git |
| `README.md` | This file |

---

## Requirements

- **Python 3.9.x – 3.11.x** (Python 3.12+ is not supported by some wheels here; **3.11.x recommended**).
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

| Package | Purpose |
|---------|----------|
| `requests` | Call the Census Data API |
| `pandas`, `numpy` | Data assembly & cleaning |
| `pyarrow` / `fastparquet` | Read/write the cached Parquet extract |
| `scikit-learn` | Splits, pipelines, linear/forest models, metrics |
| `xgboost`, `lightgbm`, `catboost` | Boosted-tree regressors (and the stack's base learners) |
| `matplotlib`, `seaborn` | EDA & learning-curve plots |
| `joblib` | (Optional) persist fitted models in Stage 12 |

---

## How to Run

1. Clone the repo and open it in VS Code or Jupyter:
   ```
   git clone https://github.com/iceyatom/Project_Echo2.git
   cd Project_Echo2
   ```
2. Create and activate a virtual environment, then install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Provide your **Census API key** (see *Data Sourcing & Census API Key* above).
4. Run **`acs2024v1.1_income_extraction.ipynb`** top-to-bottom to produce
   `acs2024v1.1_income.csv` (first run pulls from the API and caches to `./data`;
   later runs use the cache).
5. Run **`acs2024v1.1_exp4_model.ipynb`** top-to-bottom to train and benchmark all ten
   models. The results table and Experiment Log at the end summarize Test R² / MAE.

> CatBoost dominates the runtime (full 1.36M-row fit is several minutes); a complete model
> run is on the order of ~10 minutes on a typical laptop.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No CENSUS_API_KEY found` | Get a free key and either set `$env:CENSUS_API_KEY` (restart the kernel) or create `census_api_key.txt` next to the notebook. |
| Extraction is slow / re-downloads every run | After the first successful pull the frame is cached to `./data/acs2024_pums_api_all.parquet`; keep that file to skip the network. |
| `acs2024v1.1_income.csv not found` (model notebook) | Run the extraction notebook first — the CSV is git-ignored and not shipped in the repo. |
| `ModuleNotFoundError: xgboost / lightgbm / catboost` | `pip install -r requirements.txt` (or install the named package). |
| Wheels fail to install | Use **Python 3.11.x**; 3.12+ lacks some required pre-built wheels. |

---

## Contact / Project

- **Repository:** <https://github.com/iceyatom/Project_Echo2>
- **Project:** Project Echo: Socio-Economic Income Predictor (CSC 180)
