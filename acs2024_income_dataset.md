# `acs2024_income.csv` — Dataset Description

A person-level wage/income dataset built from the **2024 American Community Survey
(ACS) Public Use Microdata Sample (PUMS), 1-Year file**. It was produced by
[acs2024_income_extraction.ipynb](acs2024_income_extraction.ipynb) and is consumed by the
modeling notebooks ([acs2024_base_model.ipynb](acs2024_base_model.ipynb),
`acs2024_exp1..6_model.ipynb`, etc.).

**At a glance**

| | |
|---|---|
| Rows (after all filtering) | **339,727** |
| Columns | **19** — 18 features + 1 target |
| Continuous features | 3 (`AGEP`, `WKHP`, `WKWN`) |
| Categorical / coded features | 15 |
| Target | `WAGP` (wages, normalized — see note) |
| Missing values | **0** |

> ⚠️ **The target `WAGP` is *not* in raw dollars.** It is a normalized value in
> roughly `[0, 0.92]`. Empirically `WAGP × 10⁶ ≈` inflation-adjusted annual wage in
> dollars (median `0.0528` ≈ **\$52.8k**; 99th pct `0.517` ≈ \$517k). Evaluate models
> with **R²**, not absolute-dollar MAE — a raw MAE of ~0.04 is an artifact of the
> scale, not "\$0.04 of error."

---

## 1. How it was sourced

The data is pulled live from the **U.S. Census Bureau's
[Microdata API](https://www.census.gov/data/developers/data-sets/census-microdata-api.html)**
(`https://api.census.gov/data/2024/acs/acs1/pums`), not a bulk file download.

| Decision | Value | Rationale (from the notebook) |
|---|---|---|
| **Survey** | ACS PUMS 1-Year | Latest released; 2025 PUMS not out yet. ~3.3M national records. |
| **Year** | 2024 | Single national year is plenty of data. |
| **Geography** | All 50 states + DC | One API call per state, concatenated. PR omitted. |
| **Columns requested** | ~21 variables only | The `get=` parameter pulls only the columns used (~21, not the full ~285-column PUMS), so the transfer is tens of MB. |
| **Row density** | `DENSITY = 0.25` | Keeps a 25% random subsample (`random_state=42`) — nationally representative but ~¼ the RAM/parse cost. This is why ~3.3M raw records become ~340k after filtering. |
| **Auth** | Free Census API key | Via `CENSUS_API_KEY` env var or a local `census_api_key.txt`. |

The assembled national frame is cached to `./data` as Parquet so re-runs skip the
network. The geography column `state` (FIPS) is renamed to `ST`; all JSON-string
values are coerced to numeric (blanks → `NaN`).

**Why this dataset exists:** the prior `new_dataset.csv` (9 features) was
signal-limited — every baseline converged at **R² ≈ 0.30**. This extraction adds
genuinely new predictors (industry, weeks worked, geography, nativity, English
ability, disability) plus a clean target population to try to break that ceiling.

## 2. How it was preprocessed

The pipeline (extraction notebook, Stages 2–5):

1. **Variable-name resolution** — handles ACS year-to-year renames:
   `RELSHIPP` (2019+) over the discontinued `RELP`; `WKWN` (continuous weeks worked)
   over the older `WKW` recode.
2. **Inflation adjustment** — `WAGP × (ADJINC / 1e6)` rescales wages to constant
   dollars. *(This is also the step that leaves the target in its normalized
   `[0, 0.92]` range rather than raw dollars.)*
3. **Population filter** — the high-impact step, restricting to employed,
   working-age earners. `before → after`:
   - `ESR ∈ {1, 2}` — civilian employed (at work / with a job)
   - `18 ≤ AGEP ≤ 64` — working age
   - `WKHP ≥ 1` — works at least 1 hour/week
   - `WAGP > 0` — positive wages
4. **Missing-value handling** — categoricals filled with sentinel `-1`
   ("not applicable" as its own category); continuous filled with the column median.
5. **Typing** — categoricals cast to integer codes for native categorical learners
   (CatBoost).
6. **Helper columns dropped** — `ESR`, `PWGTP`, `ADJINC` are used for filtering/adjustment, then removed (not features).
7. **Save** — feature columns + `WAGP` written to `acs2024_income.csv`.

## 3. Selected features

18 features survive to the CSV. **3 are genuinely continuous**; the other **15 are
PUMS coded variables** — integers that name categories, not magnitudes.

### Continuous

| Feature | Meaning | Notes |
|---|---|---|
| `AGEP` | Age in years | Clipped to 18–64 by the population filter |
| `WKHP` | Usual hours worked per week | 1–99 |
| `WKWN` | Weeks worked in the past 12 months | 1–52 (continuous since 2019) |

### Categorical / coded

| Feature | Meaning | Cardinality |
|---|---|---|
| `COW` | Class of worker (private, govt, self-employed…) | 8 |
| `SCHL` | Educational attainment (ordinal, 1–24) | 24 |
| `MAR` | Marital status | 5 |
| `OCCP` | Occupation code | **525** |
| `INDP` | Industry code | **257** |
| `POBP` | Place of birth | **222** |
| `RELSHIPP` | Relationship to householder | codes 20–38 |
| `SEX` | 1 = male, 2 = female | 2 |
| `RAC1P` | Race (recode) | 9 |
| `HISP` | Hispanic origin | 24 |
| `CIT` | Citizenship status | 5 |
| `NATIVITY` | 1 = native, 2 = foreign-born | 2 |
| `ENG` | Ability to speak English (0 = N/A, only-English household) | 5 |
| `DIS` | 1 = with disability, 2 = without | 2 |
| `ST` | State FIPS code | 51 |

**Target — `WAGP`:** wages/salary income. Chosen over `PINCP` (total income)
because it has fewer negatives and no capital-gains noise. Normalized — see the
note at the top.

## 4. Feature distributions

### Target (`WAGP`)

Strongly **right-skewed** (skewness ≈ **+4.05**), as wage data always is — which is
why the modeling notebooks fit `log1p(WAGP)`.

| pct | 1% | 5% | 25% | 50% | 75% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| value | 0.0012 | 0.0059 | 0.0305 | **0.0528** | 0.0914 | 0.203 | 0.517 | 0.921 |
| ≈ \$ | \$1.2k | \$5.9k | \$30.5k | **\$52.8k** | \$91.4k | \$203k | \$517k | \$921k |

### Continuous features

| Feature | mean | std | min | 25% | 50% | 75% | max | Shape |
|---|---|---|---|---|---|---|---|---|
| `AGEP` | 41.3 | 13.0 | 18 | 30 | 41 | 53 | 64 | ~Symmetric/flat (skew ≈ 0), bounded by the 18–64 filter |
| `WKHP` | 39.3 | 11.5 | 1 | 40 | 40 | 40 | 99 | Sharp spike at 40 (**51.6%** work exactly 40 hrs); part-time left tail |
| `WKWN` | 49.0 | 9.2 | 1 | 52 | 52 | 52 | 52 | Heavily **left-skewed** (≈ −3.4); **84.8%** worked the full 52 weeks |

### Categorical features — notable shares

| Feature | Distribution |
|---|---|
| `SEX` | 51.1% male / 48.9% female |
| `DIS` | 93.1% no disability / 6.9% with |
| `NATIVITY` | 84.0% native / 16.0% foreign-born |
| `CIT` | 82.3% born in US, 8.8% naturalized, 7.2% non-citizen, rest born abroad |
| `MAR` | 54.4% married, 33.8% never married, 9.2% divorced, remainder widowed/separated |
| `COW` | 68.2% private for-profit, ~19% government (local/state/federal), ~5% self-employed |
| `ENG` | **Zero-inflated**: 78.1% code 0 (only-English household; question is only asked of others). Skew ≈ +2.6 |
| `HISP` | 83.2% non-Hispanic (code 1). Long thin tail → skew ≈ +5.3 (highest of any column) |
| `RAC1P` | 65.9% White (code 1) |

**High-cardinality coded fields:** `OCCP` (525 distinct), `INDP` (257), `POBP`
(222), `ST` (51). Their numeric codes are **nominal labels, not magnitudes** — treat
them as categorical.

## 5. Correlations

### With the target

Computed two ways because most features are coded: **Pearson** assumes the code
order is linear/meaningful (only fair for the truly ordinal/continuous fields);
**Spearman** (rank) is more robust but still mis-reads purely nominal codes.

| Feature | Pearson r | Spearman ρ | Read |
|---|---|---|---|
| `WKHP` | **+0.319** | **+0.510** | More hours → higher wage. Strongest signal. |
| `SCHL` | +0.279 | +0.430 | Education (genuinely ordinal) → higher wage. |
| `OCCP` | −0.278 | −0.363 | Code-ordering artifact (nominal); real effect is large but non-linear. |
| `MAR` | −0.234 | −0.320 | Higher code = never-married → lower wage. |
| `RELSHIPP` | −0.216 | −0.327 | Life-stage proxy. |
| `AGEP` | +0.208 | +0.293 | Experience/age premium. |
| `WKWN` | +0.201 | +0.361 | More weeks worked → higher annual wage. |
| `SEX` | −0.145 | −0.171 | Code 2 = female → lower wage (the wage gap). |
| all others (`ENG`,`DIS`,`RAC1P`,`HISP`,`INDP`,`NATIVITY`,`ST`,`COW`,`CIT`,`POBP`) | \|r\| < 0.07 | \|ρ\| ≤ 0.13 | Weak *linear* signal; tree models still extract value from `INDP`/`OCCP`/`ST`. |

> ⚠️ Low linear correlation ≠ low importance. `OCCP`, `INDP`, `POBP`, `ST`, `RAC1P`,
> and `HISP` are nominal codes — their relationship to wage is non-monotonic, so
> Pearson/Spearman understate them. CatBoost captures these natively.

### Between features (collinearity clusters)

The strongest inter-feature pairs form three clusters:

**Immigration / nativity (near-redundant — strongest in the dataset):**

| Pair | r |
|---|---|
| `CIT` ↔ `NATIVITY` | **+0.97** |
| `POBP` ↔ `CIT` | +0.92 |
| `POBP` ↔ `NATIVITY` | +0.90 |
| `CIT` ↔ `ENG` | +0.68 |
| `NATIVITY` ↔ `ENG` | +0.66 |
| `POBP` ↔ `ENG` | +0.63 |
| `RAC1P` ↔ `ENG` | +0.49 |

These six variables largely encode the same foreign-born signal; expect heavy
redundancy.

**Life stage:**

| Pair | r |
|---|---|
| `AGEP` ↔ `MAR` | −0.48 |
| `MAR` ↔ `RELSHIPP` | +0.41 |
| `AGEP` ↔ `RELSHIPP` | −0.36 |

**Work / education:**

| Pair | r |
|---|---|
| `SCHL` ↔ `OCCP` | −0.39 |
| `WKHP` ↔ `WKWN` | +0.28 |
| `OCCP` ↔ `INDP` | −0.28 |
| `RAC1P` ↔ `HISP` | +0.34 |

## 6. Modeling relevance

- The biggest single predictors are **labor-supply** (`WKHP`, `WKWN`) and
  **human-capital** (`SCHL`, `AGEP`, `OCCP`) variables — consistent with standard
  wage models.
- The immigration cluster is highly collinear; most of its members are redundant.
- Despite the richer feature set, the benchmark notebooks plateau around
  **R² ≈ 0.51** — the remaining ceiling appears to be **feature-bound**, not
  model-bound (see the exp1–exp6 family, where the two best configs tie at ~0.5085).
- Always report **R²** for this target; the normalized scale makes raw MAE
  misleading.
