# `acs2024v2_income.csv` — Dataset Description

A person-level wage/income dataset built from the **2024 American Community Survey
(ACS) Public Use Microdata Sample (PUMS), 1-Year file**. Produced by
[acs2024v2_income_extraction.ipynb](acs2024v2_income_extraction.ipynb).

**This is v2 of [`acs2024_income.csv`](acs2024_income.csv)** — the *identical* extraction
pipeline with one new feature added: **`PUMA`** (sub-state geography), to capture local
labor-market and cost-of-living effects that the coarse `ST` (state) field cannot. It is
statistically equivalent to v1 (every shared-column mean within ±0.4%, identical target
scaling) but **not row-for-row identical**: a fresh API pull reorders records and the
`sample(frac=0.25, random_state=42)` draw is order-dependent, so the deterministic 25%
landed on a different — equally representative — subset (+121 rows).

**At a glance**

| | |
|---|---|
| Rows (after all filtering) | **339,848** |
| Columns | **20** — 19 features + 1 target |
| Continuous features | 3 (`AGEP`, `WKHP`, `WKWN`) |
| Categorical / coded features | 16 (**`PUMA` is the new one**) |
| Target | `WAGP` (wages, normalized — see note) |
| Missing values | **0** |
| New vs. v1 | `PUMA` column added (geography → **2,462** distinct localities vs. 51 states) |

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
| **Columns requested** | ~22 variables only | The `get=` parameter pulls only the columns used (now incl. `PUMA`), not the full ~285-column PUMS — transfer is tens of MB. |
| **Row density** | `DENSITY = 0.25` | Keeps a 25% random subsample (`random_state=42`) — nationally representative but ~¼ the RAM/parse cost. This is why ~3.3M raw records become ~340k. |
| **Cache** | `data/acs2024_pums_api_all_v2.parquet` | Separate from v1's cache, which predated `PUMA`, forcing a fresh pull that includes it. |

The assembled national frame is cached to `./data` as Parquet so re-runs skip the
network. The geography column `state` (FIPS) is renamed to `ST`; all JSON-string
values are coerced to numeric (blanks → `NaN`).

**Why v2 exists:** the v1 feature set plateaued at **R² ≈ 0.51** in the benchmark
notebooks — the ceiling appeared *feature-bound*. `PUMA` was identified as the
highest-impact missing predictor: it's the only candidate that applies to **every
row** and fills the single largest structural gap (geography finer than state).

## 2. How it was preprocessed

The pipeline (extraction notebook, Stages 2–5) — unchanged from v1 except the added column:

1. **Variable-name resolution** — handles ACS year-to-year renames:
   `RELSHIPP` (2019+) over the discontinued `RELP`; `WKWN` (continuous weeks worked)
   over the older `WKW` recode.
2. **Inflation adjustment** — `WAGP × (ADJINC / 1e6)` rescales wages. *(This is also
   the step that leaves the target in its normalized `[0, 0.92]` range rather than raw
   dollars.)*
3. **Population filter** — the high-impact step, restricting to employed,
   working-age earners:
   - `ESR ∈ {1, 2}` — civilian employed (at work / with a job)
   - `18 ≤ AGEP ≤ 64` — working age
   - `WKHP ≥ 1` — works at least 1 hour/week
   - `WAGP > 0` — positive wages
4. **Missing-value handling** — categoricals (incl. `PUMA`) filled with sentinel `-1`
   ("not applicable"); continuous filled with the column median.
5. **Typing** — categoricals cast to integer codes for native categorical learners
   (CatBoost).
6. **Helper columns dropped** — `ESR`, `PWGTP`, `ADJINC` are used for filtering/adjustment, then removed (not features).
7. **Save** — feature columns + `WAGP` written to `acs2024v2_income.csv`.

## 3. Selected features

19 features survive to the CSV. **3 are genuinely continuous**; the other **16 are
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
| `OCCP` | Occupation code | 525 |
| `INDP` | Industry code | 257 |
| `POBP` | Place of birth | 222 |
| **`PUMA`** | **Public Use Microdata Area — sub-state geography (NEW)** | **1,150 codes / 2,462 localities** |
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

Strongly **right-skewed** (skewness ≈ **+4.08**), as wage data always is — which is
why the modeling notebooks fit `log1p(WAGP)`.

| pct | 1% | 5% | 25% | 50% | 75% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| value | 0.0012 | 0.0060 | 0.0305 | **0.0528** | 0.0904 | 0.203 | 0.517 | 0.921 |
| ≈ \$ | \$1.2k | \$6.0k | \$30.5k | **\$52.8k** | \$90.4k | \$203k | \$517k | \$921k |

### Continuous features

| Feature | mean | std | min | 25% | 50% | 75% | max | Shape |
|---|---|---|---|---|---|---|---|---|
| `AGEP` | 41.4 | 13.1 | 18 | 30 | 41 | 53 | 64 | ~Symmetric/flat (skew ≈ 0), bounded by the 18–64 filter |
| `WKHP` | 39.3 | 11.5 | 1 | 40 | 40 | 40 | 99 | Sharp spike at 40 (**51.5%** work exactly 40 hrs); part-time left tail |
| `WKWN` | 49.0 | 9.2 | 1 | 52 | 52 | 52 | 52 | Heavily **left-skewed** (≈ −3.4); **84.8%** worked the full 52 weeks |

### Categorical features — notable shares

| Feature | Distribution |
|---|---|
| `SEX` | 51.1% male / 48.9% female |
| `DIS` | 93.1% no disability / 6.9% with |
| `NATIVITY` | 84.1% native / 15.9% foreign-born |
| `CIT` | 82.4% born in US, 8.7% naturalized, 7.2% non-citizen, rest born abroad |
| `MAR` | 54.5% married, 33.7% never married, 9.2% divorced, remainder widowed/separated |
| `COW` | 68.1% private for-profit, ~19% government (local/state/federal), ~5% self-employed |
| `ENG` | **Zero-inflated**: 78.2% code 0 (only-English household; question is only asked of others). Skew ≈ +2.6 |
| `HISP` | 83.2% non-Hispanic (code 1). Long thin tail → skew ≈ +5.3 (highest of the demographic columns) |
| `RAC1P` | 65.8% White (code 1) |

**High-cardinality coded fields:** `PUMA` (see §6), `OCCP` (525 distinct), `INDP`
(257), `POBP` (222), `ST` (51). Their numeric codes are **nominal labels, not
magnitudes** — treat them as categorical.

## 5. Correlations

### With the target

Computed two ways because most features are coded: **Pearson** assumes the code
order is linear/meaningful (only fair for the truly ordinal/continuous fields);
**Spearman** (rank) is more robust but still mis-reads purely nominal codes.

| Feature | Pearson r | Spearman ρ | Read |
|---|---|---|---|
| `WKHP` | **+0.320** | **+0.509** | More hours → higher wage. Strongest signal. |
| `SCHL` | +0.276 | +0.428 | Education (genuinely ordinal) → higher wage. |
| `OCCP` | −0.277 | −0.363 | Code-ordering artifact (nominal); real effect is large but non-linear. |
| `MAR` | −0.233 | −0.320 | Higher code = never-married → lower wage. |
| `RELSHIPP` | −0.216 | −0.326 | Life-stage proxy. |
| `AGEP` | +0.208 | +0.293 | Experience/age premium. |
| `WKWN` | +0.201 | +0.361 | More weeks worked → higher annual wage. |
| `SEX` | −0.147 | −0.171 | Code 2 = female → lower wage (the wage gap). |
| **`PUMA`** | **+0.021** | **+0.028** | **≈ 0 — but this is expected for a nominal geographic code, not "no signal." See §6.** |
| all other (`ENG`,`DIS`,`RAC1P`,`HISP`,`INDP`,`COW`,`NATIVITY`,`ST`,`CIT`,`POBP`) | \|r\| < 0.07 | \|ρ\| ≤ 0.13 | Weak *linear* signal; tree models still extract value from `INDP`/`OCCP`/`ST`/`PUMA`. |

> ⚠️ Low linear correlation ≠ low importance. `PUMA`, `OCCP`, `INDP`, `POBP`, `ST`,
> `RAC1P`, and `HISP` are nominal codes — their relationship to wage is non-monotonic,
> so Pearson/Spearman understate them. CatBoost captures these natively.

### Between features (collinearity clusters)

The strongest inter-feature pairs are essentially unchanged from v1 — three clusters,
plus `PUMA` standing largely on its own.

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

These variables largely encode the same foreign-born signal; expect heavy redundancy.

**Life stage:** `AGEP` ↔ `MAR` −0.48, `MAR` ↔ `RELSHIPP` +0.41, `AGEP` ↔ `RELSHIPP` −0.36.

**Work / education:** `SCHL` ↔ `OCCP` −0.39, `WKHP` ↔ `WKWN` +0.28, `OCCP` ↔ `INDP` −0.28, `RAC1P` ↔ `HISP` +0.34.

**`PUMA`** is **nearly orthogonal to everything** — its only non-trivial linear pairing
is `PUMA` ↔ `ST` (+0.18, since PUMA codes are loosely state-correlated). That low
redundancy is exactly why it's a promising *new* signal rather than a restatement of
existing columns.

## 6. The `PUMA` feature (new in v2)

`PUMA` (Public Use Microdata Area) resolves geography from 51 states down to ~2,400
local areas of ~100k people each — the finest geography ACS PUMS publishes.

**Two numbers that matter:**

- **1,150** distinct integer values, but **2,462** distinct `(ST, PUMA)` localities.
  The gap is because **PUMA codes are unique only *within* a state** — `00100` is
  reused by many states, so as a bare integer they collide. The *real* geographic key
  is the **`(ST, PUMA)` pair**.

**It carries real wage signal that its ≈0 Pearson hides.** Average wage by geography:

| Grouping | # areas | std of area-mean `WAGP` | range of area means |
|---|---|---|---|
| By **state** (`ST`) | 51 | 0.0126 | 0.055 – 0.124 (≈ \$55k – \$124k) |
| By **PUMA locality** (`ST,PUMA`, ≥50 obs) | 2,450 | **0.0258** | **0.032 – 0.220 (≈ \$32k – \$220k)** |

Local wages vary about **twice as widely** across PUMAs as across states — that spread
is invisible to a linear correlation (the integer code has no ordinal meaning) but is
precisely what a tree model can exploit.

**How to use it (important):**

- Treat `PUMA` as **categorical, alongside `ST`** — never as a number, never one-hot
  (2,400+ levels). CatBoost handles it natively.
- Better still, build a **combined locality key** (e.g. `ST*100000 + PUMA`) so that
  San Jose and rural Mississippi never share a code.
- A handful of high codes (>70000, all in Virginia / `ST=51`) are legitimate PUMA
  numbers, not corruption.

## 7. Modeling relevance

- The biggest *linear* predictors remain **labor-supply** (`WKHP`, `WKWN`) and
  **human-capital** (`SCHL`, `AGEP`, `OCCP`) variables — standard wage-model territory.
- **`PUMA` is the headline addition** and the reason v2 exists: it injects local
  geographic wage variation (~2× the state-level spread) that was previously absent.
  Its payoff shows up only in models that handle categoricals natively — judge it by
  ablation R², never by its correlation coefficient.
- The immigration cluster is highly collinear; most of its members are redundant.
- **Recommended next step:** run an **ablation** — train the benchmark model on the v1
  feature set vs. v1 + `PUMA` (or + the `(ST,PUMA)` key) and compare R², so the
  geography gain is cleanly attributable.
- Always report **R²** for this target; the normalized scale makes raw MAE misleading.
