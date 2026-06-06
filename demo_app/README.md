# Project Echo — Income Predictor Demo Apps

Two standalone Windows desktop demos that load the **ACS 2024 PUMS income model
(Experiment 4)** and predict an individual's annual wage/salary income (`WAGP`)
from 18 socio-economic, demographic, and geographic inputs. Built with `tkinter`;
the trained pipeline is loaded from a `.pkl` via `joblib`.

There are **two builds of the same demo**, because the full exp4 model is large:

| App | Folder | Model it runs | `model.pkl` size | Ships as | Launch | RAM |
|---|---|---|---|---|---|---|
| **Lite** (recommended) | [`lite/`](lite/) | 3-booster blend: XGBoost + LightGBM + CatBoost, averaged | ~60 MB | `.exe` (~330 MB) + `model.pkl` | instant | < 1 GB |
| **Full** | [`full/`](full/) | the actual winning **Stacking(GBMs)** ensemble — all 6 tree learners + Ridge meta | ~6.7 GB | `.exe` folder (~350 MB) + `model.pkl` | ~10 s | ~8 GB |

> **Why two?** ~6.4 GB of the exp4 bundle is the RandomForest + ExtraTrees base
> learners. The **Lite** app drops them and keeps the three gradient boosters,
> which the meta-learner already leans on the most. Accuracy is essentially
> identical — Test R² ≈ **0.51** (boosters) vs **0.5154** (full stack) — because
> the wage signal is feature-bound. The Lite app is the practical, distributable
> demo; the Full app is for showing the literal final model.

Both apps share the same GUI, the same 18 input fields, and the same feature
engineering. They differ only in which model runs inside.

In both apps the model is a **separate `model.pkl` that sits in the same folder as
the `.exe`** (it is *not* baked into the executable). Build the `.exe` once, drop
the matching `.pkl` next to it, and ship the two together.

---

## 0. Getting started from a GitHub clone (read this first)

The repository does **not** include the dataset or the trained model — both are
git-ignored (`*.csv`, `*.pkl`) because they are large and regenerable. After
cloning, you must produce them yourself before the apps can run. Full pipeline:

```powershell
# 1. Clone and enter the project
git clone https://github.com/iceyatom/Project_Echo2.git
cd Project_Echo2

# 2. Create the environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # repo-root reqs (notebooks + apps)

# 3. Provide a free Census API key (see the repo-root README "Data Sourcing")
#    either set $env:CENSUS_API_KEY = "your-key"  OR  create census_api_key.txt

# 4. Run the EXTRACTION notebook end-to-end -> writes acs2024v1.1_income.csv
#    acs2024v1.1_income_extraction.ipynb   (pulls 2024 ACS PUMS from the API)

# 5. Run the MODELING notebook end-to-end -> writes the trained bundle
#    acs2024v1.1_exp4_model.ipynb
#    => produces  acs2024v1.1_income_models_exp4.pkl  (~6.7 GB, at the repo root)
```

Now you have the **full model bundle** at the repo root. Then, per app:

**Full app** — uses that 6.7 GB bundle directly:
```powershell
# run from source (loads the pkl from the repo root automatically):
python demo_app\full\app.py
# …or build the .exe, then copy the pkl next to it (build.bat does this for you):
cd demo_app\full
.\build.bat
# -> dist\ProjectEcho_IncomeFull\ProjectEcho_IncomeFull.exe  + the .pkl beside it
```

**Lite app** — needs a slim 60 MB `model.pkl` derived from the full bundle:
```powershell
cd demo_app\lite
python make_model.py        # reads ..\..\acs2024v1.1_income_models_exp4.pkl -> model.pkl
# run from source:
python app.py
# …or build the .exe (build.bat copies model.pkl next to it for you):
.\build.bat
# -> dist\ProjectEcho_IncomeLite.exe  + model.pkl beside it
```

**The golden rule for running a built `.exe`: the `.exe` and its `model.pkl` must
live in the same folder.** That's all the runtime needs — no Python, no notebooks.

> Shortcut: if you only want to *try* the apps and already have the 6.7 GB pkl,
> skip the notebooks (steps 3–5) and go straight to "run from source".

---

## 1. What is this?

- **Model type:** gradient-boosted / stacked tree ensemble regression.
- **GUI toolkit:** `tkinter` + `ttk` (ships with Python; no external GUI deps).
- **Data source:** 2024 ACS PUMS (American Community Survey, Public Use Microdata
  Sample), pulled from the U.S. Census Data API; trained on **1,359,710** employed
  prime-age (18–64) wage earners.
- **Target:** `WAGP` (wage/salary income). Trained on `log1p(WAGP)` where `WAGP`
  is stored in *millions* of constant dollars (scaled ÷1e6). The apps invert with
  `expm1` and multiply by `1e6` to display whole dollars.

---

## 2. Files

```
demo_app/
├── README.md                  ← this file
├── lite/
│   ├── app.py                 ← Lite GUI (3-booster blend)
│   ├── make_model.py          ← regenerates model.pkl from the full bundle (run once)
│   ├── model.pkl              ← slim bundle (~60 MB), made by make_model.py (git-ignored)
│   ├── requirements.txt
│   └── build.bat              ← builds dist\ProjectEcho_IncomeLite.exe (+ copies model.pkl beside it)
└── full/
    ├── app.py                 ← Full GUI (Stacking ensemble); defines StackedGBM
    ├── requirements.txt
    └── build.bat              ← builds dist\ProjectEcho_IncomeFull\ (+ copies the 6.7 GB pkl beside it)
```

Neither `.pkl` is committed to git, and **neither is embedded in its `.exe`** — the
model is always a separate file that lives **next to the executable** (or next to
`app.py` when running from source):

- **Lite** loads `model.pkl` (the 60 MB slim bundle from `make_model.py`).
- **Full** loads `model.pkl` *or* `acs2024v1.1_income_models_exp4.pkl` (the 6.7 GB
  notebook output) from beside the exe; from source it falls back to the repo root.

---

## 3. Requirements

```powershell
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `scikit-learn` | meta-learner (Ridge), encoders, and the sklearn API the boosters use |
| `xgboost`, `lightgbm`, `catboost` | the gradient-boosted base learners |
| `pandas`, `numpy` | build the model's input frame |
| `joblib` | load the `.pkl` bundle |
| `pyinstaller` | compile the standalone `.exe` |

> **`tkinter` is not a pip package** — it ships with the Python installer (tick
> *“tcl/tk and IDLE”* on Windows; `sudo apt-get install python3-tk` on Debian/Ubuntu).

**Python version:** built and tested on **Python 3.13**. The original
AI_BUILD_GUIDE recommends 3.9–3.11 as the safest range for ML wheels; 3.13 works
here because the project venv already has matching wheels. **Use the same
package versions the model was trained with** — a version-mismatched
`scikit-learn`/`xgboost`/`lightgbm`/`catboost` can fail to unpickle the bundle
(this matters most for the Full app).

---

## 4. Option A — Run from source

Prerequisite: the trained bundle `acs2024v1.1_income_models_exp4.pkl` exists at the
repo root (see §0). For the Lite app, also generate its slim model once.

```powershell
# from the repo root, with the project venv:
.\.venv\Scripts\Activate.ps1
pip install -r demo_app\lite\requirements.txt

# Lite — first build the slim model.pkl, then run:
python demo_app\lite\make_model.py
python demo_app\lite\app.py
# Full (loads the 6.7 GB bundle from the repo root; needs ~8 GB RAM):
python demo_app\full\app.py
```

Enter attributes, click **Predict income**. The Lite app shows the 3-booster
blend plus a per-model breakdown; the Full app shows the Stacking(GBMs) result
plus all six base-model predictions.

---

## 5. Option B — Compile a standalone `.exe`

From inside the chosen app folder, with the venv active:

```powershell
cd demo_app\lite          # or demo_app\full
..\..\.venv\Scripts\Activate.ps1
.\build.bat
```

Each `build.bat` builds the executable **and copies the matching `model.pkl` next
to it** (the model is *not* embedded in the exe). The result is always an
**`.exe` + `model.pkl` pair in the same folder**:

- **Lite** → `demo_app\lite\dist\` containing `ProjectEcho_IncomeLite.exe` (~330 MB)
  **and** `model.pkl` (60 MB). Copy *both* files to any Windows PC; no Python required.
  (`build.bat` runs `make_model.py`'s output copy for you, so generate `model.pkl`
  with `python make_model.py` first.)
- **Full** → `demo_app\full\dist\ProjectEcho_IncomeFull\` containing the exe
  (~350 MB of libraries) **and** the 6.7 GB `acs2024v1.1_income_models_exp4.pkl`.
  Ship the whole folder; run the `.exe` inside it.

The build commands include `--collect-all` for xgboost/lightgbm/catboost/sklearn
because PyInstaller does not auto-detect their native binaries — omitting them
makes the `.exe` import-crash on launch. They also `--exclude-module matplotlib/PIL/
IPython`, which the apps never use at predict time, to keep the build smaller.

---

## 6. Option C — Run over the web (optional)

The same `model.pkl` and `preprocess()` could be wrapped in a **Streamlit** or
**Flask** app for a browser-based demo (`st.number_input` / `st.selectbox` in
place of the tkinter widgets). Not implemented here — just a pointer.

---

## 7. Input fields reference

3 continuous + 15 categorical raw inputs → expanded to the model's 32 engineered
columns (interaction/Mincer terms + leak-safe target & frequency encodings).

| Field | Meaning | Widget | Range / options |
|---|---|---|---|
| `AGEP` | Age (years) | spinbox | 18–64 |
| `WKHP` | Usual hours worked / week | spinbox | 1–99 |
| `WKWN` | Weeks worked in past year | spinbox | 1–52 |
| `COW` | Class of worker | dropdown | 8 sectors |
| `SCHL` | Education | dropdown | 24 levels |
| `MAR` | Marital status | dropdown | 5 |
| `RELSHIPP` | Relationship to householder | dropdown | 18 |
| `SEX` | Sex | dropdown | Male / Female |
| `RAC1P` | Race | dropdown | 9 |
| `HISP` | Hispanic origin | dropdown | 24 |
| `CIT` | Citizenship | dropdown | 5 |
| `NATIVITY` | Native / foreign-born | dropdown | 2 |
| `ENG` | English ability | dropdown | 5 |
| `DIS` | Disability | dropdown | 2 |
| `ST` | State of residence (FIPS) | dropdown | 51 |
| `OCCP` | Occupation code | editable combo | 525 valid codes (type any) |
| `INDP` | Industry code | editable combo | 257 valid codes |
| `POBP` | Place of birth code | editable combo | 222 valid codes |

The three high-cardinality fields (`OCCP`/`INDP`/`POBP`) list every code the model
was trained on (taken from its own encoding maps), with friendly labels on common
ones. You can also type a raw ACS code. Unseen codes fall back to the global mean
target encoding so the app never crashes.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| App opens but says **"model not loaded"** | The `.exe` can't find its `.pkl`. **Put `model.pkl` (Full also accepts `acs2024v1.1_income_models_exp4.pkl`) in the same folder as the `.exe`.** From source, it must be beside `app.py` or at the repo root. |
| Lite: `model.pkl` missing | Generate it: `cd demo_app\lite && python make_model.py` (needs the 6.7 GB bundle at the repo root). |
| Full: no model at repo root | Run the modeling notebook (`acs2024v1.1_exp4_model.ipynb`) end-to-end to regenerate `acs2024v1.1_income_models_exp4.pkl`. |
| `Can't get attribute 'StackedGBM'` | Only affects the Full app; `app.py` already defines `StackedGBM` and registers it on `__main__`. Don't rename the class. |
| `ModuleNotFoundError: xgboost/lightgbm/catboost` | `pip install -r requirements.txt`. |
| `No module named 'tkinter'` | Reinstall Python with the *tcl/tk and IDLE* option (Windows) or `apt-get install python3-tk` (Linux). |
| `.exe` crashes immediately on launch | Rebuild with the `--collect-all` flags in `build.bat` (native booster binaries weren't bundled). |
| Predictions look off / unpickle errors | Use the **same** scikit-learn/xgboost/lightgbm/catboost versions the model was trained with. |
| Full app is slow to open | Normal — it loads ~6.7 GB. Use the **Lite** app for an instant demo. |

---

## 9. How the prediction works

1. Read the 18 raw inputs from the form.
2. `preprocess()` rebuilds the model's 32 features: the raw values, the engineered
   `ANNUAL_HOURS` / `AGE2` / `SCHLxAGEP` terms, the smoothed **target encodings** and
   **frequency encodings** of `OCCP`/`INDP`/`POBP`/`ST` (+ the `OCCP×INDP` pair), and
   the `…_te × ANNUAL_HOURS` interactions — reindexed to the exact training order.
3. Run the model(s) on `log1p(WAGP)`, invert with `expm1`, multiply by `1e6`.
4. **Lite:** average the three boosters' log-predictions. **Full:** feed all six
   base models' predictions to the Ridge meta-learner (the real stack).
5. Bucket the dollar result into a color-coded income band and show a per-model
   breakdown line.
