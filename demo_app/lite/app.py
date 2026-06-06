"""
Project Echo - Income Predictor (Lite / 3-booster blend)
=========================================================
Interactive tkinter demo over the ACS 2024 PUMS income model (Experiment 4).

This LITE app bundles the three gradient-boosted base learners from the exp4
ensemble -- XGBoost + LightGBM + CatBoost -- and averages them. It is a faithful,
lightweight stand-in for the full 6.7 GB stacking ensemble: the boosters each
score Test R2 ~0.51 and their blend ~0.5125, statistically identical to the full
Stacking(GBMs) model (0.5154), because the wage signal is feature-bound.

Model bundle (model.pkl) is loaded via joblib. Target was trained on log1p(WAGP)
where WAGP is in millions of constant dollars, so predictions are inverted with
expm1 and multiplied by 1e6 to report whole dollars.
"""
import os
import sys
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

warnings.filterwarnings("ignore")   # silence ML-lib chatter so it never reaches the GUI

# --- Locate model.pkl -- works as a plain script and as a PyInstaller .exe -----
def _resolve_model(filenames):
    """Find the model bundle. Looks NEXT TO the .exe / script first (the
    recommended 'drop model.pkl beside the app' layout), then other fallbacks."""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(sys.executable))     # beside the .exe  <-- primary
        dirs.append(sys._MEIPASS)                        # type: ignore[attr-defined]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        dirs += [here, os.path.join(here, "..", "..")]   # beside app.py, then repo root
    cands = [os.path.join(d, f) for d in dirs for f in filenames]
    return next((p for p in cands if os.path.exists(p)), cands[0])

MODEL_PATH = _resolve_model(["model.pkl"])


# =============================================================================
# ACS PUMS 2024 option labels  (code -> human-readable label)
# Codes are the raw values the model expects; labels are what the GUI shows.
# =============================================================================

# COW - Class of worker (employment sector)
COW_LABELS = {
    1: "Private, for-profit company",
    2: "Private, non-profit organization",
    3: "Local government employee",
    4: "State government employee",
    5: "Federal government employee",
    6: "Self-employed (own business, not incorporated)",
    7: "Self-employed (own incorporated business)",
    8: "Working without pay in family business/farm",
}

# SCHL - Educational attainment
SCHL_LABELS = {
    1: "No schooling completed", 2: "Nursery / preschool", 3: "Kindergarten",
    4: "Grade 1", 5: "Grade 2", 6: "Grade 3", 7: "Grade 4", 8: "Grade 5",
    9: "Grade 6", 10: "Grade 7", 11: "Grade 8", 12: "Grade 9", 13: "Grade 10",
    14: "Grade 11", 15: "12th grade - no diploma", 16: "High school diploma",
    17: "GED / alternative credential", 18: "Some college, < 1 year",
    19: "Some college, 1+ years, no degree", 20: "Associate's degree",
    21: "Bachelor's degree", 22: "Master's degree",
    23: "Professional degree (beyond bachelor's)", 24: "Doctorate degree",
}

# MAR - Marital status
MAR_LABELS = {
    1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
    5: "Never married",
}

# RELSHIPP - Relationship to the householder (reference person)
RELSHIPP_LABELS = {
    20: "Reference person (householder)", 21: "Opposite-sex spouse",
    22: "Opposite-sex unmarried partner", 23: "Same-sex spouse",
    24: "Same-sex unmarried partner", 25: "Biological child", 26: "Adopted child",
    27: "Stepchild", 28: "Sibling", 29: "Parent", 30: "Grandchild",
    31: "Parent-in-law", 32: "Child-in-law", 33: "Other relative",
    34: "Roommate / housemate", 35: "Foster child", 36: "Other non-relative",
    38: "Institutionalized group-quarters resident",
}

# SEX
SEX_LABELS = {1: "Male", 2: "Female"}

# RAC1P - Race (recoded, single)
RAC1P_LABELS = {
    1: "White alone", 2: "Black / African American alone",
    3: "American Indian alone", 4: "Alaska Native alone",
    5: "American Indian / Alaska Native (other)", 6: "Asian alone",
    7: "Native Hawaiian / Pacific Islander alone", 8: "Some other race alone",
    9: "Two or more races",
}

# HISP - Hispanic origin
HISP_LABELS = {
    1: "Not Hispanic / Latino", 2: "Mexican", 3: "Puerto Rican", 4: "Cuban",
    5: "Dominican", 6: "Costa Rican", 7: "Guatemalan", 8: "Honduran",
    9: "Nicaraguan", 10: "Panamanian", 11: "Salvadoran", 12: "Other Central American",
    13: "Argentinean", 14: "Bolivian", 15: "Chilean", 16: "Colombian",
    17: "Ecuadorian", 18: "Paraguayan", 19: "Peruvian", 20: "Uruguayan",
    21: "Venezuelan", 22: "Other South American", 23: "Spaniard",
    24: "Other Spanish / Hispanic / Latino",
}

# CIT - Citizenship status
CIT_LABELS = {
    1: "Born in the United States",
    2: "Born in U.S. territory (PR, Guam, USVI, etc.)",
    3: "Born abroad to U.S. citizen parent(s)",
    4: "U.S. citizen by naturalization",
    5: "Not a U.S. citizen",
}

# NATIVITY
NATIVITY_LABELS = {1: "Native", 2: "Foreign-born"}

# ENG - Ability to speak English
ENG_LABELS = {
    0: "Speaks only English (N/A)", 1: "Very well", 2: "Well",
    3: "Not well", 4: "Not at all",
}

# DIS - Disability status
DIS_LABELS = {1: "With a disability", 2: "Without a disability"}

# ST / POBP US births - state FIPS codes
STATE_FIPS = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}
ST_LABELS = dict(STATE_FIPS)

# OCCP - Occupation (high-cardinality, 525 codes). Curated common labels; any code
# not listed still appears in the dropdown as a bare code (all valid codes come
# from the model's own encoding maps at load time).
OCCP_LABELS = {
    10: "Chief executives", 20: "General & operations managers",
    800: "Accountants & auditors", 1021: "Software developers",
    2100: "Lawyers", 2205: "Postsecondary teachers",
    2310: "Elementary & middle school teachers", 3090: "Physicians",
    3255: "Registered nurses", 4220: "Janitors & building cleaners",
    4700: "First-line supervisors, retail sales", 4720: "Cashiers",
    6230: "Carpenters", 6260: "Construction laborers",
    7200: "Automotive technicians & mechanics",
    9130: "Driver / sales workers & truck drivers",
}

# INDP - Industry (high-cardinality, 257 codes). Curated common labels.
INDP_LABELS = {
    170: "Crop production", 770: "Construction",
    7860: "Elementary & secondary schools",
    8680: "Restaurants & other food services",
}

# POBP - Place of birth (high-cardinality). US births reuse the state FIPS map;
# a few common foreign birthplaces are labeled too.
POBP_LABELS = {code: f"{name} (USA)" for code, name in STATE_FIPS.items()}
POBP_LABELS.update({
    207: "China", 210: "India", 215: "Philippines", 303: "Mexico",
})


# Continuous fields: (label, from, to, default)
CONTINUOUS_FIELDS = {
    "AGEP": ("Age (years)", 18, 64, 41),
    "WKHP": ("Usual hours worked / week", 1, 99, 40),
    "WKWN": ("Weeks worked in past year", 1, 52, 52),
}

# Low-cardinality categoricals: (label, {code: label}, default_code)
LOWCARD_FIELDS = {
    "COW":      ("Class of worker", COW_LABELS, 1),
    "SCHL":     ("Education", SCHL_LABELS, 21),
    "MAR":      ("Marital status", MAR_LABELS, 1),
    "RELSHIPP": ("Relationship to householder", RELSHIPP_LABELS, 20),
    "SEX":      ("Sex", SEX_LABELS, 1),
    "RAC1P":    ("Race", RAC1P_LABELS, 1),
    "HISP":     ("Hispanic origin", HISP_LABELS, 1),
    "CIT":      ("Citizenship", CIT_LABELS, 1),
    "NATIVITY": ("Nativity", NATIVITY_LABELS, 1),
    "ENG":      ("English ability", ENG_LABELS, 0),
    "DIS":      ("Disability", DIS_LABELS, 2),
    "ST":       ("State of residence", ST_LABELS, 6),
}

# High-cardinality categoricals: (label, {code: label}, default_code)
HIGHCARD_FIELDS = {
    "OCCP": ("Occupation (OCCP code)", OCCP_LABELS, 4720),
    "INDP": ("Industry (INDP code)", INDP_LABELS, 8680),
    "POBP": ("Place of birth (POBP code)", POBP_LABELS, 6),
}


# =============================================================================
# Feature engineering -- must reproduce the exp4 notebook exactly.
# =============================================================================
def preprocess(raw: dict, bundle: dict) -> pd.DataFrame:
    """Turn the 18 raw GUI inputs into the model's 32-column feature frame."""
    te = bundle["target_enc_maps"]
    fq = bundle["freq_maps"]
    gm = bundle["global_mean"]
    cat = bundle["cat_features"]
    feat_cols = bundle["features"]

    r = dict(raw)
    # Engineered interaction / Mincer terms
    r["ANNUAL_HOURS"] = r["WKHP"] * r["WKWN"]
    r["AGE2"]         = r["AGEP"] ** 2
    r["SCHLxAGEP"]    = r["SCHL"] * r["AGEP"]

    # Leak-safe target + frequency encodings for the 4 high-card codes
    for c in ("OCCP", "INDP", "POBP", "ST"):
        r[c + "_te"]   = float(te[c].get(r[c], gm))
        r[c + "_freq"] = float(fq[c].get(r[c], 0.0))

    # OCCP x INDP pair target encoding (same key as training)
    pair_key = int(r["OCCP"]) * 100000 + int(r["INDP"])
    r["OCCPINDP_te"] = float(te["_OCCP_INDP"].get(pair_key, gm))

    # Expected occupation/industry pay x labour supply
    r["OCCPte_x_HOURS"] = r["OCCP_te"] * r["ANNUAL_HOURS"]
    r["INDPte_x_HOURS"] = r["INDP_te"] * r["ANNUAL_HOURS"]

    df = pd.DataFrame([r])
    for c in cat:
        df[c] = df[c].astype(int)
    return df.reindex(columns=feat_cols)   # exact columns, exact order


# =============================================================================
# Small UI helpers
# =============================================================================
def _options_to_display(labels: dict, codes):
    """Build 'code - label' strings (label optional) and a parallel code list."""
    display, code_list = [], []
    for code in codes:
        lbl = labels.get(code)
        display.append(f"{code} - {lbl}" if lbl else f"{code}")
        code_list.append(code)
    return display, code_list


# =============================================================================
# Application
# =============================================================================
class App(tk.Tk):
    PAD = dict(padx=8, pady=4)

    def __init__(self):
        super().__init__()
        self.title("Project Echo - Income Predictor (Lite)")
        self.resizable(False, False)
        self.bundle = None
        self.widgets = {}      # field -> (widget, code_list or None)
        self._load_model()
        self._build_ui()

    # ---- model loading (never crash on a bad/missing model) -----------------
    def _load_model(self):
        try:
            self.bundle = joblib.load(MODEL_PATH)
            # rebuild the category dtypes XGBoost/LightGBM need at predict time
            self.cat_dtypes = {
                c: CategoricalDtype(categories=cats)
                for c, cats in self.bundle["cat_categories"].items()
            }
        except Exception as e:
            self.bundle = None
            messagebox.showwarning(
                "Model not loaded",
                f"Could not load model.pkl:\n{e}\n\n"
                "The form still opens, but predictions are disabled.\n"
                f"Expected file: {MODEL_PATH}")

    # ---- UI construction ----------------------------------------------------
    def _build_ui(self):
        # Header banner
        header = tk.Frame(self, bg="#1f3a5f")
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text="Project Echo  -  Socio-Economic Income Predictor",
                 bg="#1f3a5f", fg="white",
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        tk.Label(header, text="2024 ACS PUMS  -  XGBoost + LightGBM + CatBoost blend (Experiment 4)",
                 bg="#1f3a5f", fg="#bcd2ee",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=(0, 10))

        # Form (two columns of fields)
        form = tk.LabelFrame(self, text="  Person attributes  ", font=("Segoe UI", 10, "bold"),
                             padx=10, pady=8)
        form.grid(row=1, column=0, sticky="ew", padx=12, pady=10)

        # Assemble the field order: continuous, then low-card, then high-card.
        ordered = (list(CONTINUOUS_FIELDS) + list(LOWCARD_FIELDS) + list(HIGHCARD_FIELDS))
        half = (len(ordered) + 1) // 2
        for idx, field in enumerate(ordered):
            col_block = 0 if idx < half else 1
            row = idx if idx < half else idx - half
            self._add_field(form, field, row, col_block * 3)

        # Buttons
        btns = tk.Frame(self)
        btns.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        ttk.Button(btns, text="Predict income", command=self._predict).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self._reset).pack(side="left", padx=4)

        # Result panel
        res = tk.LabelFrame(self, text="  Prediction  ", font=("Segoe UI", 10, "bold"),
                            padx=10, pady=8)
        res.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        res.columnconfigure(0, weight=1)
        self.result_var = tk.StringVar(value="--")
        self.detail_var = tk.StringVar(value="Enter attributes and click Predict.")
        self.result_lbl = tk.Label(res, textvariable=self.result_var,
                                    font=("Segoe UI", 22, "bold"), fg="#1f3a5f")
        self.result_lbl.grid(row=0, column=0, sticky="w")
        tk.Label(res, textvariable=self.detail_var, font=("Segoe UI", 9),
                 fg="#444", justify="left").grid(row=1, column=0, sticky="w")

    def _add_field(self, parent, field, row, col):
        """Create one labeled widget for a field and register it."""
        if field in CONTINUOUS_FIELDS:
            label, lo, hi, default = CONTINUOUS_FIELDS[field]
            var = tk.StringVar(value=str(default))
            w = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=22)
            self.widgets[field] = (w, None)
        elif field in LOWCARD_FIELDS:
            label, labels, default = LOWCARD_FIELDS[field]
            codes = sorted(labels)
            display, code_list = _options_to_display(labels, codes)
            w = ttk.Combobox(parent, values=display, state="readonly", width=30)
            w.current(code_list.index(default) if default in code_list else 0)
            self.widgets[field] = (w, code_list)
        else:  # high-cardinality -> editable combobox of all valid codes
            label, labels, default = HIGHCARD_FIELDS[field]
            codes = self._valid_codes(field)
            display, code_list = _options_to_display(labels, codes)
            w = ttk.Combobox(parent, values=display, state="normal", width=30)
            if default in code_list:
                w.current(code_list.index(default))
            elif code_list:
                w.current(0)
            self.widgets[field] = (w, code_list)

        tk.Label(parent, text=label, anchor="w").grid(row=row, column=col, sticky="w", **self.PAD)
        w.grid(row=row, column=col + 1, sticky="ew", **self.PAD)

    def _valid_codes(self, field):
        """Valid codes for a high-card field, taken from the model's own maps."""
        if self.bundle is not None:
            try:
                return sorted(int(v) for v in self.bundle["freq_maps"][field].index)
            except Exception:
                pass
        return sorted(HIGHCARD_FIELDS[field][1])   # fallback: just the labeled ones

    # ---- input collection + validation --------------------------------------
    def _collect_raw(self) -> dict:
        raw = {}
        # continuous
        for field, (label, lo, hi, _d) in CONTINUOUS_FIELDS.items():
            w, _ = self.widgets[field]
            try:
                val = int(float(w.get()))
            except ValueError:
                raise ValueError(f"'{label}' must be a whole number.")
            if not (lo <= val <= hi):
                raise ValueError(f"'{label}' must be between {lo} and {hi}.")
            raw[field] = val
        # low-card -> map combobox index to code
        for field, (label, _labels, _d) in LOWCARD_FIELDS.items():
            w, code_list = self.widgets[field]
            raw[field] = code_list[w.current()]
        # high-card -> parse leading integer from selection or free text
        for field, (label, _labels, _d) in HIGHCARD_FIELDS.items():
            w, _code_list = self.widgets[field]
            text = w.get().strip()
            try:
                raw[field] = int(text.split("-")[0].strip())
            except ValueError:
                raise ValueError(f"'{label}' must be a numeric ACS code.")
        return raw

    # ---- prediction ---------------------------------------------------------
    def _predict(self):
        if self.bundle is None:
            messagebox.showerror("No model", "model.pkl is not loaded; cannot predict.")
            return
        try:
            raw = self._collect_raw()
            X = preprocess(raw, self.bundle)
            Xc = X.copy()
            for c, dt in self.cat_dtypes.items():
                Xc[c] = Xc[c].astype(dt)

            scale = self.bundle["wagp_to_dollars"]
            models = self.bundle["models"]
            # XGBoost & LightGBM need category dtype; CatBoost takes the int frame.
            log_preds = {
                "XGBoost":  float(models["XGBoost"].predict(Xc)[0]),
                "LightGBM": float(models["LightGBM"].predict(Xc)[0]),
                "CatBoost": float(models["CatBoost"].predict(X)[0]),
            }
            dollars = {n: float(np.expm1(lp) * scale) for n, lp in log_preds.items()}
            blend = float(np.expm1(np.mean(list(log_preds.values()))) * scale)
        except ValueError as ve:
            messagebox.showerror("Invalid input", str(ve))
            return
        except Exception as e:
            messagebox.showerror("Prediction failed", f"{type(e).__name__}: {e}")
            return

        band, color = self._band(blend)
        self.result_var.set(f"${blend:,.0f} / year")
        self.result_lbl.config(fg=color)
        breakdown = "   ".join(f"{n} ${v/1000:,.1f}k" for n, v in dollars.items())
        self.detail_var.set(
            f"{band}\nPer-model:  {breakdown}\n"
            f"(3-booster blend of exp4; predicted annual wage/salary income, WAGP)")

    @staticmethod
    def _band(amount):
        if amount < 25_000:
            return "Lower income band", "#c0392b"
        if amount < 50_000:
            return "Lower-middle income band", "#e67e22"
        if amount < 80_000:
            return "Middle income band", "#2980b9"
        if amount < 120_000:
            return "Upper-middle income band", "#16a085"
        return "High income band", "#27ae60"

    # ---- reset --------------------------------------------------------------
    def _reset(self):
        for field, (w, code_list) in self.widgets.items():
            if field in CONTINUOUS_FIELDS:
                w.delete(0, "end")
                w.insert(0, str(CONTINUOUS_FIELDS[field][3]))
            elif field in LOWCARD_FIELDS:
                default = LOWCARD_FIELDS[field][2]
                w.current(code_list.index(default) if default in code_list else 0)
            else:
                default = HIGHCARD_FIELDS[field][2]
                w.set(f"{default} - {HIGHCARD_FIELDS[field][1].get(default, '')}".rstrip(" -")
                      if default in code_list else (w["values"][0] if w["values"] else ""))
        self.result_var.set("--")
        self.result_lbl.config(fg="#1f3a5f")
        self.detail_var.set("Enter attributes and click Predict.")


if __name__ == "__main__":
    App().mainloop()
