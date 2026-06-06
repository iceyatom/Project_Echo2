"""
make_model.py  --  generate the LITE app's slim model.pkl.

The lite app only needs the three gradient boosters (XGBoost, LightGBM, CatBoost)
plus the shared preprocessing metadata -- not the 6.7 GB RandomForest/ExtraTrees.
This script loads the full exp4 bundle produced by the modeling notebook and writes
a ~60 MB `model.pkl` next to itself.

Run it ONCE after the notebook has produced the full bundle at the repo root:

    cd demo_app/lite
    python make_model.py

It expects ../../acs2024v1.1_income_models_exp4.pkl to exist (or pass a path:
    python make_model.py path\to\acs2024v1.1_income_models_exp4.pkl
).
"""
import os
import sys
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


# The full bundle pickles a StackedGBM instance referenced as __main__.StackedGBM,
# so the class must exist here for joblib to reconstruct the bundle on load.
class StackedGBM:
    def __init__(self, base_models, cat_dtypes, meta=None):
        self.base_models = base_models
        self.cat_dtypes = cat_dtypes
        self.meta = meta if meta is not None else LinearRegression()

    def _stack(self, X):
        Xc = X.copy()
        for c, dt in self.cat_dtypes.items():
            Xc[c] = Xc[c].astype(dt)
        preds = [m.predict(Xc if need_cat else X) for _, m, need_cat in self.base_models]
        return np.column_stack(preds)

    def fit(self, X_blend, y_blend):
        self.meta.fit(self._stack(X_blend), y_blend)
        return self

    def predict(self, X):
        return self.meta.predict(self._stack(X))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.join(here, "..", "..", "acs2024v1.1_income_models_exp4.pkl")
    src = sys.argv[1] if len(sys.argv) > 1 else default_src
    out = os.path.join(here, "model.pkl")

    if not os.path.exists(src):
        sys.exit(f"ERROR: full bundle not found:\n  {src}\n"
                 "Run the modeling notebook (acs2024v1.1_exp4_model.ipynb) first, "
                 "or pass the path as an argument.")

    sys.modules["__main__"].StackedGBM = StackedGBM   # ensure unpickle resolves the class
    print(f"Loading full bundle: {src}")
    b = joblib.load(src)
    stack = b["models"]["Stacking(GBMs)"]

    # Persist the boosters' training category sets so the lite app can rebuild the
    # pandas CategoricalDtype that XGBoost/LightGBM need at predict time.
    cat_categories = {c: [int(x) for x in dt.categories.tolist()]
                      for c, dt in stack.cat_dtypes.items()}

    slim = {
        "models": {k: b["models"][k] for k in ("XGBoost", "LightGBM", "CatBoost")},
        "best_single": "LightGBM",
        "blend_models": ["XGBoost", "LightGBM", "CatBoost"],
        "features": b["features"],
        "cat_features": b["cat_features"],
        "cat_categories": cat_categories,
        "target": b["target"],
        "target_transform": b["target_transform"],
        "wagp_to_dollars": b["wagp_to_dollars"],
        "generated_features": b["generated_features"],
        "encoding_cols": b["encoding_cols"],
        "target_enc_maps": b["target_enc_maps"],
        "freq_maps": b["freq_maps"],
        "global_mean": b["global_mean"],
        "smooth": b["smooth"],
        "pair_smooth": b["pair_smooth"],
        "survey_year": b["survey_year"],
        "note": "3-booster blend (XGB+LGBM+CatBoost) extracted from exp4",
    }
    joblib.dump(slim, out, compress=3)
    print(f"Wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
