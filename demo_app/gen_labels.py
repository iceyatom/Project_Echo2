"""
gen_labels.py -- regenerate the full OCCP / INDP code->label dictionaries that
the demo apps embed for their high-cardinality dropdowns.

Source of truth: the official Census ACS PUMS 2024 Data Dictionary (CSV). Every
OCCP (occupation) and INDP (industry) value row is parsed; the Census category
prefix ("MGR-", "CMM-", "AGR-", ...) is stripped so the GUI shows a clean title.

Run:  python demo_app/gen_labels.py
It prints two ready-to-paste Python dict literals (OCCP_LABELS, INDP_LABELS).
"""
import csv
import io
import re
import urllib.request

DATA_DICT_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/"
    "PUMS_Data_Dictionary_2024.csv"
)
PREFIX_RE = re.compile(r"^[A-Z]{2,4}-")        # e.g. "MGR-", "CMM-", "AGR-"


def _clean(label: str) -> str:
    """Strip the Census major-group prefix, leaving a plain occupation/industry title."""
    return PREFIX_RE.sub("", label).strip()


def fetch_labels(var: str) -> dict:
    """Return {int code: clean label} for one PUMS variable (OCCP or INDP)."""
    raw = urllib.request.urlopen(DATA_DICT_URL, timeout=60).read().decode("latin-1")
    out = {}
    for row in csv.reader(io.StringIO(raw)):
        # VAL rows: [VAL, NAME, type, len, min_code, max_code, label]
        if len(row) >= 7 and row[0] == "VAL" and row[1] == var:
            code = row[4].strip()
            if not code.isdigit():            # skip "bbbb" N/A sentinel
                continue
            out[int(code)] = _clean(row[6])
    return dict(sorted(out.items()))


def emit(name: str, mapping: dict) -> str:
    lines = [f"{name} = {{"]
    for code, label in mapping.items():
        safe = label.replace('"', '\\"')
        lines.append(f'    {code}: "{safe}",')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    occp = fetch_labels("OCCP")
    indp = fetch_labels("INDP")
    print(f"# {len(occp)} OCCP codes, {len(indp)} INDP codes "
          "(generated from Census PUMS 2024 Data Dictionary)\n")
    print(emit("OCCP_LABELS", occp))
    print()
    print(emit("INDP_LABELS", indp))
