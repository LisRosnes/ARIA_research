#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse a radiology free-text cell and emit a one-row CSV with the desired fields.

Usage examples:
  python parse_mri_report.py --textfile report.txt --out out.csv
  python parse_mri_report.py --stdin > out.csv
  # or to process a whole Excel column:
  python parse_mri_report.py --xlsx reports.xlsx --sheet Sheet1 --column Report --out out.csv
"""

import re
import csv
import sys
import argparse

COLUMNS = [
    "Apoe",
    "Lecanemab dose",
    "acute hemmorage",
    "Acute/subacute cortical infarct",
    "Chronic hemorrhage",
    "Chronic ischemic changes",
    "MTA-right",
    "MTA-left",
    "ERiCA-right",
    "ERiCA-left",
    "Extra-axial Collection",
    "Ventricular System",
    "Major Intracranial Flow Voids",
    "Included Orbits",
    "Paranasal Sinuses",
    "Typanomastoid Cavaties",
]

# Pass-through columns that live in the source sheet but are not inside the
# report text column. These will be copied verbatim into the output CSV.
COLUMNS += ["ARIA-E", "ARIA-H"]

# --------- helpers ---------
def _clean(s: str) -> str:
    """Clean whitespace and normalize the string."""
    return re.sub(r"\s+", " ", s.strip())

def _sanitize_value(s: str) -> str:
    """
    Additional cleaning to remove problematic characters that might cause CSV issues.
    Removes leading/trailing quotes and normalizes internal quotes.
    """
    if not s:
        return s
    # Remove leading/trailing whitespace
    s = s.strip()
    # Remove leading/trailing quotes that might interfere with CSV quoting
    s = s.strip('"').strip("'")
    # Replace any remaining double quotes with single quotes to avoid CSV confusion
    s = s.replace('"', "'")
    return s

def _first(text: str, patterns, default=""):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            g = next((g for g in m.groups() if g is not None), m.group(0))
            return _sanitize_value(_clean(g))
    return default

def _present_to_yes_no(value: str, default=""):
    if not value:
        return default
    v = value.lower()
    if "not present" in v or v in {"none", "no", "normal", "absent"}:
        return "no"
    if "present" in v or "yes" in v or "mild" in v or "moderate" in v or "severe" in v:
        return "yes"
    return default

def _normalize_grade(s: str) -> str:
    """Return mild/moderate/severe/normal/none text if found, else original short form like MTA 2."""
    if not s:
        return ""
    v = s.lower()
    for word in ["none", "normal", "mild", "moderate", "severe"]:
        if word in v:
            return word
    # keep compact codes like "MTA 2" or "(MTA 2)"
    m = re.search(r"\b(mta|erica)\s*([0-3])\b", v, flags=re.I)
    if m:
        # map numeric to mild/mod/severe-ish if you prefer; here we keep "2" ~ "mild" per your example
        n = int(m.group(2))
        return {0: "normal", 1: "mild", 2: "mild", 3: "moderate"}.get(n, f"{m.group(1).upper()} {n}")
    return _clean(s)

# --------- core parser ---------
def parse_report(text: str) -> dict:
    t = text.replace("\r", "\n")

    out = {k: "" for k in COLUMNS}

    # APOE
    # APOE — include noncarrier / non-carrier, normalize
    out["Apoe"] = _first(
        t,
        r"APOE[\s\-]*e?4[^:\n]*:\s*(hetero(?:zygous)?|homo(?:zygous)?|non[\-\s]*carrier|negative|pos(?:itive)?|unknown)",
        default="",
    ).lower().replace(" ", "").replace("-", "")

    # Lecanemab dose — capture ordinals/numbers/phrases and strip trailing punctuation
    lec_raw = _first(
        t,
        [
            r"Lecanemab\s*dose\s*:\s*([^\n\r\.]+)",            # anything up to newline/period
            r"Lecanemab\s*dose\s*:\s*([0-9]{1,2}(?:st|nd|rd|th))",  # 1st, 2nd, ...
            r"\bprior to first dose\b",                        # fallback phrasing
        ],
        default="",
    )
    lec_raw = lec_raw.strip().rstrip(".").lower()
    if not lec_raw and re.search(r"prior to first dose", t, re.I):
        lec_raw = "prior to first dose"
    out["Lecanemab dose"] = lec_raw

    # Acute hemorrhage
    out["acute hemmorage"] = _first(
        t,
        [r"Acute/?subacute hemorrhage:\s*([^\n\.]+)", r"\bacute hemorrhage:\s*([^\n\.]+)"],
        default="",
    ).lower() or ("none" if re.search(r"acute[/\s]?subacute hemorrhage:\s*none", t, re.I) else "")

    # Acute/subacute cortical infarct > 1.5 cm
    out["Acute/subacute cortical infarct"] = _first(
        t,
        r"Acute/?subacute cortical infarct[^:]*:\s*([^\n\.]+)",
        default="",
    ).lower()

    # Chronic hemorrhage
    out["Chronic hemorrhage"] = _first(
        t, r"Chronic hemorrhage:\s*([^\n\.]+)", default=""
    ).lower()

    # Chronic ischemic changes -> yes/no from narrative
    chronic_isch = _first(
        t,
        [
            r"Chronic ischemic changes:\s*([^\n]+)",
            r"microvascular ischemic changes.*?(mild|moderate|severe|present|none|absent)",
        ],
        default="",
    )
    out["Chronic ischemic changes"] = "yes" if chronic_isch and re.search(r"mild|mod|sev|present|changes", chronic_isch, re.I) else ("no" if re.search(r"\bnone|absent\b", chronic_isch, re.I) else "")

    # MTA
    out["MTA-right"] = _normalize_grade(
        _first(t, [r"Medial Temporal Atrophy.*?Right:\s*([^\n]+)"], default="")
    )
    out["MTA-left"] = _normalize_grade(
        _first(t, [r"Medial Temporal Atrophy.*?Left:\s*([^\n]+)"], default="")
    )

    # ERiCA
    out["ERiCA-right"] = _normalize_grade(
        _first(t, [r"Entorhinal Cortex Atrophy.*?Right:\s*([^\n]+)"], default="")
    )
    out["ERiCA-left"] = _normalize_grade(
        _first(t, [r"Entorhinal Cortex Atrophy.*?Left:\s*([^\n]+)"], default="")
    )

    # Extra-axial Collection
    out["Extra-axial Collection"] = _first(
        t, r"Extra-axial Collection:\s*([^\n\.]+)", default=""
    ).lower()

    # Ventricular System
    out["Ventricular System"] = _first(
        t, r"Ventricular System:\s*([^\n\.]+)", default=""
    )

    # Major Intracranial Flow Voids
    out["Major Intracranial Flow Voids"] = _first(
        t, r"Major Intracranial Flow Voids:\s*([^\n\.]+)", default=""
    ).lower()

    # Included Orbits
    out["Included Orbits"] = _first(
        t, r"Included Orbits:\s*([^\n\.]+)", default=""
    ).lower()

    # Paranasal Sinuses
    out["Paranasal Sinuses"] = _first(
        t, r"Paranasal Sinuses:\s*([^\n\.]+)", default=""
    )
    # Normalize common phrase "Predominantly clear" -> "clear"
    if re.search(r"\bclear\b", out["Paranasal Sinuses"], re.I):
        out["Paranasal Sinuses"] = "clear"

    # Tympanomastoid Cavities (keep your column spelling)
    out["Typanomastoid Cavaties"] = _first(
        t,
        [r"Tympanomastoid Cavit(?:y|ies):\s*([^\n\.]+)",
         r"Tympanomastoid.*?:\s*([^\n\.]+)"],
        default=""
    ).lower() or "normal" if re.search(r"Tympanomastoid.*normal", t, re.I) else ""

    # Final sanitization pass on all values
    for key in out:
        out[key] = _sanitize_value(out[key])

    return out

def write_csv_row(path, rowdict, header=COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerow({k: rowdict.get(k, "") for k in header})

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--textfile", help="Text file with a single report")
    ap.add_argument("--stdin", action="store_true", help="Read report text from STDIN")
    ap.add_argument("--xlsx", help="(Optional) Excel file to batch-parse a column")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (default first)")
    ap.add_argument("--column", default=None, help="Column name in Excel that holds the report text")
    ap.add_argument("--out", default="out.csv", help="Output CSV path")
    args = ap.parse_args()

    if args.xlsx:
        # batch mode
        try:
            import pandas as pd
        except Exception as e:
            sys.exit("Pandas required for Excel mode: pip install pandas openpyxl")

        if not args.column:
            sys.exit("--column is required with --xlsx")
        df = pd.read_excel(args.xlsx, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.xlsx)
        out_rows = []
        for _, r in df.iterrows():
            text = str(r.get(args.column, "") or "")
            if text.strip():
                row = parse_report(text)
                # copy ARIA columns from the source row (handle pandas NA)
                try:
                    row["ARIA-E"] = _sanitize_value("" if pd.isna(r.get("ARIA-E")) else str(r.get("ARIA-E")))
                except Exception:
                    row["ARIA-E"] = ""
                try:
                    row["ARIA-H"] = _sanitize_value("" if pd.isna(r.get("ARIA-H")) else str(r.get("ARIA-H")))
                except Exception:
                    row["ARIA-H"] = ""
                out_rows.append(row)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        print(f"Wrote {len(out_rows)} rows to {args.out}")
        return

    # single-report mode
    if args.stdin:
        text = sys.stdin.read()
    elif args.textfile:
        text = open(args.textfile, "r", encoding="utf-8").read()
    else:
        sys.exit("Provide --textfile or --stdin, or use --xlsx for batch mode.")

    row = parse_report(text)
    write_csv_row(args.out, row)
    print(f"Wrote 1 row to {args.out}")

if __name__ == "__main__":
    main()