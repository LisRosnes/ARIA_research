 # ARIA_research

This repository provides tools to extract structured features from MRI radiology reports and run univariate analyses to find associations with ARIA adverse events (ARIA-E and ARIA-H).

## Contents

- `feature_extraction.py` — Parse free-text MRI reports into a standardized CSV of clinical and imaging features.
- `univariate_analysis.py` — Run univariate statistical tests and produce summary tables and visualizations.
- `subject_mapping.py` — Aggregate subject-level ARIA status and event dates across longitudinal studies; produces `aria_mapping_output.csv`.
- `merged_stats_full_tp.csv`, `output_features_2.csv` — example/derived data files (if present).
- `univariate_results.csv` — generated results summary (output of `univariate_analysis.py`).
- `univariate_plots/` — folder where generated PNG plots are saved.

## Quick Start

1. (Optional) Create a new conda environment with Python 3.11:

```bash
conda create -n aria_py311 python=3.11 -y
conda activate aria_py311
```

2. Install required Python packages:

```bash
pip install pandas openpyxl numpy scipy matplotlib seaborn
```

3. Extract features from reports (examples):

```bash
# Single text-file report -> CSV
python feature_extraction.py --textfile report.txt --out features.csv

# Read from stdin
cat report.txt | python feature_extraction.py --stdin > features.csv

# From Excel: specify sheet and column
python feature_extraction.py --xlsx reports.xlsx --sheet Sheet1 --column Report --out features.csv
```

4. Run univariate analysis on the feature CSV:

```bash
python univariate_analysis.py
# (Edit the csv input path in the script's __main__ or provide arguments if implemented)
```

## `feature_extraction.py` (summary)

Purpose: parse radiology report free text and extract structured fields relevant to ARIA analyses.

Key behaviors:
- Robust regex-based parsing to handle varied report language.
- Normalizes findings to standardized values (e.g., `yes`/`no`, `none`).
- Maps numeric atrophy grades (0–3) to descriptive levels (normal, mild, moderate, severe).
- Sanitizes CSV output to avoid quoting/character issues and normalizes whitespace.

Typical output: a CSV with one row per report and columns for extracted features (genetics, treatment, acute/chronic findings, structural grades, ARIA flags).

## `subject_mapping.py` (summary)

Purpose: Aggregate subject-level ARIA status and derive event dates across longitudinal imaging studies.

Key behaviors:
- Reads an Excel input of imaging results and metadata (expects columns like `SubjectID`, `Study Date`, `ARIA-E`, `ARIA-H`).
- Converts `Study Date` to datetime and sorts records by `SubjectID` and date to ensure chronological processing.
- Normalizes `ARIA-E` and `ARIA-H` indicators to binary values (1 for "Yes", 0 otherwise).
- For each subject, finds the first date an ARIA flag appears and the date it is resolved (first subsequent `0` after a `1`), marking unresolved events as "Not Resolved".
- Outputs a subject-level CSV (`aria_mapping_output.csv`) listing per-subject ARIA presence and event dates and prints a preview to the console.

Dependencies: `pandas` (for reading Excel, datetime handling, grouping and CSV output).

## `univariate_analysis.py` (summary)

Purpose: standardize the extracted features, choose appropriate univariate statistical tests, and produce a results table and visualizations.

Key behaviors:
- Converts binary features to 0/1 and ordinal features to numeric scales.
- Handles APOE carrier status and lecanemab dosing encodings.
- Chooses tests automatically: chi-square for categorical (falls back to Fisher's exact for small counts), Mann–Whitney U for ordinal comparisons.
- Exports `univariate_results.csv` and saves plots to `univariate_plots/` for significant findings.

## Dependencies

- Python 3.8+ (Python 3.11 recommended).
- Required packages: `pandas`, `openpyxl` (for Excel input), `numpy`, `scipy`, `matplotlib`, `seaborn`.

Install with pip:

```bash
pip install pandas openpyxl numpy scipy matplotlib seaborn
```

Or with conda:

```bash
conda install pandas openpyxl numpy scipy matplotlib seaborn -c conda-forge
```

## Outputs

- `features.csv` (or your chosen output filename) — feature table produced by `feature_extraction.py`.
- `univariate_results.csv` — summary of statistical tests and p-values.
- `univariate_plots/` — directory with PNG plots for significant features.
