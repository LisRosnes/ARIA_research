 # ARIA_research

This repository provides tools to extract structured features from MRI radiology reports and run univariate and multivariate analyses to find associations with ARIA adverse events (ARIA-E and ARIA-H).

## Contents

- `feature_extraction.py` — Parse free-text MRI reports into a standardized CSV of clinical and imaging features.
- `univariate_analysis.py` — Run univariate statistical tests and produce summary tables and visualizations.
- `multivariate_analysis.py` —  Elastic Net logistic regression pipeline for ARIA prediction using FreeSurfer brain MRI features.
- `subject_mapping.py` — Aggregate subject-level ARIA status and event dates across longitudinal studies; produces `aria_mapping_output.csv`.
- `merged_stats_full_tp.csv` — FreeSurfer volumetric and cortical thickness measurements for all subjects.
- `aria_mapping_output.csv` — Subject-level ARIA outcomes (ARIA-E and ARIA-H status).
- `output_features_2.csv` — example/derived data files (if present).
- `univariate_results.csv` — generated results summary (output of `univariate_analysis.py`).
- `univariate_plots/` — folder where generated PNG plots are saved.
- `multivariate_plots/` — folder for multivariate analysis visualizations (ROC curves, PR curves, feature importance).

## Quick Start

1. (Optional) Create a new conda environment with Python 3.11:

```bash
conda create -n aria_py311 python=3.11 -y
conda activate aria_py311
```

2. Install required Python packages:

```bashpip install pandas openpyxl numpy scipy matplotlib seaborn scikit-learn
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

5. Run multivariate analysis (Elastic Net):

```bash
python multivariate_analysis.py
# Automatically analyzes both ARIA-E and ARIA-H outcomes
# Generates feature importance CSVs and visualization plots
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
- Required packages: `pandas`, `openpyxl` (for Excel input), `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn` (for multivariate analysis).

Install with pip:

```bash
pip install pandas openpyxl numpy scipy matplotlib seaborn scikit-learn
```

## `multivariate_analysis.py` (summary)

**NEW**: Purpose: Build predictive models for ARIA-E and ARIA-H outcomes using elastic net logistic regression with FreeSurfer brain MRI features.

Key behaviors:
- **Data Integration**: Merges FreeSurfer volumetric/cortical measurements with ARIA outcome data.
- **Feature Preparation**: Handles missing data, removes zero-variance features, and applies standardization.
- **Sample Size Aware**: Automatically adapts analysis strategy based on number of events:
  - <10 events: Descriptive analysis only (univariate associations)
  - 10-14 events: Simple modeling without cross-validation
  - ≥15 events: Full elastic net with cross-validated hyperparameter tuning
- **Model Training**: Elastic net logistic regression with:
  - Balanced class weights to handle imbalanced outcomes
  - Grid search over regularization strength (C) and elastic net mixing (l1_ratio)
  - Stratified k-fold cross-validation when sufficient events
- **Feature Selection**: L1/L2 regularization automatically selects relevant features and sets others to zero
- **Outputs**:
  - Feature importance CSVs showing non-zero coefficients
  - Univariate association CSVs with Mann-Whitney U test results
  - Visualization plots (ROC curves, PR curves, feature importance, confusion matrices) in `multivariate_plots/`
- **Performance Metrics**: AUROC, AUPRC, classification reports, confusion matrices

Current data status (as of run):
- ARIA-E: 2 events (insufficient for modeling - descriptive analysis only)
- ARIA-H: 0 events (no cases - cannot perform analysis)

The pipeline will automatically perform full modeling when sufficient events are accumulated.

Or with conda:

```bash
conda install pandas openpyxl numpy scipy matplotlib seaborn -c conda-forge
```

## Outputs

- `features.csv` (or your chosen output filename) — feature table produced by `feature_extraction.py`.
- `univariate_results.csv` — summary of statistical tests and p-values.
- `univariate_plots/` — directory with PNG plots for significant features.
