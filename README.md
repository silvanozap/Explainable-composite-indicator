# EI-SCORE: Explainable-Interpretable and Simple Customized Overall Ranking Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.58-red.svg)](https://streamlit.io/)

A user-friendly GUI to build customized composite indicators based on Decision Rules, implementing the Dominance-based Rough Set Approach (DRSA).

> **Reference:** Corrente S., Greco S., Słowiński R., Zappalà S. (2026). *An explainable and interpretable composite indicator based on decision rules.* Omega, 142, 103513. [DOI: 10.1016/j.omega.2026.103513](https://doi.org/10.1016/j.omega.2026.103513)

---

## Overview

EI-SCORE provides an interactive Streamlit interface for building explainable composite indicators using decision rules induced from preference-ordered data. The tool supports:

- **Classification problems** — assigning units to ordered classes
- **Scoring problems** — ranking units by a continuous score
- **Qualitative criteria** — non-numeric criteria with user-defined preference ordering
- **Full pipeline** — from raw data to minimal rule sets
- **Incremental learning** — New unit assignment via MILP formulations
- **Rule reuse** — apply saved rules to new datasets

The methodology recognises and handles missing values in criteria.

---

## Requirements

- Python 3.14.3 or later
- Dependencies (see `requirements.txt`):

```
numpy==2.4.2
pandas==3.0.1
scipy==1.17.0
streamlit==1.58.0
openpyxl==3.1.5
```

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/silvanozap/Explainable-composite-indicator.git
cd Explainable-composite-indicator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch the application:**
```bash
streamlit run drsa/interface/app.py
```

The application opens automatically in your default browser at `http://localhost:8501`.

---

## Repository Structure

```
Explainable-composite-indicator/
├── assets/
│   ├── ei-score.svg               # Application logo
│   └── user_guide.pdf             # Compiled user guide
├── docs/
│   └── user_guide.tex             # LaTeX source of the user guide
├── drsa/
│   ├── core/
│   │   ├── rules.py               # Rule induction algorithms
│   │   ├── utils.py               # Supporting functions
│   │   ├── formatting.py          # Rule text formatting
│   │   ├── classifier.py          # Unit assignment
│   │   ├── measures.py            # Quality measures
│   │   ├── step_forward.py        # Greedy selection
│   │   ├── milp.py                # MILP minimal set of rules
│   │   ├── pipeline.py            # Full pipeline orchestration
│   │   └── new_units.py           # MILP assign new units
│   └── interface/
│       └── app.py                 # Streamlit application
├── examples/
│   ├── Ex.1_HDI.csv               # Human Development Index (classification)
│   ├── Ex.2_PTF.csv               # Stock portfolio (full pipeline)
│   ├── Ex.2_PTF_new_units.csv     # New stocks for PTF
│   ├── Ex.3_ElectreScore.csv      # Scoring problem
│   ├── Ex.4_MISSING.csv           # Dataset with missing values
│   ├── Ex.4_MISSING_new_units.csv # New units with missing values
│   ├── Ex.5_GCS_original.csv      # Glasgow Coma Scale (Apply Rules)
│   ├── Ex.5_GCS_new_units.csv     # New patients for GCS
│   ├── Ex.5_maximal_rules.csv     # Pre-computed GCS rules
│   ├── Ex.6_GCS_qualitative.xlsx  # GCS dataset with qualitative criteria
│   ├── Ex.7_qualitative_maximal_rules.csv  # Pre-computed rules (qualitative)
│   ├── Ex.7_GCS_qualitative.txt   # GCS units for Apply Rules (qualitative)
│   └── Ex.7_GCS_qualitative_new.xlsx       # New units (qualitative)
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Input File Format

### Units file (CSV, TXT, or Excel)

```
Name,criterion_1,criterion_2,...,criterion_k,class
A1,4,3,2,1
A2,5,4,3,2
A3,7,6,5,
A4,6,5,4,
```

- **First column**: unit names
- **Criteria columns (intermediate)**: numeric or qualitative values
- **Last column**: class/score label — non-empty for reference units, empty for non-reference units
- **Missing values**: detected and handled automatically
- **File format**: CSV and TXT files require a separator (comma, semicolon, tab, or space); Excel files (`.xlsx`) are read directly from the first sheet

For **scoring problems**, the last column contains a continuous score. Select *Scoring* in the Problem type radio button in Tab Run.

For **qualitative criteria**, non-numeric columns are automatically detected. A mapping widget appears in Tab Data & Setup to assign a numeric rank to each qualitative value (1 = worst, p = best).

### Rules file (CSV)

```
#directions,increasing,decreasing,...
#mode,class
#mapping,criterion_name,value1:1,value2:2,...
type,assignment,criterion_1,criterion_2,...
at-least,2,0.762,,
at-most,1,,2.71,
```

- `#directions`: preference direction for each criterion (`increasing` or `decreasing`)
- `#mode`: `class` or `score`
- `#mapping` (optional): qualitative mapping for each non-numeric criterion and/or class column
- `type`: `at-least` or `at-most`
- `assignment`: class index or score value

---

## Application Workflow

### Tab 1 — Data & Setup
Upload your dataset (CSV, TXT, or Excel), define criterion preference directions (increasing/decreasing), select reference units, and configure settings (confidence level, random seed). If qualitative criteria are detected, a mapping widget allows assigning a numeric rank to each qualitative value.

### Tab 2 — Run
- Select **problem type** (Classification or Scoring)
- **Rule induction only**: induces at-least (R≥) and at-most (R≤) rules from reference units. If all units are reference units, a **Find minimal set** button appears to compute the minimal subset of rules that preserves the same classification.
- **Full pipeline**: greedy selection → assignment of all units → maximal rule set → minimal rule set via MILP

### Tab 3 — Assignment
Displays the assignment of all units with Minimal assignment and Maximal assignment bounds. Unit-by-unit explanation available showing which rules are satisfied.

### Tab 4 — Incremental learning
Classify new units using MILP Problems. Requires the Full pipeline to be run first. New units files may contain qualitative values, which are automatically converted using the mapping defined in Tab 1.

### Tab 5 — Apply Rules
Load saved rules and assign units. Qualitative mappings stored in the `#mapping` lines of the rules file are applied automatically. Supports new unit assignment using loaded rules. Always accessible regardless of whether a dataset is loaded in the Sidebar.

---

## Examples

### Example 1 — Human Development Index (HDI)
**File:** `examples/Ex.1_HDI.csv`  
**Problem:** Classification — rule induction only  
**Setup:** 193 countries, 4 criteria (all increasing ↑), 4 classes  
**Workflow:** Upload → set directions → Rule induction only → inspect rules → Tab Assignment

### Example 2 — Stock Portfolio (PTF)
**Files:** `examples/Ex.2_PTF.csv`, `examples/Ex.2_PTF_new_units.csv`  
**Problem:** Full pipeline + Incremental learning  
**Setup:** 50 stocks, 8 financial criteria (all increasing ↑), 3 classes  
**Workflow:** Upload PTF.csv → Full pipeline → Tab Assignment → Tab Incremental learning → upload PTF_new_units.csv

### Example 3 — Scoring (ElectreScore)
**File:** `examples/Ex.3_ElectreScore.csv`  
**Problem:** Scoring  
**Setup:** 10 investment projects, 5 criteria (g1↓, g2↓, g3↑, g4↑, g5↑), continuous score 0–100  
**Workflow:** Upload → set directions → select Scoring → Rule induction only → inspect rules → Tab Assignment

### Example 4 — Missing Values (PISA dataset)
**Files:** `examples/Ex.4_MISSING.csv`, `examples/Ex.4_MISSING_new_units.csv`  
**Problem:** Classification with missing values, Full pipeline  
**Setup:** 20 students, 3 criteria (all increasing ↑), 3 classes, several missing values  
**Workflow:** Upload MISSING.csv → Full pipeline → Tab Assignment → Tab Incremental learning → upload MISSING_new_units.csv

### Example 5 — Apply Rules (Glasgow Coma Scale)
**Files:** `examples/Ex.5_GCS_original.csv`, `examples/Ex.5_GCS_new_units.csv`, `examples/Ex.5_maximal_rules.csv`  
**Problem:** Load previous rules, assign units, assign new units  
**Setup:** Pre-computed GCS rules (3 numeric criteria, class mode)  
**Workflow:** Tab Apply Rules → upload Ex.5_maximal_rules.csv → upload Ex.5_GCS_original.csv → inspect assignment → upload Ex.5_GCS_new_units.csv → Assign new units

### Example 6 — Qualitative Criteria (Glasgow Coma Scale)
**File:** `examples/Ex.6_GCS_qualitative.xlsx`  
**Problem:** Classification with qualitative criteria  
**Setup:** GCS dataset with qualitative entries (Eye opening, Verbal response, Motor response, Severity class)  
**Workflow:** Upload Ex.6_GCS_qualitative.xlsx → assign ranks in Qualitative criteria mapping → set directions → Full pipeline → Tab Assignment

### Example 7 — Apply Rules with Qualitative Criteria
**Files:** `examples/Ex.7_qualitative_maximal_rules.csv`, `examples/Ex.7_GCS_qualitative.txt`, `examples/Ex.7_GCS_qualitative_new.xlsx`  
**Problem:** Load pre-computed rules with qualitative criteria, assign units and new units  
**Setup:** Pre-computed qualitative GCS rules with `#mapping` lines  
**Workflow:** Tab Apply Rules → upload Ex.7_qualitative_maximal_rules.csv → upload Ex.7_GCS_qualitative.txt → inspect assignment → upload Ex.7_GCS_qualitative_new.xlsx → Assign new units

---

## Authors

- **Salvatore Corrente** — University of Catania, Italy
- **Salvatore Greco** — University of Catania, Italy
- **Roman Słowiński** — Poznań University of Technology, Poland; Polish Academy of Sciences, Poland
- **Silvano Zappalà** — University of Catania, Italy (corresponding author)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this software in your research, projects, or publications, please cite the following papers:

- Corrente S., Greco S., Słowiński R., Zappalà S. (2026). *An explainable and interpretable composite indicator based on decision rules.* Omega, 142, 103513. [DOI: 10.1016/j.omega.2026.103513](https://doi.org/10.1016/j.omega.2026.103513)
- Greco S., Matarazzo B., Słowiński R. (2001). *Rough sets theory for multicriteria decision analysis.* European Journal of Operational Research, 129(1), 1–47. [DOI: 10.1016/S0377-2217(00)00167-3](https://doi.org/10.1016/S0377-2217(00)00167-3)

```bibtex
@article{corrente2026omega,
  author  = {Corrente, Salvatore and Greco, Salvatore and S{\l}owi{\'n}ski, Roman and Zappal{\`a}, Silvano},
  title   = {An explainable and interpretable composite indicator based on decision rules},
  journal = {Omega},
  volume  = {142},
  pages   = {103513},
  year    = {2026},
  doi     = {10.1016/j.omega.2026.103513}
}
@article{greco2001,
  author    = {Greco, Salvatore and Matarazzo, Benedetto and S{\l}owi{\'n}ski, Roman},
  title     = {Rough sets theory for multicriteria decision analysis},
  journal   = {European Journal of Operational Research},
  volume    = {129},
  number    = {1},
  pages     = {1--47},
  year      = {2001},
  publisher = {Elsevier},
  doi       = {10.1016/S0377-2217(00)00167-3}
}
```