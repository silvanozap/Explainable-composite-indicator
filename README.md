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
- **Full pipeline** — from raw data to minimal rule sets (Steps 1–7)
- **New unit assignment** — via MILP formulations (Problems 6, 7, 8)
- **Rule reuse** — apply saved rules to new datasets

The methodology implements Algorithms 1, 2 and 4 (for missing values) from the reference paper, along with the full pipeline including greedy rule selection and MILP-based minimisation.

---

## Requirements

- Python 3.14.3 or later
- Dependencies (see `requirements.txt`):

```
numpy==2.4.2
pandas==3.0.1
scipy==1.17.0
streamlit==1.58.0
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
│   └── logo.png
├── docs/
│   └── user_guide.tex          # LaTeX source of the user guide
├── drsa/
│   ├── core/
│   │   ├── rules.py            # Rule induction algorithms
│   │   ├── utils.py            # Supporting functions
│   │   ├── formatting.py       # Rule text formatting
│   │   ├── classifier.py       # Unit assignment — eq. (4)
│   │   ├── measures.py         # Quality measures
│   │   ├── step_forward.py     # Greedy selection
│   │   ├── milp.py             # MILP minimal set of rules
│   │   ├── pipeline.py         # Full pipeline orchestration
│   │   └── new_units.py        # MILP assign new units
│   └── interface/
│       └── app.py              # Streamlit application
├── examples/
│   ├── Ex.1_HDI.csv                  # Human Development Index
│   ├── Ex.2_PTF.csv                  # Stock portfolio
│   ├── Ex.2_PTF_new_units.csv        # New stocks for PTF
│   ├── Ex.3_ElectreScore.csv         # Scoring problem
│   ├── Ex.4_MISSING.csv              # Dataset with missing values
│   ├── Ex.4_MISSING_new_units.csv    # New units with missing values
│   ├── Ex.5_GCS_original.csv         # Glasgow Coma Scale
│   ├── Ex.5_GCS_new_units.csv        # New patient for GCS
│   └── Ex.5_maximal_rules.csv        # Pre-computed GCS rules
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## Input File Format

### Units file (CSV or TXT)
```
Name,criterion_1,criterion_2,...,criterion_k,class
A1,4,3,2,1
A2,5,4,3,2
A3,7,6,5,
A4,6,5,4,
```

- **First column** (optional): unit names
- **Criteria columns**: numeric values
- **Last column**: class/score label — numeric for reference units, empty for non-reference units
- **Missing values**: detected automatically; Algorithm 4 applied when present

For **scoring problems**, the last column contains a continuous score. Select "Scoring" in the *Problem type* radio button in Tab Run.

### Rules file (CSV)
```
#directions,increasing,decreasing,...
#mode,class
type,class,criterion_1,criterion_2,...
at-least,2,0.762,,
at-most,1,,2.71,
```

For scoring problems, use `#mode,score` and score values in the `class` column.

---

## Application Workflow

### Tab 1 — Data & Setup
Upload your dataset, select criterion preference directions (increasing/decreasing), select reference units, and configure settings (min confidence, random seed).

### Tab 2 — Run
- **Rule induction only**: induces at-least (R≥) and at-most (R≤) rules from reference units
- **Full pipeline**: greedy selection → assignment of all units → maximal rule set → MILP minimal rule set

### Tab 3 — Assignment
Displays assignment of all units with lower (s⁻) and upper (s⁺) bounds. Unit-by-unit explanation available.

### Tab 4 — New Units
Classify new units using MILP Problems (6), (7), and (8). Requires Full pipeline to be run first.

### Tab 5 — Apply Rules
Load saved rules and assign units. Extends to new unit assignment using loaded rules. Always accessible regardless of whether a data file is loaded.

---

## Examples

### Example 1 — Human Development Index (HDI)
**File:** `examples/Ex.1_HDI.csv`  
**Problem:** Classification — rule induction only  
**Setup:** 193 countries, 4 criteria (all increasing ↑), 4 classes  
**Workflow:** Upload → set directions → Rule induction only → inspect rules → Tab Assignment

### Example 2 — Stock Portfolio (PTF)
**Files:** `examples/Ex.2_PTF.csv`, `examples/Ex.2_PTF_new_units.csv`  
**Problem:** Full pipeline + New units  
**Setup:** 50 stocks, 8 financial criteria (all increasing ↑), 3 classes  
**Workflow:** Upload PTF.csv → Full pipeline → Tab Assignment → Tab New Units → upload PTF_new_units.csv

### Example 3 — Scored units (ElectreScore)
**File:** `examples/Ex.3_ElectreScore.csv`  
**Problem:** Scoring  
**Setup:** 10 investment projects, 5 criteria (g1↓, g2↓, g3↑, g4↑, g5↑), continuous score 0–100  
**Workflow:** Upload → set directions → select Scoring → Rule induction only → inspect rules → Tab Assignment

### Example 4 — Missing Values (PISA dataset)
**Files:** `examples/Ex.4_MISSING.csv`, `examples/Ex.4_MISSING_new_units.csv`  
**Problem:** Classification with missing values, Full pipeline  
**Setup:** 20 students, 3 criteria (all increasing ↑), 3 classes, several missing values  
**Workflow:** Upload MISSING.csv → Full pipeline → Tab Assignment → Tab New Units → upload MISSING_new_units.csv

### Example 5 — Apply Rules (Glasgow Coma Scale)
**Files:** `examples/Ex.5_GCS_original.csv`, `examples/Ex.5_GCS_new_units.csv`, `examples/Ex.5_maximal_rules.csv`  
**Problem:** Load previous rules, assign units, assign new units  
**Setup:** Pre-computed GCS rules (3 criteria, class mode)  
**Workflow:** Tab Apply Rules → upload Ex.5_maximal_rules.csv → upload Ex.5_GCS_original.csv → inspect assignment → upload Ex.5_GCS_new_units.csv → Assign new units

---

## Authors

- **Salvatore Corrente** — University of Catania, Italy
- **Salvatore Greco** — University of Catania, Italy
- **Roman Słowiński** — Poznań University of Technology, Poland; Polish Academy of Sciences, Poland
- **Silvano Zappalà** — University of Catania, Italy (corresponding author)

---

## License

This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt) for details.

---

## Citation

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
```
