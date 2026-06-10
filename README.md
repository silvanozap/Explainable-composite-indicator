# EI-SCORE: Explainable-Interpretable and Simple Customized Overall Ranking Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
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
- **New unit classification** — via MILP formulations (Problems 6, 7, 8)
- **Rule reuse** — apply saved rules to new datasets

The methodology implements Algorithms 1, 2 and 4 (for missing values) from the reference paper, along with the full 7-step pipeline including greedy rule selection and MILP-based minimisation.

---

## Requirements

- Python 3.14.3
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
├── drsa/
│   ├── core/
│   │   ├── rules.py                # Algorithms rule induction
│   │   ├── utils.py                # Supporting functions
│   │   ├── formatting.py           # Rule text formatting
│   │   ├── classifier.py           # Unit classification
│   │   ├── measures.py             # Quality measures
│   │   ├── step_forward.py         # Greedy selection
│   │   ├── milp.py                 # MILP minimal set of rules
│   │   ├── pipeline.py             # Full pipeline orchestration
│   │   └── new_units.py            # MILP classify new units
│   └── interface/
│       └── app.py                  # Streamlit application
├── examples/
│   ├── HDI.csv                     # Human Development Index
│   ├── PTF.csv                     # Stock portfolio
│   ├── PTF_new_units.csv           # New stocks for PTF
│   ├── ElectreScore.csv            # Scoring problem
│   ├── rules_PTF.csv               # Pre-computed PTF rules
│   ├── GCS.csv                     # Glasgow Coma Scale
│   ├── GCS_new_units.csv           # New patient for GCS
│   ├── MISSING.csv                 # Dataset with missing values
│   └── MISSING_new_units.csv       # New units with missing values
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## Input File Format

### Data file (CSV or TXT)
```
Name,criterion_1,criterion_2,...,criterion_k,class
A1,4,3,2,1
A2,5,4,3,2
A3,7,6,5,
A4,6,5,4,
```

- **First column** (optional): unit names
- **Criteria columns**: numeric values
- **Last column**: class label — integer for reference units, empty for non-reference units
- **Missing values**: detected automatically;

For **scoring problems**, the last column contains a continuous score. Select "Scoring" in the *Problem type* radio button.

### Rules file (CSV) for classification
```
#directions,increasing,decreasing,...
#mode,class
type,class,criterion_1,criterion_2,...
at-least,2,0.762,,
at-most,1,,2.71,
```
### Rules file (CSV) for scoring
```
#directions,increasing,decreasing,...
#mode,score
type,class,criterion_1,criterion_2,...
at-least,2,0.762,,
at-most,1,,2.71,
```

---

## Application Workflow

### Tab 1 — Data & Setup
Upload your dataset, select criterion preference directions (increasing/decreasing), choose problem type (Classification or Scoring), and configure analysis settings:
- select reference units
- select level of confidence
- set a different seed

### Tab 2 — Run
- **Rule induction only**: induces at-least (R≥) and at-most (R≤) rules from reference units
- **Full pipeline**: greedy selection of rules to avoid contradictory classification of all units, rule induction from all units and find and minimal set of rules.

### Tab 3 — Classification
Displays classification of all units with lower (s⁻) and upper (s⁺) class bounds. Unit-by-unit explanation available.

### Tab 4 — New Units
Classify new alternatives finding the set of rules that handle contradictions.

### Tab 5 — Apply Rules
Load a saved rules CSV to visualise rules and classify new alternatives. Always accessible regardless of whether a data file is loaded.

---

## Examples

### Example 1 — Human Development Index (HDI)
**Files:** `examples/HDI.csv`  
**Problem:** Classification  
**Setup:** 193 countries, 4 criteria, 4 classes  
**Direction:** all increasing
**Workflow:** Upload HDI.csv → set all criteria increasing → Tab Run → Rule induction only → inspect rules → tab classification → inspect classification

### Example 2 — Stock Portfolio (PTF)
**Files:** `examples/PTF.csv`, `examples/PTF_new_units.csv`  
**Problem:** Full pipeline + New units  
**Setup:** 50 stocks, 8 financial criteria, 3 classes  
**Direction:** all increasing
**Workflow:** Upload PTF.csv → set all criteria increasing → tab Run → Full pipeline → New Units tab → upload PTF_new_units.csv → Classify new units → Inspect classification

### Example 3 — ElectreScore (Scoring)
**Files:** `examples/ElectreScore.csv`, `examples/rules_PTF.csv`  
**Problem:** Scoring workflow and classification
**Setup:** 10 investment projects, 5 criteria, continuous score 0–100  
**Direction:** decreasing, decreasing, increasing, increasing, increasing
**Workflow (scoring):** Upload ElectreScore.csv → select Scoring → set criteria direction → tab Run → Rule induction only → inspect rules → Tab Classification → inspect


### Example 4 — Missing Values (Classification)
**Files:** `examples/MISSING.csv`, `examples/MISSING_new_units.csv`  
**Problem:** Classification with missing values, Full pipeline  
**Setup:** 20 students, 3 criteria, 3 classes, several missing values
**Direction:** all increasing
**Workflow:** Upload MISSING.csv → tab Run → Full pipeline → tab New Units → upload MISSING_new_units.csv → Classify new units → inspect classification

### Example 5 — Rule reuse
**Files:** `examples/rules_score.csv`, `examples/ElectreScore.csv`  
**Problem:** Visualize inducted rules + score units
**Setup:** 26 at-least rules, 25 at-most rules, 5 criteria, continuous score 0–100
**Direction:** decreasing, decreasing, increasing, increasing, increasing
**Workflow (Apply Rules):** Tab Apply Rules → upload rules_score.csv → visualize rules → upload ElectreScore.csv → visualize scoring

---

## Authors

- **Salvatore Corrente** — University of Catania, Italy
- **Salvatore Greco** — University of Catania, Italy
- **Roman Słowiński** — Poznań University of Technology, Poland; Polish Academy of Sciences, Poland
- **Silvano Zappalà** — University of Catania, Italy

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
