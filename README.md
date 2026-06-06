# DRSA Rule Induction

Python implementation of the Dominance-based Rough Set Approach (DRSA) for
explainable and interpretable composite indicators.

## Reference

Corrente S., Greco S., Słowiński R., Zappalà S. (2026).
*An explainable and interpretable composite indicator based on decision rules.*
Omega 142, 103513. https://doi.org/10.1016/j.omega.2026.103513

## Implemented algorithms

| Algorithm | Description |
|-----------|-------------|
| Algorithm 1 | Induction of all minimal "at-least" decision rules |
| Algorithm 2 | Induction of all minimal "at-most" decision rules (duality) |
| Algorithm 4 | Algorithms 1 & 2 extended to datasets with missing values |

## Installation

```bash
pip install -r requirements.txt
```

## Running the interface

```bash
streamlit run drsa/interface/app.py
```

## Input file format

- CSV or TXT file
- One unit per row
- Columns: criteria evaluations + class label (last column)
- Class labels: integers (1 = worst, p = best)
- Missing values: empty cell or `NaN`

Example:

```
g1,g2,g3,class
4,3,2,1
5,4,3,2
7,6,5,3
```

## Python API

```python
import numpy as np
from drsa import induce_atleast_rules, induce_atmost_rules
from drsa import format_atleast_rules, format_atmost_rules

# example_matrix: rows = units, last column = class
matrix = np.array([
    [4, 3, 2, 1],
    [5, 4, 3, 2],
    [7, 6, 5, 3],
])

increasing = [0, 1, 2]   # 0-based indices
decreasing = []

# Induce rules
al_rules, al_match, al_dec, al_bases = induce_atleast_rules(
    matrix, increasing, decreasing
)
am_rules, am_match, am_dec, am_bases = induce_atmost_rules(
    matrix, increasing, decreasing
)

# Format as natural language
al_texts = format_atleast_rules(al_rules, increasing, decreasing)
am_texts = format_atmost_rules(am_rules, increasing, decreasing)

for t in al_texts:
    print(t)
```

## Missing values

```python
al_rules, *_ = induce_atleast_rules(
    matrix, increasing, decreasing,
    handle_missing=True   # activates Algorithm 4
)
```

## License

MIT License. See LICENSE file.
