"""
MILP formulation for finding the minimal set of rules.
Corresponds to: minimal_set_rules.m (problem E_min, equation 5 in the paper)
Uses scipy.optimize.milp (available since scipy 1.7)
"""

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP = True
except ImportError:
    SCIPY_MILP = False


def build_milp(matrix_s_minus: np.ndarray,
               match_al: np.ndarray,
               matrix_s_plus: np.ndarray,
               match_am: np.ndarray,
               atleast_rules: np.ndarray,
               atmost_rules: np.ndarray) -> tuple:
    """
    Build the MILP problem for minimal rule set selection.
    Corresponds to minimal_set_rules.m

    Variables: ρ≥_i ∈ {0,1} for at-least rules,
               ρ≤_i ∈ {0,1} for at-most rules
    Objective: minimize Σρ≥_i + Σρ≤_i
    Constraints: for each unit a with s⁻(a)>1, at least one matching
                 at-least rule with d≥=s⁻(a) must be selected;
                 analogously for at-most rules.

    Returns
    -------
    c       : objective coefficients
    A_ub    : inequality constraint matrix (Ax <= b)
    b_ub    : inequality RHS
    integrality : array of 1s (all variables integer)
    bounds  : (lb, ub) = (0, 1)
    n_al, n_am : number of at-least and at-most rules
    """
    p = int(np.nanmax(matrix_s_plus[:, -1]))
    n_al = len(atleast_rules)
    n_am = len(atmost_rules)
    n_vars = n_al + n_am
    n_units = len(matrix_s_minus)

    d_al = matrix_s_minus[:, -1]
    d_am = matrix_s_plus[:, -1]

    rows_A = []
    rows_b = []

    # At-least constraints: for each unit with s⁻ > 1
    for i in range(n_units):
        if d_al[i] <= 1:
            continue
        # Rule must match unit AND have decision == s⁻(i)
        tt_al = (atleast_rules[:, -1] == d_al[i]).astype(float)
        coeff = -tt_al * match_al[i, :]
        row = np.zeros(n_vars)
        row[:n_al] = coeff
        rows_A.append(row)
        rows_b.append(-1.0)

    # At-most constraints: for each unit with s⁺ < p
    for i in range(n_units):
        if d_am[i] >= p:
            continue
        tt_am = (atmost_rules[:, -1] == d_am[i]).astype(float)
        coeff = -tt_am * match_am[i, :]
        row = np.zeros(n_vars)
        row[n_al:] = coeff
        rows_A.append(row)
        rows_b.append(-1.0)

    if rows_A:
        A_ub = np.array(rows_A)
        b_ub = np.array(rows_b)
    else:
        A_ub = np.zeros((1, n_vars))
        b_ub = np.zeros(1)

    c = np.ones(n_vars)
    integrality = np.ones(n_vars)

    return c, A_ub, b_ub, integrality, n_al, n_am


def solve_minimal_rules(matrix_s_minus: np.ndarray,
                         match_al: np.ndarray,
                         matrix_s_plus: np.ndarray,
                         match_am: np.ndarray,
                         atleast_rules: np.ndarray,
                         atmost_rules: np.ndarray) -> tuple:
    """
    Solve MILP to find the minimal set of rules.

    Returns
    -------
    al_minimal : selected at-least rules
    am_minimal : selected at-most rules
    al_idx     : 0-based indices of selected at-least rules
    am_idx     : 0-based indices of selected at-most rules
    success    : bool
    message    : str
    """
    if not SCIPY_MILP:
        return None, None, [], [], False, "scipy.optimize.milp not available. Please install scipy >= 1.7"

    c, A_ub, b_ub, integrality, n_al, n_am = build_milp(
        matrix_s_minus, match_al, matrix_s_plus, match_am,
        atleast_rules, atmost_rules
    )

    n_vars = n_al + n_am
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if result.success:
        x = np.round(result.x).astype(int)
        al_idx = np.where(x[:n_al] == 1)[0].tolist()
        am_idx = np.where(x[n_al:] == 1)[0].tolist()
        al_minimal = atleast_rules[al_idx] if al_idx else np.empty((0, atleast_rules.shape[1]))
        am_minimal = atmost_rules[am_idx] if am_idx else np.empty((0, atmost_rules.shape[1]))
        return al_minimal, am_minimal, al_idx, am_idx, True, "Optimal solution found"
    else:
        return None, None, [], [], False, f"MILP solver failed: {result.message}"
