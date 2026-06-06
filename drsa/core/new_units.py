"""
Classification of new units using MILP problems (6), (7) and (8).
Corresponds to: classify_new_alt_v3.m + main.m Steps for new alternatives.

Based on:
Corrente, Greco, Slowiński, Zappalà (2026)
"An explainable and interpretable composite indicator based on decision rules"
Omega 142, 103513.
"""

import numpy as np
from .classifier import classify_units
from .milp import solve_minimal_rules, SCIPY_MILP

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except ImportError:
    pass


def _match_atleast_single(unit: np.ndarray,
                           atleast_rules: np.ndarray,
                           decreasing: list) -> np.ndarray:
    """
    For a single unit, check which at-least rules it matches.
    Corresponds to SupportATLEASTRulesSINGLEVector.m
    Returns binary array (n_rules,)
    """
    ev = unit.copy().astype(float)
    for d in decreasing:
        ev[d] = -ev[d]

    n_rules = len(atleast_rules)
    match = np.zeros(n_rules)
    for i, rule in enumerate(atleast_rules):
        rule_body = rule[:-1]
        crit_pos = np.where(rule_body[0::2] != 0)[0]
        cols = (rule_body[0::2][crit_pos] - 1).astype(int)
        vals = rule_body[1::2][crit_pos]
        if len(cols) > 0:
            match[i] = float(np.all(ev[cols] >= vals))
        else:
            match[i] = 1.0
    return match


def _match_atmost_single(unit: np.ndarray,
                          atmost_rules: np.ndarray,
                          increasing: list) -> np.ndarray:
    """
    For a single unit, check which at-most rules it matches.
    Corresponds to SupportATMOSTRulesSINGLEVector.m
    Returns binary array (n_rules,)
    """
    ev = unit.copy().astype(float)
    for i in increasing:
        ev[i] = -ev[i]

    n_rules = len(atmost_rules)
    match = np.zeros(n_rules)
    for i, rule in enumerate(atmost_rules):
        rule_body = rule[:-1]
        crit_pos = np.where(rule_body[0::2] != 0)[0]
        cols = (rule_body[0::2][crit_pos] - 1).astype(int)
        vals = rule_body[1::2][crit_pos]
        if len(cols) > 0:
            match[i] = float(np.all(ev[cols] >= vals))
        else:
            match[i] = 1.0
    return match


def _createsigma(p: int) -> np.ndarray:
    """
    Build sigma matrix for non-contradictory constraint.
    Corresponds to createsigma.m
    """
    sigma = np.zeros((p, 2 * p))
    y = 0
    for h in range(p):
        sigma[h, y]     = -1
        sigma[h, y + 1] =  1
        y += 2
    return sigma


def _build_milp6(new_units: np.ndarray,
                 matrix_s_minus: np.ndarray,
                 matrix_s_plus: np.ndarray,
                 atleast_rules: np.ndarray,
                 match_al: np.ndarray,
                 atmost_rules: np.ndarray,
                 match_am: np.ndarray,
                 increasing: list,
                 decreasing: list,
                 p: int) -> tuple:
    """
    Build MILP problem (6) for classifying new units.
    Corresponds to classify_new_alt_v3.m

    Variables: [ρ≥ (n_al), ρ≤ (n_am), s⁻(x) (n_new), s⁺(x) (n_new), η≥(a) (n_A), η≤(a) (n_A)]
    """
    n_al   = len(atleast_rules)
    n_am   = len(atmost_rules)
    n_new  = len(new_units)
    n_A    = len(matrix_s_minus)

    # Base MILP from minimal_set_rules
    from .milp import build_milp
    c_base, A_base, b_base, _, _, _ = build_milp(
        matrix_s_minus, match_al,
        matrix_s_plus,  match_am,
        atleast_rules, atmost_rules
    )

    # Extend A with columns for new unit variables [s⁻(x), s⁺(x)] and η
    n_base_vars = n_al + n_am
    # New variables: s⁻(x_k), s⁺(x_k) for each new unit k → 2*n_new vars
    # η variables: η≥(a), η≤(a) for each existing unit → 2*n_A vars
    n_vars_total = n_base_vars + 2 * n_new + 2 * n_A

    # Extend A_base with zero columns for new variables
    A_ext = np.hstack([A_base, np.zeros((A_base.shape[0], 2 * n_new + 2 * n_A))])
    b_ext = b_base.copy()

    pos_start = n_base_vars  # start of s⁻/s⁺ variables for new units

    # Add constraints for each new unit
    d = 0
    for k in range(n_new):
        unit = new_units[k]
        test_al = _match_atleast_single(unit, atleast_rules, decreasing)
        test_am = _match_atmost_single(unit, atmost_rules, increasing)

        # ── At-least constraints ───────────────────────────────────────────
        if np.sum(test_al) == 0:
            # No at-least rule matched → s⁻ free (encoded via η)
            V = np.zeros(n_vars_total)
            V[pos_start + d] = -1
            A_ext = np.vstack([A_ext, V])
            b_ext = np.append(b_ext, -1)
        else:
            # At least one at-least rule matched → must select one
            V = np.zeros(n_vars_total)
            V[:n_al] = -test_al
            A_ext = np.vstack([A_ext, V])
            b_ext = np.append(b_ext, -1)

            # s⁻(x_k) >= d≥_i * ρ≥_i for each matched rule i
            for i in range(n_al):
                if test_al[i] == 1:
                    V = np.zeros(n_vars_total)
                    V[i] = -atleast_rules[i, -1]
                    V[pos_start + d] = 1
                    A_ext = np.vstack([A_ext, V])
                    b_ext = np.append(b_ext, 0)

        # ── At-most constraints ────────────────────────────────────────────
        if np.sum(test_am) == 0:
            # No at-most rule matched → s⁺ = p
            V = np.zeros(n_vars_total)
            V[pos_start + d + 1] = 1
            A_ext = np.vstack([A_ext, V])
            b_ext = np.append(b_ext, p)
        else:
            # At least one at-most rule matched → must select one
            V = np.zeros(n_vars_total)
            V[n_al:n_al + n_am] = -test_am
            A_ext = np.vstack([A_ext, V])
            b_ext = np.append(b_ext, -1)

            # s⁺(x_k) <= d≤_i or p*(1-ρ≤_i) for each matched rule i
            for i in range(n_am):
                if test_am[i] == 1:
                    V = np.zeros(n_vars_total)
                    V[n_al + i] = -(p - atmost_rules[i, -1])
                    V[pos_start + d + 1] = 1
                    A_ext = np.vstack([A_ext, V])
                    b_ext = np.append(b_ext, p)

        d += 2

    # ── Non-contradictory constraint: s⁻(x_k) <= s⁺(x_k) ─────────────────
    sigma = _createsigma(n_new)
    V_dis = np.hstack([np.zeros((n_new, n_base_vars)), -sigma,
                       np.zeros((n_new, 2 * n_A))])
    A_ext = np.vstack([A_ext, V_dis])
    b_ext = np.append(b_ext, np.zeros(n_new))

    # ── η variables constraints (0 <= η <= 1) ──────────────────────────────
    pos_eta = n_base_vars + 2 * n_new

    # Link η to existing unit constraints (relaxed)
    # For units with s⁻_final > 1: constraint relaxed by η≥(a)
    # For units with s⁺_final < p: constraint relaxed by η≤(a)
    d_al = matrix_s_minus[:, -1]
    d_am = matrix_s_plus[:, -1]
    for i in range(n_A):
        if d_al[i] > 1:
            # add η≥(a_i) to relax at-least constraint
            V = np.zeros(n_vars_total)
            V[pos_eta + 2 * i] = -1
            # Find rows in A_ext corresponding to this unit's at-least constraint
            tt_al = (atleast_rules[:, -1] == d_al[i]).astype(float)
            coeff = -tt_al * match_al[i, :]
            row_candidate = np.zeros(n_vars_total)
            row_candidate[:n_al] = coeff
            # Add relaxation row
            A_ext = np.vstack([A_ext, row_candidate + V])
            b_ext = np.append(b_ext, 0)

        if d_am[i] < p:
            V = np.zeros(n_vars_total)
            V[pos_eta + 2 * i + 1] = -1
            tt_am = (atmost_rules[:, -1] == d_am[i]).astype(float)
            coeff = -tt_am * match_am[i, :]
            row_candidate = np.zeros(n_vars_total)
            row_candidate[n_al:n_al + n_am] = coeff
            A_ext = np.vstack([A_ext, row_candidate + V])
            b_ext = np.append(b_ext, 0)

    # ── Objective: minimize Σ η≥(a) + η≤(a) ───────────────────────────────
    c = np.zeros(n_vars_total)
    c[pos_eta:] = 1.0

    return c, A_ext, b_ext, n_base_vars, pos_start, pos_eta, n_vars_total


def classify_new_units(new_units: np.ndarray,
                        matrix_s_minus: np.ndarray,
                        matrix_s_plus: np.ndarray,
                        atleast_rules_max: np.ndarray,
                        match_al: np.ndarray,
                        atmost_rules_max: np.ndarray,
                        match_am: np.ndarray,
                        increasing: list,
                        decreasing: list) -> dict:
    """
    Full classification pipeline for new units.
    Implements equations (4), (6), (7), (8) from the paper.

    Parameters
    ----------
    new_units         : (n_new, n_criteria) — new units WITHOUT class column
    matrix_s_minus    : (n_A, n_criteria+1) — existing units with s⁻ as last col
    matrix_s_plus     : (n_A, n_criteria+1) — existing units with s⁺ as last col
    atleast_rules_max : maximal at-least rules (Step 6)
    match_al          : (n_A, n_al) match matrix for existing units
    atmost_rules_max  : maximal at-most rules (Step 6)
    match_am          : (n_A, n_am) match matrix for existing units
    increasing, decreasing : 0-based criterion indices

    Returns
    -------
    dict with classification results and rule sets
    """
    if not SCIPY_MILP:
        return {'error': 'scipy >= 1.7 required for MILP. Run: pip install scipy'}

    p = int(np.nanmax(matrix_s_plus[:, -1]))
    n_new = len(new_units)
    n_al  = len(atleast_rules_max)
    n_am  = len(atmost_rules_max)

    # ── Step 1: Classify new units with maximal rules (eq. 4) ─────────────
    new_with_nan = np.hstack([new_units, np.full((n_new, 1), np.nan)])
    s_minus_new, s_plus_new, al_m_new, am_m_new = classify_units(
        new_with_nan, atleast_rules_max, atmost_rules_max, increasing, decreasing
    )

    contradictions = np.where(s_minus_new > s_plus_new)[0].tolist()

    result = {
        'step1_s_minus': s_minus_new,
        'step1_s_plus':  s_plus_new,
        'contradictions_step1': contradictions,
        'n_contradictions': len(contradictions),
    }

    if len(contradictions) == 0:
        # No contradictions — just find minimal rules (eq. 8)
        # Combine existing + new units
        all_sm = np.vstack([matrix_s_minus,
                            np.hstack([new_units, s_minus_new.reshape(-1,1)])])
        all_sp = np.vstack([matrix_s_plus,
                            np.hstack([new_units, s_plus_new.reshape(-1,1)])])
        all_match_al = np.vstack([match_al, al_m_new])
        all_match_am = np.vstack([match_am, am_m_new])

        al_min, am_min, al_idx, am_idx, ok, msg = solve_minimal_rules(
            all_sm, all_match_al, all_sp, all_match_am,
            atleast_rules_max, atmost_rules_max
        )
        result.update({
            'milp6_needed': False,
            'step8_al_rules': al_min,
            'step8_am_rules': am_min,
            'final_s_minus': s_minus_new,
            'final_s_plus':  s_plus_new,
            'milp_success': ok,
            'milp_message': msg,
        })
        return result

    # ── Step 2: Solve MILP (6) ─────────────────────────────────────────────
    result['milp6_needed'] = True

    try:
        c6, A6, b6, n_base, pos_s, pos_eta, n_vars = _build_milp6(
            new_units, matrix_s_minus, matrix_s_plus,
            atleast_rules_max, match_al,
            atmost_rules_max, match_am,
            increasing, decreasing, p
        )

        integrality6 = np.ones(n_vars)
        bounds6 = Bounds(lb=np.zeros(n_vars), ub=np.full(n_vars, p))
        # ρ variables must be binary (0 or 1)
        bounds6.lb[:n_base] = 0
        bounds6.ub[:n_base] = 1

        constraints6 = LinearConstraint(A6, -np.inf, b6)
        res6 = milp(c6, constraints=constraints6,
                    integrality=integrality6, bounds=bounds6)

        if not res6.success:
            result['error'] = f"MILP (6) failed: {res6.message}"
            result['final_s_minus'] = s_minus_new
            result['final_s_plus']  = s_plus_new
            return result

        x6 = np.round(res6.x).astype(int)
        eta_star = x6[pos_eta:]
        eta_al = eta_star[0::2]  # η≥(a)
        eta_am = eta_star[1::2]  # η≤(a)

        result['eta_star_al'] = eta_al
        result['eta_star_am'] = eta_am
        result['eta_total']   = int(res6.fun)

        # Units that need reclassification
        reclassify_al = np.where(eta_al == 1)[0].tolist()
        reclassify_am = np.where(eta_am == 1)[0].tolist()
        result['reclassify_units_sminus'] = reclassify_al
        result['reclassify_units_splus']  = reclassify_am

    except Exception as e:
        result['error'] = f"MILP (6) error: {str(e)}"
        result['final_s_minus'] = s_minus_new
        result['final_s_plus']  = s_plus_new
        return result

    # ── Step 3: Solve MILP (7) — maximal rules ────────────────────────────
    try:
        # Fix objective: maximize number of selected rules
        c7 = c6.copy()
        c7[:] = 0
        c7[:n_base] = -1  # maximize ρ = minimize -ρ

        # Add equality constraints for η based on MILP (6) solution
        n_eta = len(eta_star)
        A_eq7 = np.zeros((n_eta, n_vars))
        b_eq7 = np.zeros(n_eta)
        for i in range(n_eta):
            A_eq7[i, pos_eta + i] = 1
            b_eq7[i] = eta_star[i]

        constraints7 = [
            LinearConstraint(A6, -np.inf, b6),
            LinearConstraint(A_eq7, b_eq7, b_eq7)
        ]
        res7 = milp(c7, constraints=constraints7,
                    integrality=integrality6, bounds=bounds6)

        if not res7.success:
            result['error'] = f"MILP (7) failed: {res7.message}"
            result['final_s_minus'] = s_minus_new
            result['final_s_plus']  = s_plus_new
            return result

        x7 = np.round(res7.x).astype(int)
        al_idx7 = np.where(x7[:n_al] == 1)[0].tolist()
        am_idx7 = np.where(x7[n_al:n_al + n_am] == 1)[0].tolist()

        al_max7 = atleast_rules_max[al_idx7] if al_idx7 else np.empty((0, atleast_rules_max.shape[1]))
        am_max7 = atmost_rules_max[am_idx7] if am_idx7 else np.empty((0, atmost_rules_max.shape[1]))

        result['step7_al_rules'] = al_max7
        result['step7_am_rules'] = am_max7

        # Classify all units with MILP (7) rules
        all_units = np.vstack([matrix_s_minus[:, :-1], new_units])
        all_with_nan = np.hstack([all_units, np.full((len(all_units), 1), np.nan)])
        sm7, sp7, al_m7, am_m7 = classify_units(
            all_with_nan, al_max7, am_max7, increasing, decreasing)

        n_existing = len(matrix_s_minus)
        result['final_s_minus'] = sm7[n_existing:]
        result['final_s_plus']  = sp7[n_existing:]
        result['final_s_minus_all'] = sm7
        result['final_s_plus_all']  = sp7

    except Exception as e:
        result['error'] = f"MILP (7) error: {str(e)}"
        result['final_s_minus'] = s_minus_new
        result['final_s_plus']  = s_plus_new
        return result

    # ── Step 4: Solve MILP (8) — minimal rules ────────────────────────────
    try:
        all_sm8 = np.vstack([
            matrix_s_minus,
            np.hstack([new_units, result['final_s_minus'].reshape(-1,1)])
        ])
        all_sp8 = np.vstack([
            matrix_s_plus,
            np.hstack([new_units, result['final_s_plus'].reshape(-1,1)])
        ])
        all_match_al8 = np.vstack([
            match_al[:, al_idx7] if al_idx7 else np.zeros((len(match_al), 0)),
            al_m7[n_existing:, :]
        ])
        all_match_am8 = np.vstack([
            match_am[:, am_idx7] if am_idx7 else np.zeros((len(match_am), 0)),
            am_m7[n_existing:, :]
        ])

        al_min8, am_min8, _, _, ok8, msg8 = solve_minimal_rules(
            all_sm8, all_match_al8,
            all_sp8, all_match_am8,
            al_max7, am_max7
        )
        result['step8_al_rules'] = al_min8
        result['step8_am_rules'] = am_min8
        result['milp_success']   = ok8
        result['milp_message']   = msg8

    except Exception as e:
        result['step8_al_rules'] = al_max7
        result['step8_am_rules'] = am_max7
        result['milp_success']   = True
        result['milp_message']   = f"MILP (8) skipped: {str(e)}"

    return result
