"""
Classification of new units using MILP problems (6), (7) and (8).
Corresponds exactly to: classify_new_alt_v3.m + main.m Steps for new alternatives.

Variable layout in MILP (6):
  [ρ≥ (n_al) | ρ≤ (n_am) | s⁻(x_k), s⁺(x_k) for k=1..n_new (2*n_new) | η (2*n_A)]

Based on:
Corrente, Greco, Slowiński, Zappalà (2026)
Omega 142, 103513.
"""

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP = True
except ImportError:
    SCIPY_MILP = False


def _match_atleast_single(unit, atleast_rules, decreasing):
    """
    SupportATLEASTRulesSINGLEVector.m
    Returns binary array (n_al,) — 1 if unit matches rule condition.
    """
    ev = unit.copy().astype(float)
    for d in decreasing:
        ev[d] = -ev[d]
    n_rules = len(atleast_rules)
    match = np.zeros(n_rules)
    for i, rule in enumerate(atleast_rules):
        body = rule[:-1]
        pos = np.where(body[0::2] != 0)[0]
        cols = (body[0::2][pos] - 1).astype(int)
        vals = body[1::2][pos]
        if len(cols) > 0:
            match[i] = float(np.all(ev[cols] >= vals))
        else:
            match[i] = 1.0
    return match


def _match_atmost_single(unit, atmost_rules, increasing):
    """
    SupportATMOSTRulesSINGLEVector.m
    Returns binary array (n_am,).
    """
    ev = unit.copy().astype(float)
    for i in increasing:
        ev[i] = -ev[i]
    n_rules = len(atmost_rules)
    match = np.zeros(n_rules)
    for i, rule in enumerate(atmost_rules):
        body = rule[:-1]
        pos = np.where(body[0::2] != 0)[0]
        cols = (body[0::2][pos] - 1).astype(int)
        vals = body[1::2][pos]
        if len(cols) > 0:
            match[i] = float(np.all(ev[cols] >= vals))
        else:
            match[i] = 1.0
    return match


def _match_atleast_matrix(units, atleast_rules, decreasing):
    """Match matrix for multiple units. Returns (n_units, n_al)."""
    return np.array([_match_atleast_single(u, atleast_rules, decreasing) for u in units])


def _match_atmost_matrix(units, atmost_rules, increasing):
    """Match matrix for multiple units. Returns (n_units, n_am)."""
    return np.array([_match_atmost_single(u, atmost_rules, increasing) for u in units])


def _createsigma(p):
    """createsigma.m — non-contradictory constraint matrix."""
    sigma = np.zeros((p, 2 * p))
    for h in range(p):
        sigma[h, 2*h]   = -1
        sigma[h, 2*h+1] =  1
    return sigma


def _classify_single(unit, atleast_rules, atmost_rules, increasing, decreasing, p):
    """CLASSIFICATIONFunction.m — eq. (4) for a single unit."""
    test_al = _match_atleast_single(unit, atleast_rules, decreasing)
    test_am = _match_atmost_single(unit, atmost_rules, increasing)
    s_minus = int(np.max(atleast_rules[test_al == 1, -1])) if np.any(test_al == 1) else 1
    s_plus  = int(np.min(atmost_rules[test_am == 1, -1])) if np.any(test_am == 1) else p
    return s_minus, s_plus


def _build_minimal_set_rules(mat_sm, match_al, mat_sp, match_am, al_rules, am_rules):
    """
    minimal_set_rules.m
    Returns (c, A_ub, b_ub) for the MILP min Σρ s.t. coverage constraints.
    Variables: [ρ≥ (n_al) | ρ≤ (n_am)]
    """
    p     = int(np.nanmax(mat_sp[:, -1]))
    n_al  = len(al_rules)
    n_am  = len(am_rules)
    n_A   = len(mat_sm)
    d_al  = mat_sm[:, -1]
    d_am  = mat_sp[:, -1]

    rows_A, rows_b = [], []

    for i in range(n_A):
        if d_al[i] > 1:
            tt = (al_rules[:, -1] == d_al[i]).astype(float)
            row = np.zeros(n_al + n_am)
            row[:n_al] = -tt * match_al[i, :]
            rows_A.append(row); rows_b.append(-1.0)

    for i in range(n_A):
        if d_am[i] < p:
            tt = (am_rules[:, -1] == d_am[i]).astype(float)
            row = np.zeros(n_al + n_am)
            row[n_al:] = -tt * match_am[i, :]
            rows_A.append(row); rows_b.append(-1.0)

    if rows_A:
        A_ub = np.array(rows_A)
        b_ub = np.array(rows_b)
    else:
        A_ub = np.zeros((1, n_al + n_am))
        b_ub = np.zeros(1)

    c = np.ones(n_al + n_am)
    return c, A_ub, b_ub


def _build_classify_new_alt(new_units, mat_sm, mat_sp, al_rules, am_rules,
                             match_al, match_am, increasing, decreasing, p):
    """
    classify_new_alt_v3.m
    Builds the MILP (6) system.

    Variables: [ρ≥(n_al) | ρ≤(n_am) | s⁻(x_k),s⁺(x_k) k=1..n_new | η(2*n_A)]
    Objective: minimize Σ η  (= find which existing units must be reclassified)
    """
    n_al  = len(al_rules)
    n_am  = len(am_rules)
    n_new = len(new_units)
    n_A   = len(mat_sm)

    # Base constraints from minimal_set_rules (on existing units A)
    _, A_base, b_base = _build_minimal_set_rules(
        mat_sm, match_al, mat_sp, match_am, al_rules, am_rules)

    # Total variables: ρ≥, ρ≤, s⁻/s⁺ for each new unit, η for each existing unit
    n_vars = n_al + n_am + 2 * n_new + 2 * n_A
    pos_s   = n_al + n_am          # start of s⁻/s⁺ variables
    pos_eta = n_al + n_am + 2*n_new  # start of η variables

    # Extend A_base with zeros for new variable columns
    A = np.hstack([A_base, np.zeros((A_base.shape[0], 2*n_new + 2*n_A))])
    b = b_base.copy()

    # ── Constraints for each new unit x_k ─────────────────────────────────
    d = 0  # offset into s⁻/s⁺ variables (2 per unit)
    for k in range(n_new):
        test_al = _match_atleast_single(new_units[k], al_rules, decreasing)
        test_am = _match_atmost_single(new_units[k], al_rules if False else am_rules, increasing)

        # ── At-least constraints ──────────────────────────────────────────
        if np.sum(test_al) == 0:
            # No at-least rule matches → s⁻(x_k) free, set via η-like variable
            row = np.zeros(n_vars)
            row[pos_s + d] = -1
            A = np.vstack([A, row]); b = np.append(b, -1.0)
        else:
            # At least one rule selected must match x_k: Σ ρ≥_i * TEST_i >= 1
            row = np.zeros(n_vars)
            row[:n_al] = -test_al
            A = np.vstack([A, row]); b = np.append(b, -1.0)

            # s⁻(x_k) >= d≥_i * ρ≥_i for each matched rule i
            # → d≥_i * ρ≥_i - s⁻(x_k) <= 0
            for i in range(n_al):
                if test_al[i] == 1:
                    row = np.zeros(n_vars)
                    row[i]          = -al_rules[i, -1]   # -d≥_i * ρ≥_i
                    row[pos_s + d]  =  1                  # +s⁻(x_k)
                    A = np.vstack([A, row]); b = np.append(b, 0.0)

        # ── At-most constraints ───────────────────────────────────────────
        if np.sum(test_am) == 0:
            # No at-most rule matches → s⁺(x_k) = p
            row = np.zeros(n_vars)
            row[pos_s + d + 1] = 1
            A = np.vstack([A, row]); b = np.append(b, float(p))
        else:
            # At least one rule selected must match x_k
            row = np.zeros(n_vars)
            row[n_al:n_al+n_am] = -test_am
            A = np.vstack([A, row]); b = np.append(b, -1.0)

            # s⁺(x_k) <= d≤_i + (p - d≤_i)*(1 - ρ≤_i) for matched rules
            # → -(p - d≤_i)*ρ≤_i + s⁺(x_k) <= p
            for i in range(n_am):
                if test_am[i] == 1:
                    row = np.zeros(n_vars)
                    row[n_al + i]      = -(p - am_rules[i, -1])
                    row[pos_s + d + 1] = 1
                    A = np.vstack([A, row]); b = np.append(b, float(p))

        d += 2

    # ── Non-contradictory: s⁻(x_k) <= s⁺(x_k) ────────────────────────────
    # → -s⁻(x_k) + s⁺(x_k) >= 0  →  s⁻(x_k) - s⁺(x_k) <= 0
    sigma = _createsigma(n_new)   # (n_new, 2*n_new): [-1,+1] per row
    # -sigma gives [+1,-1] per row: s⁻ - s⁺ <= 0
    neg_sigma = -sigma
    row_block = np.hstack([
        np.zeros((n_new, n_al+n_am)),
        neg_sigma,
        np.zeros((n_new, 2*n_A))
    ])
    A = np.vstack([A, row_block])
    b = np.append(b, np.zeros(n_new))

    # ── η variables: link to existing unit constraints ─────────────────────
    # The first 2*n_A rows of A_base correspond to at-least and at-most
    # constraints for existing units. η_i relaxes constraint i:
    # original: -Σ ρ * match <= -1  → with η: -Σ ρ * match - η_i <= -1
    # → add -η_i to the RHS: -Σ ρ * match <= -1 + η_i
    # Implemented as: add column -η_i to A, η_i ∈ {0,1}

    # Add η columns to the existing constraint rows (first rows of A)
    n_base_rows = A_base.shape[0]
    eta_cols = np.zeros((A.shape[0], 2*n_A))

    # al constraints: first n_A_al rows → η≥(a_i) at pos 2*i
    d_al = mat_sm[:, -1]
    d_am = mat_sp[:, -1]
    al_row = 0
    am_row = 0
    for i in range(n_A):
        if d_al[i] > 1:
            eta_cols[al_row, 2*i] = -1   # -η≥(a_i)
            al_row += 1
    for i in range(n_A):
        if d_am[i] < p:
            eta_cols[n_A + am_row, 2*i+1] = -1  # -η≤(a_i)
            am_row += 1

    A = np.hstack([A, eta_cols])

    # η bounds: 0 <= η <= 1, integer
    # Already handled via variable bounds below

    # ── Objective: minimize Σ η ────────────────────────────────────────────
    c = np.zeros(A.shape[1])
    c[pos_eta:] = 1.0

    return c, A, b, n_al, n_am, n_new, n_A, pos_s, pos_eta


def classify_new_units(new_units, mat_sm, mat_sp,
                        al_rules_max, match_al,
                        am_rules_max, match_am,
                        increasing, decreasing):
    """
    Full classification of new units following main.m Steps 1-3 for A_new.

    Parameters
    ----------
    new_units   : (n_new, n_criteria) — new units without class column
    mat_sm      : (n_A, n_criteria+1) — existing units with s⁻ as last col
    mat_sp      : (n_A, n_criteria+1) — existing units with s⁺ as last col
    al_rules_max: maximal at-least rules (Step 6)
    match_al    : (n_A, n_al) match matrix for existing units
    am_rules_max: maximal at-most rules (Step 6)
    match_am    : (n_A, n_am) match matrix for existing units
    increasing, decreasing: 0-based criterion indices

    Returns
    -------
    dict with all intermediate and final results
    """
    if not SCIPY_MILP:
        return {'error': 'scipy >= 1.7 required. Run: pip install scipy'}

    p     = int(np.nanmax(mat_sp[:, -1]))
    n_new = len(new_units)
    n_al  = len(al_rules_max)
    n_am  = len(am_rules_max)
    n_A   = len(mat_sm)
    all_crit_A = mat_sm[:, :-1]

    result = {}

    # ── Step 1: Classify new units with maximal rules (eq. 4) ─────────────
    cl_new = np.array([
        _classify_single(new_units[k], al_rules_max, am_rules_max,
                         increasing, decreasing, p)
        for k in range(n_new)
    ])
    s_minus_new = cl_new[:, 0].astype(float)
    s_plus_new  = cl_new[:, 1].astype(float)
    contradictions = np.where(s_minus_new > s_plus_new)[0].tolist()

    result.update({
        'step1_s_minus': s_minus_new.copy(),
        'step1_s_plus':  s_plus_new.copy(),
        'contradictions_step1': contradictions,
        'n_contradictions': len(contradictions),
    })

    # ── MILP (6): build and solve ──────────────────────────────────────────
    c6, A6, b6, n_al, n_am, n_new, n_A, pos_s, pos_eta = _build_classify_new_alt(
        new_units, mat_sm, mat_sp,
        al_rules_max, am_rules_max,
        match_al, match_am,
        increasing, decreasing, p
    )

    n_vars = A6.shape[1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    # s⁻ and s⁺ variables can range from 1 to p
    for k in range(n_new):
        lb[pos_s + 2*k]     = 1.0
        ub[pos_s + 2*k]     = float(p)
        lb[pos_s + 2*k + 1] = 1.0
        ub[pos_s + 2*k + 1] = float(p)

    bounds6 = Bounds(lb=lb, ub=ub)
    integrality6 = np.ones(n_vars)
    constraints6 = LinearConstraint(A6, -np.inf, b6)

    res6 = milp(c6, constraints=constraints6,
                integrality=integrality6, bounds=bounds6)

    if not res6.success:
        result['error'] = f"MILP (6) failed: {res6.message}"
        result['final_s_minus'] = s_minus_new
        result['final_s_plus']  = s_plus_new
        return result

    x6    = np.round(res6.x).astype(int)
    eta   = x6[pos_eta:]           # shape (2*n_A,)
    eta_bool = eta == 1

    result['eta'] = eta
    result['eta_total'] = int(res6.fun)
    result['reclassify'] = np.where(eta_bool)[0].tolist()

    # ── MILP (7): fix η, maximize Σ ρ ─────────────────────────────────────
    # Same constraints as MILP (6) plus equality constraints on η
    c7 = np.zeros(n_vars)
    c7[:n_al + n_am] = -1.0   # maximize ρ = minimize -ρ

    # Equality constraints: η_i = eta_bool[i]
    n_eta = len(eta)
    A_eq = np.zeros((n_eta, n_vars))
    b_eq_lo = np.zeros(n_eta)
    b_eq_hi = np.zeros(n_eta)
    for i in range(n_eta):
        A_eq[i, pos_eta + i] = 1.0
        b_eq_lo[i] = float(eta[i])
        b_eq_hi[i] = float(eta[i])

    constraints7 = [
        LinearConstraint(A6,   -np.inf, b6),
        LinearConstraint(A_eq, b_eq_lo, b_eq_hi),
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
    am_idx7 = np.where(x7[n_al:n_al+n_am] == 1)[0].tolist()

    al_max7 = al_rules_max[al_idx7] if al_idx7 else np.empty((0, al_rules_max.shape[1]))
    am_max7 = am_rules_max[am_idx7] if am_idx7 else np.empty((0, am_rules_max.shape[1]))

    result['step7_al_rules'] = al_max7
    result['step7_am_rules'] = am_max7

    # ── Classify ALL units (A ∪ A_new) with MILP (7) rules ────────────────
    all_units   = np.vstack([all_crit_A, new_units])
    cl_all = np.array([
        _classify_single(all_units[k], al_max7, am_max7, increasing, decreasing, p)
        for k in range(len(all_units))
    ])
    cl_A   = cl_all[:n_A]
    cl_new7 = cl_all[n_A:]

    # ── Find which existing units changed classification ───────────────────
    s_minus_orig = mat_sm[:, -1]
    s_plus_orig  = mat_sp[:, -1]
    changed = np.where(
        ~np.all(cl_A == np.column_stack([s_minus_orig, s_plus_orig]), axis=1)
    )[0].tolist()
    result['changed_units'] = changed

    # ── Step 3: Update s⁺ for changed units, build support matrices ────────
    # main.m line 135: MatrixEvaluations_2_s_plus(I,end) = CLASSIFICATION_new_maximal(I,2)
    mat_sp_updated = mat_sp.copy()
    for i in changed:
        mat_sp_updated[i, -1] = cl_A[i, 1]

    # Build combined matrices A ∪ A_new
    new_sm = np.hstack([new_units, cl_new7[:, 0:1]])
    new_sp = np.hstack([new_units, cl_new7[:, 1:2]])
    all_sm = np.vstack([mat_sm, new_sm])
    all_sp = np.vstack([mat_sp_updated, new_sp])

    # Support matrices for al_max7 and am_max7
    match_al7_A   = match_al[:, al_idx7] if al_idx7 else np.zeros((n_A, 0))
    match_am7_A   = match_am[:, am_idx7] if am_idx7 else np.zeros((n_A, 0))
    match_al7_new = _match_atleast_matrix(new_units, al_max7, decreasing) if len(al_max7)>0 else np.zeros((n_new,0))
    match_am7_new = _match_atmost_matrix(new_units, am_max7, increasing)  if len(am_max7)>0 else np.zeros((n_new,0))

    all_match_al = np.vstack([match_al7_A,   match_al7_new])
    all_match_am = np.vstack([match_am7_A,   match_am7_new])

    # ── MILP (8): minimal rules for A ∪ A_new ─────────────────────────────
    if len(al_max7) > 0 or len(am_max7) > 0:
        c8, A8, b8 = _build_minimal_set_rules(
            all_sm, all_match_al, all_sp, all_match_am, al_max7, am_max7)

        n8 = len(al_max7) + len(am_max7)
        bounds8 = Bounds(lb=np.zeros(n8), ub=np.ones(n8))
        constraints8 = LinearConstraint(A8, -np.inf, b8)

        res8 = milp(c8, constraints=constraints8,
                    integrality=np.ones(n8), bounds=bounds8)

        if res8.success:
            x8 = np.round(res8.x).astype(int)
            n_al7 = len(al_max7)
            al_idx8 = np.where(x8[:n_al7] == 1)[0].tolist()
            am_idx8 = np.where(x8[n_al7:] == 1)[0].tolist()
            al_min8 = al_max7[al_idx8] if al_idx8 else np.empty((0, al_max7.shape[1]))
            am_min8 = am_max7[am_idx8] if am_idx8 else np.empty((0, am_max7.shape[1]))
            result['step8_al_rules'] = al_min8
            result['step8_am_rules'] = am_min8
            result['milp_success']   = True
            result['milp_message']   = f"Optimal: {len(al_min8)} at-least, {len(am_min8)} at-most minimal rules"
        else:
            result['step8_al_rules'] = al_max7
            result['step8_am_rules'] = am_max7
            result['milp_success']   = False
            result['milp_message']   = f"MILP (8) failed: {res8.message}"
    else:
        result['step8_al_rules'] = np.empty((0,))
        result['step8_am_rules'] = np.empty((0,))
        result['milp_success']   = True
        result['milp_message']   = "No rules available"

    result['final_s_minus'] = cl_new7[:, 0].astype(float)
    result['final_s_plus']  = cl_new7[:, 1].astype(float)
    result['classification_all'] = cl_all

    return result
