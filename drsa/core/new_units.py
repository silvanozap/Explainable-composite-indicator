"""
Classification of new units — MILP (6), (7), (8).
Direct translation of classify_new_alt_v3.m and main.m.

Variable layout in MILP(6):
  [rho_al(n_al) | rho_am(n_am) | s-(x1),s+(x1),...,s-(xK),s+(xK) | eta(2*n_A)]

A_base structure (from minimal_set_rules.m):
  rows 0..n_A-1:         at-least coverage (b=-1 if s->1, else b=0)
  rows n_A..2*n_A-1:     at-most  coverage (b=-1 if s+<p, else b=0)
  rows 2*n_A..2*n_A+n_r-1:   -I_(n_al+n_am)  rho >= 0
  rows 2*n_A+n_r..2*n_A+2*n_r-1: +I_(n_al+n_am)  rho <= 1

eta columns: -I_(2*n_A) on first 2*n_A rows only.

Corrente, Greco, Slowiński, Zappalà (Omega, 2026).
"""

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_MILP = True
except ImportError:
    SCIPY_MILP = False


def _match_al_single(unit, al_rules, decreasing):
    ev = unit.copy().astype(float)
    for d in decreasing:
        ev[d] = -ev[d]
    out = np.zeros(len(al_rules))
    for i, rule in enumerate(al_rules):
        body = rule[:-1]
        pos  = np.where(body[0::2] != 0)[0]
        cols = (body[0::2][pos] - 1).astype(int)
        vals = body[1::2][pos]
        out[i] = float(np.all(ev[cols] >= vals)) if len(cols) > 0 else 1.0
    return out


def _match_am_single(unit, am_rules, increasing):
    ev = unit.copy().astype(float)
    for i in increasing:
        ev[i] = -ev[i]
    out = np.zeros(len(am_rules))
    for i, rule in enumerate(am_rules):
        body = rule[:-1]
        pos  = np.where(body[0::2] != 0)[0]
        cols = (body[0::2][pos] - 1).astype(int)
        vals = body[1::2][pos]
        out[i] = float(np.all(ev[cols] >= vals)) if len(cols) > 0 else 1.0
    return out


def _match_al_matrix(units, al_rules, decreasing):
    return np.array([_match_al_single(u, al_rules, decreasing) for u in units])


def _match_am_matrix(units, am_rules, increasing):
    return np.array([_match_am_single(u, am_rules, increasing) for u in units])


def _classify_single(unit, al_rules, am_rules, increasing, decreasing, p):
    tal = _match_al_single(unit, al_rules, decreasing)
    tam = _match_am_single(unit, am_rules, increasing)
    s_m = int(np.max(al_rules[tal == 1, -1])) if np.any(tal == 1) else 1
    s_p = int(np.min(am_rules[tam == 1, -1])) if np.any(tam == 1) else p
    return s_m, s_p


def _createsigma(p):
    S = np.zeros((p, 2 * p))
    for h in range(p):
        S[h, 2*h]   = -1
        S[h, 2*h+1] =  1
    return S


def _build_minimal_set_rules(mat_sm, supp_al, mat_sp, supp_am,
                              al_rules, am_rules):
    """
    minimal_set_rules.m — exact translation.

    Returns A, b with structure:
      [n_A rows at-least coverage]
      [n_A rows at-most  coverage]
      [n_r rows -I  (rho >= 0)]
      [n_r rows +I  (rho <= 1)]
    where n_r = n_al + n_am
    """
    p    = int(np.nanmax(mat_sp[:, -1]))
    n_al = len(al_rules)
    n_am = len(am_rules)
    n_A  = len(mat_sm)
    n_r  = n_al + n_am
    d_al = mat_sm[:, -1]
    d_am = mat_sp[:, -1]

    # At-least coverage rows (all n_A, b=-1 only where s->1)
    V_al = np.zeros((n_A, n_r))
    b_al = np.zeros(n_A)
    for i in range(n_A):
        tt = (al_rules[:, -1] == d_al[i]).astype(float)
        V_al[i, :n_al] = -tt * supp_al[i, :]
        if d_al[i] > 1:
            b_al[i] = -1.0

    # At-most coverage rows (all n_A, b=-1 only where s+<p)
    V_am = np.zeros((n_A, n_r))
    b_am = np.zeros(n_A)
    for i in range(n_A):
        tt = (am_rules[:, -1] == d_am[i]).astype(float)
        V_am[i, n_al:] = -tt * supp_am[i, :]
        if d_am[i] < p:
            b_am[i] = -1.0

    # rho >= 0: -I
    V_lb = -np.eye(n_r)
    b_lb = np.zeros(n_r)

    # rho <= 1: +I
    V_ub = np.eye(n_r)
    b_ub = np.ones(n_r)

    A = np.vstack([V_al, V_am, V_lb, V_ub])
    b = np.concatenate([b_al, b_am, b_lb, b_ub])

    return A, b, n_al, n_am, n_A


def classify_new_units(new_units, mat_sm, mat_sp,
                        al_rules_max, match_al,
                        am_rules_max, match_am,
                        increasing, decreasing):
    """
    Full classification of new units following main.m.
    Implements MILP (6), (7), (8).

    Parameters
    ----------
    new_units    : (n_new, n_criteria)
    mat_sm       : (n_A, n_criteria+1) with s- as last col
    mat_sp       : (n_A, n_criteria+1) with s+ as last col
    al_rules_max : maximal at-least rules (Step 6 or Step 7 minimal)
    match_al     : (n_A, n_al) match matrix
    am_rules_max : maximal at-most rules
    match_am     : (n_A, n_am) match matrix
    increasing, decreasing : 0-based criterion indices
    """
    if not SCIPY_MILP:
        return {'error': 'scipy >= 1.7 required. Run: pip install scipy'}

    p     = int(np.nanmax(mat_sp[:, -1]))
    n_new = len(new_units)
    n_al  = len(al_rules_max)
    n_am  = len(am_rules_max)
    n_A   = len(mat_sm)
    n_r   = n_al + n_am
    all_crit_A = mat_sm[:, :-1]

    result = {}

    # Step 1: eq(4) with input rules
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

    # Build A_base exactly like minimal_set_rules.m
    A_base, b_base, _, _, _ = _build_minimal_set_rules(
        mat_sm, match_al, mat_sp, match_am, al_rules_max, am_rules_max)
    n_base_total = A_base.shape[0]   # = 2*n_A + 2*n_r
    n_cov        = 2 * n_A           # coverage rows (eta goes here)

    pos_start = n_r   # start of s variables (after rho)

    # Extend A_base with s columns (2 per new unit)
    A6 = np.hstack([A_base, np.zeros((n_base_total, 2 * n_new))])
    b6 = b_base.copy()

    # Add constraints for each new unit
    d = 0
    for k in range(n_new):
        test_al = _match_al_single(new_units[k], al_rules_max, decreasing)
        test_am = _match_am_single(new_units[k], am_rules_max, increasing)
        nc = A6.shape[1]

        # At-least constraints
        if np.sum(test_al) == 0:
            V = np.zeros(nc); V[pos_start + d] = -1
            A6 = np.vstack([A6, V]); b6 = np.append(b6, -1.0)
        else:
            # At least one matched rule must be selected
            V = np.zeros(nc); V[:n_al] = -test_al
            A6 = np.vstack([A6, V]); b6 = np.append(b6, -1.0)
            # MATLAB: diag(TEST.*d>=)*rho - s-(xk) <= 0 for ALL n_al rules
            # Row i: TEST_i * d>=_i * rho_i - s-(xk) <= 0
            # For TEST_i=0: 0 - s-(xk) <= 0 → s-(xk) >= 0 (redundant but present)
            for i in range(n_al):
                V = np.zeros(nc)
                V[i] = test_al[i] * al_rules_max[i, -1]   # +TEST_i * d>=_i * rho_i
                V[pos_start + d] = -1.0                     # -s-(xk)
                A6 = np.vstack([A6, V]); b6 = np.append(b6, 0.0)

        # At-most constraints
        if np.sum(test_am) == 0:
            V = np.zeros(nc); V[pos_start + d + 1] = 1.0
            A6 = np.vstack([A6, V]); b6 = np.append(b6, float(p))
        else:
            # At least one matched rule must be selected
            V = np.zeros(nc); V[n_al:n_r] = -test_am
            A6 = np.vstack([A6, V]); b6 = np.append(b6, -1.0)
            # s+(xk) <= d<=_i + (p-d<=_i)*(1-rho<=_i)
            for i in range(n_am):
                if test_am[i] == 1:
                    V = np.zeros(nc)
                    V[n_al + i]          = -am_rules_max[i, -1] + p
                    V[pos_start + d + 1] = 1.0
                    A6 = np.vstack([A6, V]); b6 = np.append(b6, float(p))
        d += 2

    # Non-contradictory: s-(xk) - s+(xk) <= 0
    neg_sigma = -_createsigma(n_new)
    A6 = np.vstack([A6, np.hstack([np.zeros((n_new, n_r)), neg_sigma])])
    b6 = np.append(b6, np.zeros(n_new))

    # Add eta columns: -I_(2*n_A) on first n_cov=2*n_A rows, zeros elsewhere
    posizione  = A6.shape[1]   # = n_r + 2*n_new
    n_rows_now = A6.shape[0]
    eta_cols   = np.zeros((n_rows_now, 2 * n_A))
    eta_cols[:n_cov, :] = -np.eye(2 * n_A)   # only on coverage rows
    A6 = np.hstack([A6, eta_cols])

    # eta >= 0
    A6 = np.vstack([A6, np.hstack([np.zeros((2*n_A, posizione)), -np.eye(2*n_A)])])
    b6 = np.append(b6, np.zeros(2 * n_A))
    # eta <= 1
    A6 = np.vstack([A6, np.hstack([np.zeros((2*n_A, posizione)), np.eye(2*n_A)])])
    b6 = np.append(b6, np.ones(2 * n_A))

    # Objective: minimize sum(eta)
    f6 = np.hstack([np.zeros(posizione), np.ones(2 * n_A)])
    n_vars6 = A6.shape[1]

    # Bounds: rho in [0,1], s in [1,p], eta in [0,1]
    lb6 = np.zeros(n_vars6)
    ub6 = np.ones(n_vars6)
    for k in range(n_new):
        lb6[pos_start + 2*k]     = 1.0; ub6[pos_start + 2*k]     = float(p)
        lb6[pos_start + 2*k + 1] = 1.0; ub6[pos_start + 2*k + 1] = float(p)

    bounds6      = Bounds(lb=lb6, ub=ub6)
    integrality6 = np.ones(n_vars6)
    constraints6 = LinearConstraint(A6, -np.inf, b6)

    res6 = milp(f6, constraints=constraints6,
                integrality=integrality6, bounds=bounds6)

    if not res6.success:
        result['error'] = f"MILP (6) failed: {res6.message}"
        result['final_s_minus'] = s_minus_new
        result['final_s_plus']  = s_plus_new
        return result

    x6  = np.round(res6.x).astype(int)
    eta = x6[posizione:]
    result['eta']       = eta
    result['eta_total'] = int(res6.fun)

    # MILP (7): fix eta, maximize sum(rho)
    f7 = np.hstack([-np.ones(n_r), np.zeros(n_vars6 - n_r)])
    n_eta = 2 * n_A
    A_eq  = np.zeros((n_eta, n_vars6))
    for i in range(n_eta):
        A_eq[i, posizione + i] = 1.0
    b_eq = eta.astype(float)

    constraints7 = [
        LinearConstraint(A6,   -np.inf, b6),
        LinearConstraint(A_eq, b_eq,    b_eq),
    ]
    res7 = milp(f7, constraints=constraints7,
                integrality=integrality6, bounds=bounds6)

    if not res7.success:
        result['error'] = f"MILP (7) failed: {res7.message}"
        result['final_s_minus'] = s_minus_new
        result['final_s_plus']  = s_plus_new
        return result

    x7      = np.round(res7.x).astype(int)
    al_idx7 = np.where(x7[:n_al] == 1)[0].tolist()
    am_idx7 = np.where(x7[n_al:n_r] == 1)[0].tolist()
    al_max7 = al_rules_max[al_idx7] if al_idx7 else np.empty((0, al_rules_max.shape[1]))
    am_max7 = am_rules_max[am_idx7] if am_idx7 else np.empty((0, am_rules_max.shape[1]))

    result['step7_al_rules'] = al_max7
    result['step7_am_rules'] = am_max7

    # Classify A union A_new with MILP(7) rules
    all_units = np.vstack([all_crit_A, new_units])
    cl_all7   = np.array([
        _classify_single(all_units[k], al_max7, am_max7, increasing, decreasing, p)
        for k in range(len(all_units))
    ])
    cl_A7   = cl_all7[:n_A]
    cl_new7 = cl_all7[n_A:]

    # Update s+ for changed units (main.m line 135)
    changed = np.where(
        ~np.all(cl_A7 == np.column_stack([mat_sm[:,-1], mat_sp[:,-1]]), axis=1)
    )[0].tolist()
    result['changed_units'] = changed
    mat_sp_upd = mat_sp.copy()
    for i in changed:
        mat_sp_upd[i, -1] = cl_A7[i, 1]

    # MILP (8): minimal rules for A union A_new
    new_sm8 = np.hstack([new_units, cl_new7[:, 0:1]])
    new_sp8 = np.hstack([new_units, cl_new7[:, 1:2]])
    all_sm8 = np.vstack([mat_sm,     new_sm8])
    all_sp8 = np.vstack([mat_sp_upd, new_sp8])

    # Support matrices for MILP(7) rules
    mal7_A   = match_al[:, al_idx7] if al_idx7 else np.zeros((n_A, 0))
    mam7_A   = match_am[:, am_idx7] if am_idx7 else np.zeros((n_A, 0))
    mal7_new = _match_al_matrix(new_units, al_max7, decreasing) if len(al_max7)>0 else np.zeros((n_new,0))
    mam7_new = _match_am_matrix(new_units, am_max7, increasing)  if len(am_max7)>0 else np.zeros((n_new,0))
    all_mal8 = np.vstack([mal7_A, mal7_new])
    all_mam8 = np.vstack([mam7_A, mam7_new])

    n_al7 = len(al_max7); n_am7 = len(am_max7)
    if n_al7 + n_am7 > 0:
        A8, b8, _, _, _ = _build_minimal_set_rules(
            all_sm8, all_mal8, all_sp8, all_mam8, al_max7, am_max7)
        n8 = n_al7 + n_am7
        # Only use coverage rows for MILP(8) (not rho bounds — handled by bounds)
        n_cov8 = 2 * len(all_sm8)
        A8_cov = A8[:n_cov8, :n8]
        b8_cov = b8[:n_cov8]
        res8 = milp(np.ones(n8),
                    constraints=LinearConstraint(A8_cov, -np.inf, b8_cov),
                    integrality=np.ones(n8),
                    bounds=Bounds(lb=np.zeros(n8), ub=np.ones(n8)))
        if res8.success:
            x8      = np.round(res8.x).astype(int)
            al_min8 = al_max7[np.where(x8[:n_al7]==1)[0]] if np.any(x8[:n_al7]==1) else np.empty((0,al_max7.shape[1]))
            am_min8 = am_max7[np.where(x8[n_al7:]==1)[0]] if np.any(x8[n_al7:]==1) else np.empty((0,am_max7.shape[1]))
            result.update({
                'step8_al_rules': al_min8, 'step8_am_rules': am_min8,
                'milp_success': True,
                'milp_message': f"Minimal: {len(al_min8)} at-least, {len(am_min8)} at-most"
            })
        else:
            result.update({
                'step8_al_rules': al_max7, 'step8_am_rules': am_max7,
                'milp_success': False,
                'milp_message': f"MILP (8) failed: {res8.message}"
            })
    else:
        result.update({
            'step8_al_rules': np.empty((0,)), 'step8_am_rules': np.empty((0,)),
            'milp_success': True, 'milp_message': "No rules available"
        })

    result['final_s_minus'] = cl_new7[:, 0].astype(float)
    result['final_s_plus']  = cl_new7[:, 1].astype(float)
    result['classification_all'] = cl_all7
    return result
