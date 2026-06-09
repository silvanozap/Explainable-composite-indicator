"""
Utility functions for DRSA rule induction.
Converted from MATLAB by Corrente, Greco, Slowiński, Zappalà (Omega, 2026).
"""

import numpy as np
from itertools import combinations


def invert_direction(evaluations: np.ndarray, increasing: list, decreasing: list) -> np.ndarray:
    """
    Invert decreasing criteria so all criteria become gain-type.
    Corresponds to invertdirection.m
    """
    ev = evaluations.copy().astype(float)
    for i in decreasing:
        ev[:, i] = -ev[:, i]
    return ev


def reverse_class(classes: np.ndarray) -> np.ndarray:
    """
    Reverse class ordering: maps [min..max] -> [max..min].
    Corresponds to reverseclass.m
    """
    min_val = np.nanmin(classes)
    max_val = np.nanmax(classes)
    return max_val - (classes - min_val)


def from_binary_to_number(P: np.ndarray, G: int) -> np.ndarray:
    """
    Sort subsets of criteria by cardinality (Hamming weight), then by
    descending binary value within the same cardinality.
    Corresponds to from_binary_to_number.m
    """
    w = P.sum(axis=1)
    weights = 2 ** np.arange(G - 1, -1, -1)
    val = P @ weights
    idx = np.lexsort((val, w))  # primary: w asc, secondary: val desc -> use -val trick
    # MATLAB sorts by [w, -val] ascending, which means w asc then val desc
    idx = np.lexsort((-val, w))
    return P[idx]


def create_subset(p_select: np.ndarray) -> np.ndarray:
    """
    Generate all proper non-empty subsets of the criteria in p_select.
    Corresponds to create_subset.m
    Returns a 2D array where each row is a binary indicator vector.
    """
    idx = np.where(p_select == 1)[0]
    n = len(idx)
    subsets = []
    for k in range(1, n + 1):
        for combo in combinations(idx, k):
            vec = np.zeros(len(p_select), dtype=int)
            vec[list(combo)] = 1
            subsets.append(vec)
    return np.array(subsets) if subsets else np.empty((0, len(p_select)), dtype=int)


def rule_add(ev_a: np.ndarray, p_select: np.ndarray, t: int) -> np.ndarray:
    """
    Create a candidate rule from unit a, subset P, class t.
    Corresponds to rule_add.m
    """
    r = np.full(len(ev_a), np.nan)
    r[p_select == 1] = ev_a[p_select == 1]
    return np.append(r, t)


def lower_approximation(a: int, t: int, ev: np.ndarray, ex_class: np.ndarray,
                         p_select: np.ndarray, missing: bool = False):
    """
    Compute credibility L^P_{>=t}(a) and the dominance set indicator I.
    Corresponds to lowerapproximation.m (Algorithm 1 step 6).

    If missing=True, uses Algorithm 4 dominance: a' dominates a if
    g_j(a') >= g_j(a) OR g_j(a') is NaN, for all j in P.
    """
    cols = np.where(p_select == 1)[0]
    ev_a_P = ev[a, cols]

    if missing:
        # Algorithm 4: NaN in a' counts as dominating
        def dominates_row(row):
            for c, val_a in zip(cols, ev_a_P):
                rv = row[c]
                if not np.isnan(rv) and rv < val_a:
                    return False
            return True
        I = np.array([dominates_row(ev[i]) for i in range(len(ev))])
    else:
        # Algorithm 1: replace NaN with column max before comparison
        ev_work = ev.copy()
        col_max = np.nanmax(ev_work, axis=0)
        for c in cols:
            mask = np.isnan(ev_work[:, c])
            ev_work[mask, c] = col_max[c]
        I = np.all(ev_work[:, cols] >= ev_work[a, cols], axis=1)

    D = np.sum(I)
    N = np.sum(ex_class[I] >= t)
    LA = N / D if D > 0 else 0.0
    return LA, I


def check_rule_creation(at_least_rules: np.ndarray, ev_a: np.ndarray,
                         p_select: np.ndarray, L_a: float, t: int,
                         conf: np.ndarray) -> bool:
    """
    Check conditions C1-C4 for whether a new candidate rule should be created.
    Corresponds to check_rule_creation.m (Algorithm 1 step 8).
    """
    subset = create_subset(p_select)
    D = np.zeros(len(at_least_rules), dtype=bool)

    for i, rule in enumerate(at_least_rules):
        d_i = rule[-1]
        ev_i = rule[:-1]
        criteria_mask = ~np.isnan(ev_i)
        criteria_vec = criteria_mask.astype(int)

        # C1: P_i not subset of P
        c1 = not any(np.all(criteria_vec == s) for s in subset)
        if c1:
            D[i] = True
            continue

        # C2: P_i subset of P and exists g_j in P_i with q_j > g_j(a)
        if np.any(ev_i[criteria_mask] > ev_a[criteria_mask]):
            D[i] = True
            continue

        # C3: L_a > conf(r_i)  [MATLAB order: L_a check before d_i < t]
        if L_a > conf[i]:
            D[i] = True
            continue

        # C4: d_i < t
        if d_i < t:
            D[i] = True

    return bool(np.all(D))


def check_rule_after_creation(Q: np.ndarray, p_select: np.ndarray,
                               L_store: np.ndarray) -> np.ndarray:
    """
    Remove dominated rules from the candidate set Q for a given P.
    Corresponds to check_rule_after_creation_new.m (Algorithm 1 steps 16-18).
    Returns boolean array: True = keep rule.
    """
    cols = np.where(p_select == 1)[0]
    n = len(Q)
    C = np.zeros(n, dtype=bool)

    for r in range(n):
        keep = True
        for r_prime in range(n):
            if r_prime == r:
                continue
            q_r = Q[r, cols]
            q_rp = Q[r_prime, cols]

            # C'1: exists g_j in P with g_j(a') > g_j(a)
            if np.any(q_rp > q_r):
                continue
            # C'2: all g_j(a') <= g_j(a) and t' < t
            if Q[r_prime, -1] < Q[r, -1]:
                continue
            # C'3: all g_j(a') <= g_j(a), t' >= t, L(a) > L(a')
            if L_store[r] > L_store[r_prime]:
                continue
            # None of the conditions hold: r is dominated by r_prime
            keep = False
            break
        C[r] = keep
    return C


def convert_format_rules(at_least_rules: np.ndarray) -> np.ndarray:
    """
    Convert internal rule format to output format:
    [crit_idx, threshold, crit_idx, threshold, ..., class]
    Corresponds to convertformatrules.m
    """
    G = at_least_rules.shape[1] - 1
    out = np.zeros((len(at_least_rules), G * 2 + 1))
    for i, rule in enumerate(at_least_rules):
        criteria = np.where(~np.isnan(rule[:-1]))[0]
        for j, c in enumerate(criteria):
            out[i, 2 * c] = c + 1        # 1-based criterion index
            out[i, 2 * c + 1] = rule[c]  # threshold value
        out[i, -1] = rule[-1]            # class
    return out


def matching_atleast_opt(at_least_rules: np.ndarray, decreasing: list,
                          matrix_eval: np.ndarray):
    """
    For each rule and each unit, check if the unit matches the rule condition.
    Corresponds to MatchingATLEASTRule_opt.m
    Returns: answer (score), match (binary), match_ans
    """
    ev = matrix_eval[:, :-1].copy().astype(float)
    for d in decreasing:
        ev[:, d] = -ev[:, d]

    n_rules = len(at_least_rules)
    n_units = len(ev)
    match = np.zeros((n_units, n_rules), dtype=float)

    for i, rule in enumerate(at_least_rules):
        dec = rule[-1]
        thresholds = rule[:-1]
        # positions where threshold != 0 (active criteria)
        active = np.where(thresholds != 0)[1] if thresholds.ndim > 1 else np.where(thresholds != 0)[0]
        # extract threshold values at active columns (convert_format gives pairs)
        # rule format: [idx, val, idx, val, ..., class]
        rule_body = rule[:-1]
        crit_positions = np.where(rule_body[::2] != 0)[0]  # 0-based positions in pairs
        cols = (rule_body[::2][crit_positions] - 1).astype(int)
        vals = rule_body[1::2][crit_positions]

        if len(cols) > 0:
            matched = np.all(ev[:, cols] >= vals, axis=1)
        else:
            matched = np.ones(n_units, dtype=bool)
        match[:, i] = matched.astype(float)

    answer = match * at_least_rules[:, -1]
    return answer, match, match


def compute_support_decision(at_least_rules: np.ndarray,
                              matrix_eval: np.ndarray) -> np.ndarray:
    """
    For each rule i and unit j, check if class(j) >= class of rule i.
    Corresponds to SupportATLEASTMatrixDECISION computation in createatleastrules.m
    """
    n_units = matrix_eval.shape[0]
    n_rules = len(at_least_rules)
    G = matrix_eval.shape[1] - 1
    classes = matrix_eval[:, -1]
    result = np.zeros((n_units, n_rules))
    for i, rule in enumerate(at_least_rules):
        rule_class = rule[-1]
        result[:, i] = (classes >= rule_class).astype(float)
    return result
