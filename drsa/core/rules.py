"""
Algorithm 1: At-Least rule induction (DRSA)
Algorithm 2: At-Most rule induction (DRSA)
Algorithm 4: Rule induction with missing values

Based on:
Corrente, Greco, Slowiński, Zappalà (2026)
"An explainable and interpretable composite indicator based on decision rules"
Omega 142, 103513.
"""

import numpy as np
from itertools import combinations
from .utils import (
    invert_direction, from_binary_to_number, rule_add,
    lower_approximation, check_rule_creation, check_rule_after_creation,
    convert_format_rules, matching_atleast_opt, compute_support_decision,
    reverse_class
)


def induce_atleast_rules(example_matrix: np.ndarray,
                          increasing: list,
                          decreasing: list,
                          min_confidence: float = 1.0,
                          handle_missing: bool = False):
    """
    Algorithm 1 (and Algorithm 4 when handle_missing=True).

    Induces all minimal "at-least" decision rules from the classification data.

    Parameters
    ----------
    example_matrix : np.ndarray
        Matrix of shape (n_units, n_criteria + 1).
        Last column contains class labels (integers, 1-based).
        NaN values are allowed when handle_missing=True.
    increasing : list
        0-based indices of gain-type (increasing) criteria.
    decreasing : list
        0-based indices of cost-type (decreasing) criteria.
    min_confidence : float
        Minimum rule confidence c (default 1.0 = exact rules only).
    handle_missing : bool
        If True, uses Algorithm 4 logic for missing values.

    Returns
    -------
    rules : np.ndarray
        Rule matrix in output format [crit_idx, threshold, ..., class].
    support_match : np.ndarray
        Binary (n_units x n_rules): unit i matches condition of rule j.
    support_decision : np.ndarray
        Binary (n_units x n_rules): class(i) >= class of rule j.
    bases : list of (unit_idx, class)
        Base units for each rule.
    """
    d_min = 2           # minimum class for at-least rules
    c = min_confidence

    ex_class = example_matrix[:, -1].copy()
    ev = invert_direction(example_matrix[:, :-1], increasing, decreasing)
    n, G = ev.shape

    at_least_rules = None   # will be np.ndarray once first rule is added
    conf = []
    bases = []

    # Generate all non-empty subsets of G criteria, sorted by cardinality
    bin_rows = np.array([[int(b) for b in f'{i:0{G}b}'] for i in range(1, 2**G)])
    P_all = from_binary_to_number(bin_rows, G)

    for p_select in P_all:
        Q = []           # candidate rules for this P
        L_store = []
        A = []           # base (unit, class) pairs
        I_store = []     # dominance indicator vectors

        for a in range(n):
            # Algorithm 4 step 3bis: skip if unit has missing in P
            if handle_missing:
                if np.any(np.isnan(ev[a, p_select == 1])):
                    continue
            else:
                if np.any(np.isnan(ev[a, p_select == 1])):
                    continue

            if ex_class[a] < d_min:
                continue

            ev_a = ev[a].copy()

            for t in range(int(d_min), int(np.nanmax(ex_class)) + 1):
                L, II = lower_approximation(a, t, ev, ex_class, p_select,
                                            missing=handle_missing)
                if L < c:
                    continue

                if at_least_rules is None or len(at_least_rules) == 0:
                    r_tilde = rule_add(ev_a, p_select, t)
                    Q.append(r_tilde)
                    L_store.append(L)
                    A.append((a, t))
                    I_store.append(II)
                else:
                    should_add = check_rule_creation(
                        at_least_rules, ev_a, p_select, L, t, np.array(conf)
                    )
                    if should_add:
                        r_tilde = rule_add(ev_a, p_select, t)
                        Q.append(r_tilde)
                        L_store.append(L)
                        A.append((a, t))
                        I_store.append(II)

        if not Q:
            continue

        Q = np.array(Q)
        L_store = np.array(L_store)
        A_arr = np.array(A)

        # Remove duplicate rules (same values on P columns and class)
        p_cols = np.where(p_select == 1)[0].tolist()
        key_cols = p_cols + [Q.shape[1] - 1]
        _, idx = np.unique(Q[:, key_cols], axis=0, return_index=True)
        idx = np.sort(idx)   # preserve stable order
        Q = Q[idx]
        L_store = L_store[idx]
        A_arr = A_arr[idx]

        # Steps 16-18: keep only non-dominated rules
        keep = check_rule_after_creation(Q, p_select, L_store)

        if np.any(keep):
            new_rules = Q[keep]
            new_conf = L_store[keep]
            new_bases = A_arr[keep].tolist()

            if at_least_rules is None or len(at_least_rules) == 0:
                at_least_rules = new_rules
            else:
                at_least_rules = np.vstack([at_least_rules, new_rules])

            conf.extend(new_conf.tolist())
            bases.extend(new_bases)

    if at_least_rules is None or len(at_least_rules) == 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), []

    # Convert to output format
    rules_out = convert_format_rules(at_least_rules)

    # Compute support matrices
    _, support_match, _ = matching_atleast_opt(rules_out, decreasing, example_matrix)
    support_decision = compute_support_decision(rules_out, example_matrix)

    return rules_out, support_match, support_decision, bases


def induce_atmost_rules(example_matrix: np.ndarray,
                         increasing: list,
                         decreasing: list,
                         min_confidence: float = 1.0,
                         handle_missing: bool = False):
    """
    Algorithm 2: Induces all minimal "at-most" decision rules.

    Implemented by duality: reverses class order and negates evaluations,
    then calls induce_atleast_rules, then reverses classes back.
    Corresponds to createatmostrules.m

    Parameters
    ----------
    example_matrix : np.ndarray
        Matrix (n_units, n_criteria + 1). Last column = class labels.
    increasing : list
        0-based indices of gain-type criteria.
    decreasing : list
        0-based indices of cost-type criteria.
    min_confidence : float
        Minimum confidence threshold.
    handle_missing : bool
        If True, uses Algorithm 4 missing-value logic.

    Returns
    -------
    Same structure as induce_atleast_rules.
    """
    em = example_matrix.copy().astype(float)

    # Reverse class labels
    em[:, -1] = reverse_class(em[:, -1])

    # Negate all evaluations (duality trick)
    em[:, :-1] = -em[:, :-1]

    rules_out, support_match, support_decision, bases = induce_atleast_rules(
        em, increasing, decreasing, min_confidence, handle_missing
    )

    if len(rules_out) == 0:
        return rules_out, support_match, support_decision, bases

    # Reverse class labels back in the rules
    classes = rules_out[:, -1].copy()
    classes_aug = np.append(classes, 1.0)
    classes_reversed = reverse_class(classes_aug)
    rules_out[:, -1] = classes_reversed[:-1]

    return rules_out, support_match, support_decision, bases
