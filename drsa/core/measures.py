"""
Relative support and Bayesian confirmation measures for DRSA rules.
"""

import numpy as np


def relative_support_al(examples_matrix: np.ndarray,
                         atleast_rules: np.ndarray,
                         support_match: np.ndarray) -> np.ndarray:
    """
    Compute relative support for at-least rules.
    RS_i = |E_i ∩ Cl>=t| / |Cl>=t|
    Corresponds to relative_support_al.m
    """
    classes = examples_matrix[:, -1]
    n_rules = atleast_rules.shape[0]
    rs = np.zeros(n_rules)
    for i in range(n_rules):
        t = atleast_rules[i, -1]
        cl_t = np.sum(classes >= t)
        if cl_t > 0:
            rs[i] = np.sum(support_match[:, i]) / cl_t
    return rs


def relative_support_am(examples_matrix: np.ndarray,
                         atmost_rules: np.ndarray,
                         support_match: np.ndarray) -> np.ndarray:
    """
    Compute relative support for at-most rules.
    RS_i = |E_i ∩ Cl<=t| / |Cl<=t|
    Corresponds to relative_support_am.m
    """
    classes = examples_matrix[:, -1]
    n_rules = atmost_rules.shape[0]
    rs = np.zeros(n_rules)
    for i in range(n_rules):
        t = atmost_rules[i, -1]
        cl_t = np.sum(classes <= t)
        if cl_t > 0:
            rs[i] = np.sum(support_match[:, i]) / cl_t
    return rs


def confirmation_measure(support_rule: np.ndarray,
                          support_decision: np.ndarray) -> tuple:
    """
    Compute Bayesian confirmation measures S and N for each rule.
    S(H,E) = a/(a+c) - b/(b+d)
    N(H,E) = a/(a+b) - c/(c+d)
    Corresponds to confirmation_measure.m

    Parameters
    ----------
    support_rule     : (n_units, n_rules) binary - unit matches rule condition
    support_decision : (n_units, n_rules) binary - unit satisfies rule decision

    Returns
    -------
    S, N : (n_rules,) arrays
    """
    a = np.sum((support_rule == 1) & (support_decision == 1), axis=0).astype(float)
    b = np.sum((support_rule == 0) & (support_decision == 1), axis=0).astype(float)
    c = np.sum((support_rule == 1) & (support_decision == 0), axis=0).astype(float)
    d = np.sum((support_rule == 0) & (support_decision == 0), axis=0).astype(float)

    S = np.where((a + c) > 0, a / (a + c), 0.0) - np.where((b + d) > 0, b / (b + d), 0.0)
    N = np.where((a + b) > 0, a / (a + b), 0.0) - np.where((c + d) > 0, c / (c + d), 0.0)

    return S, N
