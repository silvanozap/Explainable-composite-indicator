"""
Convert rule matrices to human-readable natural language strings.
Based on from_atleast_to_rules.m and from_atmost_to_rules.m
"""

import numpy as np


def _fmt_threshold(name, value, qual_mapping, n_decimals):
    """Return display string for a threshold value, using qual_mapping if available."""
    if qual_mapping and name in qual_mapping:
        _inv = {v: k for k, v in qual_mapping[name].items()}
        _rounded = round(float(value))
        if _rounded in _inv:
            return _inv[_rounded]
    return str(round(value, n_decimals))


def format_atleast_rules(rules: np.ndarray,
                          increasing: list,
                          decreasing: list,
                          criterion_names: list = None,
                          n_decimals: int = 4,
                          score_map: dict = None,
                          qual_mapping: dict = None) -> list:
    """
    Convert at-least rule matrix to natural language strings.

    Parameters
    ----------
    rules : np.ndarray
        Output of induce_atleast_rules: [crit_idx, threshold, ..., class]
    increasing : list
        0-based indices of increasing criteria.
    decreasing : list
        0-based indices of decreasing criteria.
    criterion_names : list, optional
        Names of criteria. If None, uses g1, g2, ...
    n_decimals : int
        Decimal places for threshold display.

    Returns
    -------
    list of str
    """
    result = []
    G = (rules.shape[1] - 1) // 2

    if criterion_names is None:
        criterion_names = [f'g{i+1}' for i in range(G)]

    for i, rule in enumerate(rules):
        parts = []
        # rule format: [crit_idx(1-based), threshold, crit_idx, threshold, ..., class]
        for pos in range(0, rules.shape[1] - 1, 2):
            crit_1based = int(rule[pos])
            if crit_1based == 0:
                continue
            crit_0based = crit_1based - 1
            threshold = rule[pos + 1]
            name = criterion_names[crit_0based] if crit_0based < len(criterion_names) else f'g{crit_1based}'

            if crit_0based in increasing:
                parts.append(f'{name} ≥ {_fmt_threshold(name, threshold, qual_mapping, n_decimals)}')
            elif crit_0based in decreasing:
                # threshold was negated internally; display original value
                parts.append(f'{name} ≤ {_fmt_threshold(name, -threshold, qual_mapping, n_decimals)}')
            else:
                parts.append(f'{name} ≥ {_fmt_threshold(name, threshold, qual_mapping, n_decimals)}')

        rule_class = int(rule[-1])
        class_label = score_map[rule_class] if score_map and rule_class in score_map else rule_class
        mode_word = "Score" if score_map else "Class"
        condition = ' and '.join(parts)
        text = (f'If {condition}, '
                f'then a is assigned to at least {mode_word} {class_label} ')
        result.append(text)

    return result


def format_atmost_rules(rules: np.ndarray,
                         increasing: list,
                         decreasing: list,
                         criterion_names: list = None,
                         n_decimals: int = 4,
                         score_map: dict = None,
                         qual_mapping: dict = None) -> list:
    """
    Convert at-most rule matrix to natural language strings.

    Parameters and return same as format_atleast_rules.
    """
    result = []
    G = (rules.shape[1] - 1) // 2

    if criterion_names is None:
        criterion_names = [f'g{i+1}' for i in range(G)]

    for i, rule in enumerate(rules):
        parts = []
        for pos in range(0, rules.shape[1] - 1, 2):
            crit_1based = int(rule[pos])
            if crit_1based == 0:
                continue
            crit_0based = crit_1based - 1
            threshold = rule[pos + 1]
            name = criterion_names[crit_0based] if crit_0based < len(criterion_names) else f'g{crit_1based}'

            # At-most rules: increasing criteria use <=, decreasing use >=
            # The internal negation means we display -threshold for increasing
            if crit_0based in increasing:
                parts.append(f'{name} ≤ {_fmt_threshold(name, -threshold, qual_mapping, n_decimals)}')
            elif crit_0based in decreasing:
                parts.append(f'{name} ≥ {_fmt_threshold(name, threshold, qual_mapping, n_decimals)}')
            else:
                parts.append(f'{name} ≤ {_fmt_threshold(name, -threshold, qual_mapping, n_decimals)}')

        rule_class = int(rule[-1])
        class_label = score_map[rule_class] if score_map and rule_class in score_map else rule_class
        mode_word = "Score" if score_map else "Class"
        condition = ' and '.join(parts)
        text = (f'If {condition}, '
                f'then a is assigned to at most {mode_word} {class_label} ')
        result.append(text)

    return result


def compute_relative_support(rules: np.ndarray,
                              support_match: np.ndarray,
                              support_decision: np.ndarray) -> np.ndarray:
    """
    Compute relative support for each rule: |E_i ∩ Cl>=t| / |Cl>=t|
    Equation (1) in the paper.
    """
    n_rules = rules.shape[0]
    supp = np.zeros(n_rules)
    for i in range(n_rules):
        cl_t = support_decision[:, i].sum()
        if cl_t > 0:
            supp[i] = (support_match[:, i] * support_decision[:, i]).sum() / cl_t
    return supp


def get_supporting_units(support_match: np.ndarray,
                          support_decision: np.ndarray,
                          unit_names: list = None) -> list:
    """
    For each rule, return the list of unit indices (or names) that support it.
    A unit supports rule i if it matches the condition AND belongs to Cl>=t.
    """
    n_units, n_rules = support_match.shape
    if unit_names is None:
        unit_names = [f'a{i+1}' for i in range(n_units)]

    supporting = []
    for i in range(n_rules):
        mask = (support_match[:, i] == 1) & (support_decision[:, i] == 1)
        supporting.append([unit_names[j] for j in np.where(mask)[0]])
    return supporting