"""
Classification of units using induced at-least and at-most rules.
Implements equation (4) from:
Corrente, Greco, Slowiński, Zappalà (2026)
"An explainable and interpretable composite indicator based on decision rules"
Omega 142, 103513.
"""

import numpy as np


def classify_units(example_matrix: np.ndarray,
                   atleast_rules: np.ndarray,
                   atmost_rules: np.ndarray,
                   increasing: list,
                   decreasing: list) -> tuple:
    """
    Classify units using equation (4).

    For each unit a:
      s⁻(a) = 1 if no at-least rule is matched, else max class of matched at-least rules
      s⁺(a) = p if no at-most rule is matched, else min class of matched at-most rules

    Parameters
    ----------
    example_matrix : np.ndarray
        (n_units, n_criteria + 1). Last column = class (can be NaN for new units).
    atleast_rules : np.ndarray
        Output of induce_atleast_rules.
    atmost_rules : np.ndarray
        Output of induce_atmost_rules.
    increasing : list
        0-based indices of increasing criteria.
    decreasing : list
        0-based indices of decreasing criteria.

    Returns
    -------
    s_minus : np.ndarray  (n_units,)  lower bound of class assignment
    s_plus  : np.ndarray  (n_units,)  upper bound of class assignment
    al_matched : np.ndarray (n_units, n_atleast_rules)  binary match matrix
    am_matched : np.ndarray (n_units, n_atmost_rules)   binary match matrix
    """
    ev = example_matrix[:, :-1].copy().astype(float)
    n_units = len(ev)

    # Get number of classes from rules
    all_classes = []
    if atleast_rules is not None and len(atleast_rules) > 0:
        all_classes.extend(atleast_rules[:, -1].tolist())
    if atmost_rules is not None and len(atmost_rules) > 0:
        all_classes.extend(atmost_rules[:, -1].tolist())
    p = int(max(all_classes)) if all_classes else 1

    # Negate decreasing criteria for comparison
    ev_work = ev.copy()
    for d in decreasing:
        ev_work[:, d] = -ev_work[:, d]

    # ── Match at-least rules ───────────────────────────────────────────────────
    n_al = len(atleast_rules) if atleast_rules is not None and len(atleast_rules) > 0 else 0
    al_matched = np.zeros((n_units, n_al), dtype=float)

    if n_al > 0:
        for i, rule in enumerate(atleast_rules):
            rule_body = rule[:-1]
            # Active criteria: positions where rule_body[0::2] != 0
            crit_pos = np.where(rule_body[0::2] != 0)[0]
            cols = (rule_body[0::2][crit_pos] - 1).astype(int)
            vals = rule_body[1::2][crit_pos]

            if len(cols) > 0:
                # For decreasing criteria, threshold was stored negated
                matched = np.all(ev_work[:, cols] >= vals, axis=1)
            else:
                matched = np.ones(n_units, dtype=bool)
            al_matched[:, i] = matched.astype(float)

    # ── Match at-most rules ────────────────────────────────────────────────────
    n_am = len(atmost_rules) if atmost_rules is not None and len(atmost_rules) > 0 else 0
    am_matched = np.zeros((n_units, n_am), dtype=float)

    if n_am > 0:
        # At-most rules internal representation:
        # duality negates ALL criteria, then invert_direction re-negates decreasing
        # Result: decreasing criteria have ORIGINAL values, increasing have NEGATED values
        ev_work_am = ev.copy()
        for i_col in increasing:
            ev_work_am[:, i_col] = -ev_work_am[:, i_col]
        # decreasing criteria stay as original (double negation cancelled out)

        for i, rule in enumerate(atmost_rules):
            rule_body = rule[:-1]
            crit_pos = np.where(rule_body[0::2] != 0)[0]
            cols = (rule_body[0::2][crit_pos] - 1).astype(int)
            vals = rule_body[1::2][crit_pos]

            if len(cols) > 0:
                matched = np.all(ev_work_am[:, cols] >= vals, axis=1)
            else:
                matched = np.ones(n_units, dtype=bool)
            am_matched[:, i] = matched.astype(float)

    # ── Compute s⁻ and s⁺ (equation 4) ───────────────────────────────────────
    s_minus = np.ones(n_units, dtype=float)
    s_plus  = np.full(n_units, float(p))

    if n_al > 0:
        al_classes = atleast_rules[:, -1]  # class of each at-least rule
        for a in range(n_units):
            matched_idx = np.where(al_matched[a] == 1)[0]
            if len(matched_idx) > 0:
                s_minus[a] = np.max(al_classes[matched_idx])

    if n_am > 0:
        am_classes = atmost_rules[:, -1]  # class of each at-most rule
        for a in range(n_units):
            matched_idx = np.where(am_matched[a] == 1)[0]
            if len(matched_idx) > 0:
                s_plus[a] = np.min(am_classes[matched_idx])

    return s_minus, s_plus, al_matched, am_matched


def explain_unit(unit_idx: int,
                 s_minus: np.ndarray,
                 s_plus: np.ndarray,
                 al_matched: np.ndarray,
                 am_matched: np.ndarray,
                 atleast_texts: list,
                 atmost_texts: list,
                 unit_name: str = None) -> dict:
    """
    Generate explanation for a single unit's classification.

    Returns a dict with:
      - name: unit name
      - s_minus, s_plus: class bounds
      - contradictory: bool
      - matched_atleast: list of rule texts matched
      - matched_atmost: list of rule texts matched
    """
    name = unit_name or f'a{unit_idx + 1}'
    sm = int(s_minus[unit_idx])
    sp = int(s_plus[unit_idx])

    matched_al = [atleast_texts[i] for i in range(len(atleast_texts))
                  if al_matched[unit_idx, i] == 1] if atleast_texts else []
    matched_am = [atmost_texts[i] for i in range(len(atmost_texts))
                  if am_matched[unit_idx, i] == 1] if atmost_texts else []

    contradictory = sm > sp

    if sm == sp:
        assignment = f"Class {sm}"
    elif not contradictory:
        assignment = f"Class {sm} to {sp}"
    else:
        assignment = f"CONTRADICTORY (Minimal={sm} > Maximal={sp})"

    return {
        "name": name,
        "s_minus": sm,
        "s_plus": sp,
        "assignment": assignment,
        "contradictory": contradictory,
        "matched_atleast": matched_al,
        "matched_atmost": matched_am,
    }