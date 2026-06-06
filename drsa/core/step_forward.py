"""
Greedy step-forward rule selection for non-contradictory classification.
Corresponds to: step_forward_all_4.m
"""

import numpy as np
from .classifier import classify_units
from .measures import relative_support_al, relative_support_am, confirmation_measure


def step_forward(atleast_rules: np.ndarray,
                 atmost_rules: np.ndarray,
                 support_al: np.ndarray,
                 support_am: np.ndarray,
                 support_al_dec: np.ndarray,
                 support_am_dec: np.ndarray,
                 examples_matrix: np.ndarray,
                 all_matrix: np.ndarray,
                 increasing: list,
                 decreasing: list,
                 random_seed: int = 1) -> tuple:
    """
    Greedy step-forward selection of rules ensuring non-contradictory
    classification of all units.

    Rules are sorted by (relative_support DESC, S DESC, N DESC).
    Rules with identical measures are shuffled randomly within their group.
    Each rule is tentatively added; if it causes a contradiction it is dropped.

    Corresponds to step_forward_all_4.m

    Parameters
    ----------
    atleast_rules, atmost_rules : rule matrices
    support_al, support_am      : (n_units, n_rules) match matrices
    support_al_dec, support_am_dec : (n_units, n_rules) decision matrices
    examples_matrix             : reference units only (with class column)
    all_matrix                  : all units (criteria only, no class column)
    increasing, decreasing      : 0-based criterion indices
    random_seed                 : for reproducibility within tied groups

    Returns
    -------
    sel_al : selected at-least rules
    sel_am : selected at-most rules
    pos_al : indices of selected at-least rules (0-based)
    pos_am : indices of selected at-most rules (0-based)
    """
    rng = np.random.default_rng(random_seed)

    n_al = len(atleast_rules)
    n_am = len(atmost_rules)
    p = int(np.nanmax(examples_matrix[:, -1]))

    # ── Compute sorting criteria ──────────────────────────────────────────────
    rs_al = relative_support_al(examples_matrix, atleast_rules, support_al)
    rs_am = relative_support_am(examples_matrix, atmost_rules, support_am)
    s_al, n_al_m = confirmation_measure(support_al, support_al_dec)
    s_am, n_am_m = confirmation_measure(support_am, support_am_dec)

    # Build combined array: [rs, S, N, type(1=al, 2=am), original_index]
    info_al = np.column_stack([rs_al, s_al, n_al_m,
                                np.ones(n_al),
                                np.arange(n_al)])
    info_am = np.column_stack([rs_am, s_am, n_am_m,
                                np.full(n_am, 2),
                                np.arange(n_am)])
    info_all = np.vstack([info_al, info_am])

    # ── Sort by (RS desc, S desc, N desc), shuffle within tied groups ─────────
    # Find unique (RS, S, N) combinations
    keys = info_all[:, :3]
    unique_keys = np.unique(keys, axis=0)[::-1]  # descending

    ordered = []
    for uk in unique_keys:
        mask = np.all(keys == uk, axis=1)
        group = info_all[mask]
        idx = rng.permutation(len(group))
        ordered.append(group[idx])
    ordered = np.vstack(ordered)

    # ── Step-forward loop ─────────────────────────────────────────────────────
    sel_al = np.empty((0, atleast_rules.shape[1]))
    sel_am = np.empty((0, atmost_rules.shape[1]))
    pos_al = []
    pos_am = []

    # Build matrix for classification (criteria only, add dummy class col)
    matrix_for_class = np.hstack([
        all_matrix,
        np.full((len(all_matrix), 1), np.nan)
    ])

    for row in ordered:
        rule_type = int(row[3])
        orig_idx  = int(row[4])

        if rule_type == 1:
            candidate_al = np.vstack([sel_al, atleast_rules[orig_idx]])
            candidate_am = sel_am
        else:
            candidate_al = sel_al
            candidate_am = np.vstack([sel_am, atmost_rules[orig_idx]])

        # Classify all units with candidate rule sets
        if len(candidate_al) == 0 and len(candidate_am) == 0:
            continue

        s_minus, s_plus, _, _ = classify_units(
            matrix_for_class,
            candidate_al if len(candidate_al) > 0 else None,
            candidate_am if len(candidate_am) > 0 else None,
            increasing, decreasing
        )

        # Check non-contradictory: s⁻ <= s⁺ for all units
        non_contradictory = np.all(s_minus <= s_plus)

        if non_contradictory:
            if rule_type == 1:
                sel_al = candidate_al
                pos_al.append(orig_idx)
            else:
                sel_am = candidate_am
                pos_am.append(orig_idx)

    return sel_al, sel_am, pos_al, pos_am
