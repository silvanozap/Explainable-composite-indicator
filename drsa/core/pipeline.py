"""
Full DRSA pipeline for explainable composite indicator construction.
Orchestrates all steps from main.m:
  Step 1: DM classifies reference units
  Step 2: DRSA on reference units → at-least and at-most rules
  Step 3-4: Step-forward greedy rule selection (non-contradictory)
  Step 5: Fix reference unit classifications
  Step 6: DRSA on all units
  Step 7: MILP for minimal rule set
"""

import numpy as np
from .rules import induce_atleast_rules, induce_atmost_rules
from .classifier import classify_units
from .step_forward import step_forward
from .milp import solve_minimal_rules
from .formatting import format_atleast_rules, format_atmost_rules


def run_pipeline(full_matrix: np.ndarray,
                 ref_indices: list,
                 increasing: list,
                 decreasing: list,
                 min_confidence: float = 1.0,
                 handle_missing: bool = False,
                 random_seed: int = 1) -> dict:
    """
    Run the full DRSA pipeline for composite indicator construction.

    Parameters
    ----------
    full_matrix : np.ndarray (n_units, n_criteria + 1)
        Full dataset. Last column = class label (NaN for non-reference units).
    ref_indices : list
        0-based indices of reference units (those classified by DM).
    increasing : list
        0-based indices of increasing criteria.
    decreasing : list
        0-based indices of decreasing criteria.
    min_confidence : float
        Minimum rule confidence (default 1.0).
    handle_missing : bool
        Use Algorithm 4 for missing values.
    random_seed : int
        Seed for random shuffling within tied rule groups.

    Returns
    -------
    dict with keys:
        step2_al_rules, step2_am_rules  : rules from Step 2 (reference units only)
        step3_al_rules, step3_am_rules  : rules selected by step-forward (Step 3-4)
        step6_al_rules, step6_am_rules  : rules from Step 6 (all units)
        step7_al_rules, step7_am_rules  : minimal rules from Step 7 (MILP)
        classification_step4            : (n_units, 2) s⁻ and s⁺ after step-forward
        classification_step7            : (n_units, 2) s⁻ and s⁺ after MILP
        milp_success                    : bool
        milp_message                    : str
        ref_indices                     : reference unit indices used
    """
    result = {}
    n_units = len(full_matrix)
    all_criteria = full_matrix[:, :-1]

    # ── Step 1: Extract reference units ───────────────────────────────────────
    ref_matrix = full_matrix[ref_indices, :]

    # ── Step 2: DRSA on reference units ───────────────────────────────────────
    al_rules, al_match, al_dec, _ = induce_atleast_rules(
        ref_matrix, increasing, decreasing, min_confidence, handle_missing)
    am_rules, am_match, am_dec, _ = induce_atmost_rules(
        ref_matrix, increasing, decreasing, min_confidence, handle_missing)

    result['step2_al_rules'] = al_rules
    result['step2_am_rules'] = am_rules

    if (al_rules is None or len(al_rules) == 0) and \
       (am_rules is None or len(am_rules) == 0):
        result['error'] = "No rules induced from reference units."
        return result

    # ── Steps 3-4: Step-forward greedy selection ───────────────────────────────
    sel_al, sel_am, pos_al, pos_am = step_forward(
        al_rules, am_rules,
        al_match, am_match,
        al_dec, am_dec,
        ref_matrix, all_criteria,
        increasing, decreasing,
        random_seed=random_seed
    )

    result['step3_al_rules'] = sel_al
    result['step3_am_rules'] = sel_am

    # Classify all units with selected rules
    matrix_all_with_class = np.hstack([
        all_criteria,
        np.full((n_units, 1), np.nan)
    ])
    s_minus, s_plus, _, _ = classify_units(
        matrix_all_with_class, sel_al, sel_am, increasing, decreasing)

    classification_step4 = np.column_stack([s_minus, s_plus])
    result['classification_step4'] = classification_step4

    # ── Step 5: Fix reference unit classifications ─────────────────────────────
    # s_minus for reference units must match DM classification
    s_minus_fixed = s_minus.copy()
    s_plus_fixed  = s_plus.copy()
    for idx in ref_indices:
        dm_class = full_matrix[idx, -1]
        if not np.isnan(dm_class):
            s_minus_fixed[idx] = dm_class
            s_plus_fixed[idx]  = dm_class

    # Build matrices for Step 6
    matrix_s_minus = np.hstack([all_criteria, s_minus_fixed.reshape(-1, 1)])
    matrix_s_plus  = np.hstack([all_criteria, s_plus_fixed.reshape(-1, 1)])

    # ── Step 6: DRSA on all units ──────────────────────────────────────────────
    al_rules2, al_match2, al_dec2, _ = induce_atleast_rules(
        matrix_s_minus, increasing, decreasing, min_confidence, handle_missing)
    am_rules2, am_match2, am_dec2, _ = induce_atmost_rules(
        matrix_s_plus, increasing, decreasing, min_confidence, handle_missing)

    result['step6_al_rules'] = al_rules2
    result['step6_am_rules'] = am_rules2
    result['matrix_s_minus'] = matrix_s_minus
    result['matrix_s_plus']  = matrix_s_plus
    result['al_match2']      = al_match2
    result['am_match2']      = am_match2

    # Classify all units with Step 6 rules
    s_minus2, s_plus2, _, _ = classify_units(
        matrix_all_with_class, al_rules2, am_rules2, increasing, decreasing)
    classification_step6 = np.column_stack([s_minus2, s_plus2])
    result['classification_step6'] = classification_step6

    # ── Step 7: MILP for minimal rule set ─────────────────────────────────────
    al_min, am_min, al_idx_min, am_idx_min, success, message = solve_minimal_rules(
        matrix_s_minus, al_match2,
        matrix_s_plus,  am_match2,
        al_rules2, am_rules2
    )

    result['milp_success']    = success
    result['milp_message']    = message
    result['step7_al_rules']  = al_min
    result['step7_am_rules']  = am_min
    result['step7_al_idx']    = al_idx_min
    result['step7_am_idx']    = am_idx_min

    if success and al_min is not None:
        s_minus7, s_plus7, _, _ = classify_units(
            matrix_all_with_class, al_min, am_min, increasing, decreasing)
        classification_step7 = np.column_stack([s_minus7, s_plus7])
        result['classification_step7'] = classification_step7
    else:
        result['classification_step7'] = classification_step6

    result['ref_indices'] = ref_indices
    return result
