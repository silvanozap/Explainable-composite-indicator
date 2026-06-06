"""
DRSA Composite Indicator Tool - Full Pipeline
Corrente, Greco, Slowiński, Zappalà (Omega, 2026)
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os

_HERE        = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_HERE)
_ROOT        = os.path.dirname(_PACKAGE_DIR)
sys.path.insert(0, _ROOT)

from drsa import (
    induce_atleast_rules, induce_atmost_rules,
    format_atleast_rules, format_atmost_rules,
    compute_relative_support, get_supporting_units,
    classify_units, explain_unit, run_pipeline,
)

st.set_page_config(page_title="DRSA", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;letter-spacing:-0.03em;}
.rule-box{background:#f8f9fa;border-left:4px solid #1a1a2e;padding:12px 16px;margin:6px 0;border-radius:0 6px 6px 0;font-family:'IBM Plex Mono',monospace;font-size:0.82rem;line-height:1.6;}
.rule-box.atleast{border-left-color:#2563eb;}
.rule-box.atmost{border-left-color:#dc2626;}
.metric-card{background:#1a1a2e;color:white;border-radius:8px;padding:14px 18px;text-align:center;}
.metric-card .value{font-size:2rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.metric-card .label{font-size:0.75rem;opacity:0.7;margin-top:2px;}
.info-banner{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 18px;font-size:0.88rem;color:#1e3a5f;margin-bottom:18px;}
.tag{display:inline-block;background:#dbeafe;color:#1e40af;border-radius:4px;padding:1px 7px;font-size:0.73rem;font-family:'IBM Plex Mono',monospace;margin-right:3px;}
.class-box{border-radius:8px;padding:12px 16px;margin:6px 0;font-family:'IBM Plex Mono',monospace;font-size:0.88rem;}
.class-ok{background:#f0fdf4;border-left:4px solid #16a34a;}
.class-range{background:#fffbeb;border-left:4px solid #d97706;}
.class-err{background:#fef2f2;border-left:4px solid #dc2626;}
.step-header{background:#1e3a5f;color:white;border-radius:6px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;margin:12px 0 6px 0;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ DRSA · Composite Indicator Tool")
st.markdown("""<div class="info-banner">
Dominance-based Rough Set Approach for explainable composite indicators.<br>
Full pipeline: <b>rule induction → non-contradictory selection → minimal rule set (MILP) → classification</b><br>
Corrente, Greco, Słowiński, Zappalà — <i>Omega 142</i> (2026), 103513.
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload CSV or TXT", type=["csv","txt"],
        help="Last column = class label (NaN for non-reference units). Optional first column = names.")
    sep = st.selectbox("Separator", [",",";","\\t"," "], index=0)
    sep_actual = "\t" if sep=="\\t" else sep
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    min_conf = st.slider("Min confidence (c)", 0.0, 1.0, 1.0, 0.05)
    handle_missing = st.checkbox("Missing values (Algorithm 4)", False)
    random_seed = st.number_input("Random seed", value=1, min_value=0, step=1)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#9ca3af'>Corrente et al. (2026)<br>Omega 142, 103513</div>", unsafe_allow_html=True)

# ── Welcome ────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("#### Getting started")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
**File format**
- CSV or TXT
- Optional first column: unit names
- Criteria columns (numeric)
- Last column: class label
  - Integer for reference units
  - Empty/NaN for non-reference units
""")
    with c2:
        st.markdown("""
**Example**
```
Name,g1,g2,g3,class
A1,4,3,2,1
A2,5,4,3,2
A3,7,6,5,
A4,6,5,4,
```
A3, A4 are non-reference (no class).
""")
    with c3:
        st.markdown("""
**Full pipeline**
1. Induce rules from reference units
2. Greedy non-contradictory selection
3. Fix reference classifications
4. DRSA on all units
5. MILP → minimal rule set
6. Classify new units
""")
    sample = pd.DataFrame({
        'Name':['A1','A2','A3','A4','A5','A6','A7','A8'],
        'g1':[4,5,7,3,6,8,5,4],'g2':[3,4,6,2,5,7,4,3],
        'g3':[2,3,5,1,4,6,3,2],
        'class':[1,2,3,1,'','',2,'']
    })
    st.download_button("⬇ Download sample CSV", sample.to_csv(index=False),
                       file_name="drsa_sample.csv", mime="text/csv")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded, sep=sep_actual, engine="python")
except Exception as e:
    st.error(f"Could not read file: {e}"); st.stop()

unit_names = None
df = df_raw.copy()
first_col = df.columns[0]
if pd.to_numeric(df[first_col], errors='coerce').isna().sum() > len(df)*0.5:
    unit_names = df[first_col].astype(str).tolist()
    df = df.drop(columns=[first_col])

df = df.apply(pd.to_numeric, errors='coerce')
if unit_names is None:
    unit_names = [f'a{i+1}' for i in range(len(df))]

n_units, n_cols = df.shape
n_criteria = n_cols - 1
crit_names = list(df.columns)[:-1]
matrix = df.values.astype(float)

# Identify reference units (those with non-NaN class)
ref_mask = ~np.isnan(matrix[:, -1])
ref_indices = np.where(ref_mask)[0].tolist()
n_ref = len(ref_indices)
n_nonref = n_units - n_ref

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Setup",
    "⚙️ Run Pipeline",
    "🔍 Classification",
    "🆕 New Units"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### 📊 Dataset")
    st.dataframe(df_raw, use_container_width=True, height=250)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total units", n_units)
    c2.metric("Reference units", n_ref)
    c3.metric("Non-reference units", n_nonref)

    st.markdown("#### 🎯 Reference units")
    st.markdown("Units with a class value in the file are pre-selected. You can add or remove units.")
    selected_ref_names = st.multiselect(
        "Select reference units",
        options=unit_names,
        default=[unit_names[i] for i in ref_indices],
        help="Reference units are those whose class is known and used to induce rules."
    )
    if len(selected_ref_names) == 0:
        st.error("Please select at least one reference unit.")
        st.stop()
    ref_indices = [unit_names.index(n) for n in selected_ref_names]
    n_ref    = len(ref_indices)
    n_nonref = n_units - n_ref
    c2.metric("Reference units", n_ref)
    c3.metric("Non-reference units", n_nonref)

    st.markdown("#### 🔼 Criteria preference direction")
    dir_cols = st.columns(min(n_criteria, 6))
    directions = {}
    for i, name in enumerate(crit_names):
        with dir_cols[i % len(dir_cols)]:
            directions[name] = st.selectbox(name, ["↑ Increasing","↓ Decreasing"], key=f"dir_{i}")

    increasing_0 = [i for i,n in enumerate(crit_names) if "Increasing" in directions[n]]
    decreasing_0 = [i for i,n in enumerate(crit_names) if "Decreasing" in directions[n]]

    st.session_state['increasing_0'] = increasing_0
    st.session_state['decreasing_0'] = decreasing_0
    st.session_state['matrix']       = matrix
    st.session_state['unit_names']   = unit_names
    st.session_state['crit_names']   = crit_names
    st.session_state['ref_indices']  = ref_indices

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'increasing_0' not in st.session_state:
        st.info("Set up criteria directions in the **Data & Setup** tab first.")
    else:
        increasing_0 = st.session_state['increasing_0']
        decreasing_0 = st.session_state['decreasing_0']
        matrix       = st.session_state['matrix']
        unit_names   = st.session_state['unit_names']
        crit_names   = st.session_state['crit_names']
        ref_indices  = st.session_state['ref_indices']

        mode = st.radio("Mode", ["🔬 Rule induction only (Steps 1-2)", 
                                  "🚀 Full pipeline (Steps 1-7)"],
                        horizontal=True)

        if st.button("▶ Run", type="primary", use_container_width=True):
            if "only" in mode:
                # ── Rule induction only ────────────────────────────────────────
                ref_matrix = matrix[ref_indices, :]
                with st.spinner("Inducing rules from reference units…"):
                    al_rules, al_match, al_dec, _ = induce_atleast_rules(
                        ref_matrix, increasing_0, decreasing_0, min_conf, handle_missing)
                    am_rules, am_match, am_dec, _ = induce_atmost_rules(
                        ref_matrix, increasing_0, decreasing_0, min_conf, handle_missing)

                n_al = len(al_rules) if al_rules is not None and len(al_rules)>0 else 0
                n_am = len(am_rules) if am_rules is not None and len(am_rules)>0 else 0

                st.session_state['al_rules'] = al_rules
                st.session_state['am_rules'] = am_rules
                st.session_state['al_match'] = al_match
                st.session_state['am_match'] = am_match
                st.session_state['al_dec']   = al_dec
                st.session_state['am_dec']   = am_dec
                st.session_state['pipeline_result'] = None

                mc1,mc2,mc3 = st.columns(3)
                for col,val,lbl in zip([mc1,mc2,mc3],[n_al,n_am,n_al+n_am],
                                        ["At-least (R≥)","At-most (R≤)","Total"]):
                    col.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                                 f'<div class="label">{lbl}</div></div>', unsafe_allow_html=True)
                st.markdown("")

                ref_unit_names = [unit_names[i] for i in ref_indices]

                if n_al > 0:
                    st.markdown("### R≥ · At-Least Rules")
                    al_texts = format_atleast_rules(al_rules, increasing_0, decreasing_0, crit_names)
                    al_supp  = compute_relative_support(al_rules, al_match, al_dec)
                    al_units = get_supporting_units(al_match, al_dec, ref_unit_names)
                    st.session_state['al_texts'] = al_texts
                    st.session_state['am_texts'] = []
                    for txt,supp,units in zip(al_texts,al_supp,al_units):
                        tags = "".join(f'<span class="tag">{u}</span>' for u in units)
                        st.markdown(f'<div class="rule-box atleast">{txt}'
                                    f'<br><span style="color:#6b7280;font-size:0.75rem;">Support: {supp:.3f} &nbsp;·&nbsp; </span>'
                                    f'{tags}</div>', unsafe_allow_html=True)
                if n_am > 0:
                    st.markdown("### R≤ · At-Most Rules")
                    am_texts = format_atmost_rules(am_rules, increasing_0, decreasing_0, crit_names)
                    am_supp  = compute_relative_support(am_rules, am_match, am_dec)
                    am_units = get_supporting_units(am_match, am_dec, ref_unit_names)
                    st.session_state['am_texts'] = am_texts
                    for txt,supp,units in zip(am_texts,am_supp,am_units):
                        tags = "".join(f'<span class="tag">{u}</span>' for u in units)
                        st.markdown(f'<div class="rule-box atmost">{txt}'
                                    f'<br><span style="color:#6b7280;font-size:0.75rem;">Support: {supp:.3f} &nbsp;·&nbsp; </span>'
                                    f'{tags}</div>', unsafe_allow_html=True)

            else:
                # ── Full pipeline ──────────────────────────────────────────────
                progress = st.progress(0)
                status = st.empty()

                status.info("⏳ Step 2: Inducing rules from reference units…")
                progress.progress(10)

                from drsa.core.rules import induce_atleast_rules as _ial, induce_atmost_rules as _iam
                from drsa.core.step_forward import step_forward as _sf
                from drsa.core.milp import solve_minimal_rules as _smr
                from drsa.core.classifier import classify_units as _cu
                import numpy as _np

                ref_matrix = matrix[ref_indices, :]
                al_r, al_m, al_d, _ = _ial(ref_matrix, increasing_0, decreasing_0, min_conf, handle_missing)
                am_r, am_m, am_d, _ = _iam(ref_matrix, increasing_0, decreasing_0, min_conf, handle_missing)
                n_al2 = len(al_r) if al_r is not None and len(al_r)>0 else 0
                n_am2 = len(am_r) if am_r is not None and len(am_r)>0 else 0
                status.success(f"✅ Step 2: {n_al2} at-least, {n_am2} at-most rules induced from reference units")
                progress.progress(25)

                status.info("⏳ Step 3-4: Greedy non-contradictory rule selection…")
                all_criteria = matrix[:, :-1]
                sel_al, sel_am, _, _ = _sf(al_r, am_r, al_m, am_m, al_d, am_d,
                                           ref_matrix, all_criteria, increasing_0, decreasing_0,
                                           random_seed=int(random_seed))
                status.success(f"✅ Step 3-4: {len(sel_al)} at-least, {len(sel_am)} at-most rules selected")
                progress.progress(45)

                status.info("⏳ Step 5: Fixing reference unit classifications…")
                n_units_p = len(matrix)
                mat_nc = _np.hstack([all_criteria, _np.full((n_units_p,1), _np.nan)])
                s_minus, s_plus, _, _ = _cu(mat_nc, sel_al, sel_am, increasing_0, decreasing_0)
                s_minus_f = s_minus.copy(); s_plus_f = s_plus.copy()
                for idx in ref_indices:
                    dm_c = matrix[idx, -1]
                    if not _np.isnan(dm_c):
                        s_minus_f[idx] = dm_c; s_plus_f[idx] = dm_c
                mat_sm = _np.hstack([all_criteria, s_minus_f.reshape(-1,1)])
                mat_sp = _np.hstack([all_criteria, s_plus_f.reshape(-1,1)])
                status.success("✅ Step 5: Reference unit classifications fixed")
                progress.progress(55)

                status.info("⏳ Step 6: Inducing rules from all units…")
                al_r2, al_m2, al_d2, _ = _ial(mat_sm, increasing_0, decreasing_0, min_conf, handle_missing)
                am_r2, am_m2, am_d2, _ = _iam(mat_sp, increasing_0, decreasing_0, min_conf, handle_missing)
                n_al6 = len(al_r2) if al_r2 is not None and len(al_r2)>0 else 0
                n_am6 = len(am_r2) if am_r2 is not None and len(am_r2)>0 else 0
                status.success(f"✅ Step 6: {n_al6} at-least, {n_am6} at-most rules induced from all units")
                progress.progress(75)

                status.info("⏳ Step 7: MILP — finding minimal rule set…")
                al_min, am_min, al_idx_min, am_idx_min, milp_ok, milp_msg = _smr(
                    mat_sm, al_m2, mat_sp, am_m2, al_r2, am_r2)
                if milp_ok:
                    status.success(f"✅ Step 7: Minimal set — {len(al_min)} at-least, {len(am_min)} at-most rules")
                else:
                    status.error(f"⚠️ Step 7: {milp_msg}")
                progress.progress(90)

                status.info("⏳ Finalizing classification…")
                res = {
                    'step2_al_rules': al_r, 'step2_am_rules': am_r,
                    'step3_al_rules': sel_al, 'step3_am_rules': sel_am,
                    'step6_al_rules': al_r2, 'step6_am_rules': am_r2,
                    'step7_al_rules': al_min, 'step7_am_rules': am_min,
                    'step7_al_idx': al_idx_min, 'step7_am_idx': am_idx_min,
                    'milp_success': milp_ok, 'milp_message': milp_msg,
                    'matrix_s_minus': mat_sm, 'matrix_s_plus': mat_sp,
                    'al_match2': al_m2, 'am_match2': am_m2,
                    'ref_indices': ref_indices,
                }
                if milp_ok and al_min is not None:
                    sm7, sp7, _, _ = _cu(mat_nc, al_min, am_min, increasing_0, decreasing_0)
                    res['classification_step7'] = _np.column_stack([sm7, sp7])
                else:
                    sm6, sp6, _, _ = _cu(mat_nc, al_r2, am_r2, increasing_0, decreasing_0)
                    res['classification_step7'] = _np.column_stack([sm6, sp6])
                progress.progress(100)
                status.success("🎉 Pipeline complete!")

                # ── Show minimal rules ────────────────────────────────────────
                al_final = al_min if (milp_ok and al_min is not None and len(al_min)>0) else al_r2
                am_final = am_min if (milp_ok and am_min is not None and len(am_min)>0) else am_r2

                steps_data = [
                    ("Step 2 – Reference units", len(al_r) if al_r is not None else 0, len(am_r) if am_r is not None else 0),
                    ("Step 3 – Greedy selection", len(sel_al), len(sel_am)),
                    ("Step 6 – All units", n_al6, n_am6),
                    ("Step 7 – Minimal (MILP)", len(al_final), len(am_final)),
                ]
                import pandas as _pd
                df_steps = _pd.DataFrame(steps_data, columns=["Step","At-least","At-most"])
                df_steps["Total"] = df_steps["At-least"] + df_steps["At-most"]
                st.markdown("#### Pipeline summary")
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

                # ── Maximal rules (Step 6) in expander ───────────────────
                with st.expander(f"📂 Maximal rules — Step 6 ({n_al6} at-least, {n_am6} at-most)"):
                    if al_r2 is not None and len(al_r2) > 0:
                        st.markdown("**R≥ · At-Least Rules (maximal)**")
                        al_texts_max = format_atleast_rules(al_r2, increasing_0, decreasing_0, crit_names)
                        st.session_state['al_texts_max'] = al_texts_max
                        for txt in al_texts_max:
                            st.markdown(f'<div class="rule-box atleast">{txt}</div>', unsafe_allow_html=True)
                    if am_r2 is not None and len(am_r2) > 0:
                        st.markdown("**R≤ · At-Most Rules (maximal)**")
                        am_texts_max = format_atmost_rules(am_r2, increasing_0, decreasing_0, crit_names)
                        st.session_state['am_texts_max'] = am_texts_max
                        for txt in am_texts_max:
                            st.markdown(f'<div class="rule-box atmost">{txt}</div>', unsafe_allow_html=True)

                # ── Minimal rules (Step 7) ────────────────────────────────────
                if len(al_final) > 0:
                    st.markdown("### R≥ · Minimal At-Least Rules — Step 7")
                    al_texts = format_atleast_rules(al_final, increasing_0, decreasing_0, crit_names)
                    st.session_state['al_texts'] = al_texts
                    for txt in al_texts:
                        st.markdown(f'<div class="rule-box atleast">{txt}</div>', unsafe_allow_html=True)
                if len(am_final) > 0:
                    st.markdown("### R≤ · Minimal At-Most Rules — Step 7")
                    am_texts = format_atmost_rules(am_final, increasing_0, decreasing_0, crit_names)
                    st.session_state['am_texts'] = am_texts
                    for txt in am_texts:
                        st.markdown(f'<div class="rule-box atmost">{txt}</div>', unsafe_allow_html=True)

                # ── Download rules ────────────────────────────────────────
                st.markdown("### 💾 Export rules")
                import pandas as _pd2
                rows_dl = []
                if len(al_final) > 0:
                    for txt in al_texts:
                        rows_dl.append({"type":"at-least (minimal)","rule":txt})
                if len(am_final) > 0:
                    for txt in am_texts:
                        rows_dl.append({"type":"at-most (minimal)","rule":txt})
                if st.session_state.get('al_texts_max'):
                    for txt in st.session_state['al_texts_max']:
                        rows_dl.append({"type":"at-least (maximal)","rule":txt})
                if st.session_state.get('am_texts_max'):
                    for txt in st.session_state['am_texts_max']:
                        rows_dl.append({"type":"at-most (maximal)","rule":txt})
                df_rules_dl = _pd2.DataFrame(rows_dl)
                st.download_button("⬇ Download all rules as CSV",
                                   df_rules_dl.to_csv(index=False),
                                   file_name="drsa_rules_all.csv",
                                   mime="text/csv",
                                   use_container_width=True)

                st.session_state['pipeline_result'] = res
                al7 = res.get('step7_al_rules')
                am7 = res.get('step7_am_rules')
                al6 = res.get('step6_al_rules')
                am6 = res.get('step6_am_rules')
                st.session_state['al_rules'] = al7 if (al7 is not None and len(al7) > 0) else al6
                st.session_state['am_rules'] = am7 if (am7 is not None and len(am7) > 0) else am6
                st.session_state['al_texts'] = []
                st.session_state['am_texts'] = []

                if 'error' in res:
                    st.error(res['error']); st.stop()

                # Summary
                def nrules(r): return len(r) if r is not None and len(r)>0 else 0
                steps_data = [
                    ("Step 2 – Reference units", nrules(res['step2_al_rules']), nrules(res['step2_am_rules'])),
                    ("Step 3 – Greedy selection", nrules(res['step3_al_rules']), nrules(res['step3_am_rules'])),
                    ("Step 6 – All units", nrules(res['step6_al_rules']), nrules(res['step6_am_rules'])),
                    ("Step 7 – Minimal (MILP)", nrules(res['step7_al_rules']), nrules(res['step7_am_rules'])),
                ]
                df_steps = pd.DataFrame(steps_data, columns=["Step","At-least rules","At-most rules"])
                df_steps["Total"] = df_steps["At-least rules"] + df_steps["At-most rules"]
                st.markdown("#### Pipeline summary")
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

                milp_color = "success" if res['milp_success'] else "error"
                if res['milp_success']:
                    st.success(f"MILP: {res['milp_message']}")
                else:
                    st.error(f"MILP: {res['milp_message']}")

                # Show minimal rules
                al_min = res['step7_al_rules']
                am_min = res['step7_am_rules']
                if al_min is not None and len(al_min) > 0:
                    st.markdown("### R≥ · Minimal At-Least Rules")
                    al_texts = format_atleast_rules(al_min, increasing_0, decreasing_0, crit_names)
                    st.session_state['al_texts'] = al_texts
                    for txt in al_texts:
                        st.markdown(f'<div class="rule-box atleast">{txt}</div>', unsafe_allow_html=True)
                if am_min is not None and len(am_min) > 0:
                    st.markdown("### R≤ · Minimal At-Most Rules")
                    am_texts = format_atmost_rules(am_min, increasing_0, decreasing_0, crit_names)
                    st.session_state['am_texts'] = am_texts
                    for txt in am_texts:
                        st.markdown(f'<div class="rule-box atmost">{txt}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'al_rules' not in st.session_state:
        st.info("Run the pipeline first in the **Run Pipeline** tab.")
    else:
        al_rules    = st.session_state['al_rules']
        am_rules    = st.session_state['am_rules']
        al_texts    = st.session_state.get('al_texts', [])
        am_texts    = st.session_state.get('am_texts', [])
        unit_names  = st.session_state['unit_names']
        matrix      = st.session_state['matrix']
        increasing_0 = st.session_state['increasing_0']
        decreasing_0 = st.session_state['decreasing_0']
        res         = st.session_state.get('pipeline_result')

        st.markdown("#### Classification of all units — equation (4)")

        # Use pipeline classification if available, else compute fresh
        if res is not None and 'classification_step7' in res:
            cl = res['classification_step7']
            s_minus = cl[:, 0]
            s_plus  = cl[:, 1]
            _, _, al_m, am_m = classify_units(
                np.hstack([matrix[:,:-1], np.full((len(matrix),1), np.nan)]),
                al_rules, am_rules, increasing_0, decreasing_0)
        else:
            matrix_nc = np.hstack([matrix[:,:-1], np.full((len(matrix),1), np.nan)])
            s_minus, s_plus, al_m, am_m = classify_units(
                matrix_nc, al_rules, am_rules, increasing_0, decreasing_0)

        rows = []
        for i,name in enumerate(unit_names):
            exp = explain_unit(i, s_minus, s_plus, al_m, am_m, al_texts, am_texts, name)
            rows.append({"Unit":name,"s⁻":int(exp['s_minus']),"s⁺":int(exp['s_plus']),
                         "Assignment":exp['assignment'],
                         "Status":"⚠️ Contradictory" if exp['contradictory'] else "✅ OK"})

        df_class = pd.DataFrame(rows)
        st.dataframe(df_class, use_container_width=True, height=380)

        n_ok    = df_class['Status'].str.contains('OK').sum()
        n_cont  = df_class['Status'].str.contains('Contr').sum()
        ca,cb = st.columns(2)
        ca.metric("Non-contradictory", n_ok)
        cb.metric("Contradictory", n_cont)

        st.markdown("#### Unit-by-unit explanation")
        sel = st.selectbox("Select unit", unit_names)
        idx = unit_names.index(sel)
        exp = explain_unit(idx, s_minus, s_plus, al_m, am_m, al_texts, am_texts, sel)
        css = "class-ok" if not exp['contradictory'] and exp['s_minus']==exp['s_plus'] \
              else "class-err" if exp['contradictory'] else "class-range"
        st.markdown(f'<div class="class-box {css}"><b>{sel}</b> → <b>{exp["assignment"]}</b></div>',
                    unsafe_allow_html=True)
        if exp['matched_atleast']:
            st.markdown("**Satisfied at-least rules:**")
            for r in exp['matched_atleast']:
                st.markdown(f'<div class="rule-box atleast">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown("*No at-least rule satisfied → s⁻ = 1*")
        if exp['matched_atmost']:
            st.markdown("**Satisfied at-most rules:**")
            for r in exp['matched_atmost']:
                st.markdown(f'<div class="rule-box atmost">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown("*No at-most rule satisfied → s⁺ = p*")

        st.download_button("⬇ Download classification CSV", df_class.to_csv(index=False),
                           file_name="drsa_classification.csv", mime="text/csv",
                           use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if 'pipeline_result' not in st.session_state or st.session_state['pipeline_result'] is None:
        st.info("Run the **Full pipeline** first in the Run Pipeline tab to enable new unit classification.")
    else:
        st.markdown("#### 🆕 Classify new units")
        st.markdown(
            "Upload new units (without class column). "
            "The tool applies MILP (6), (7) and (8) from the paper to handle contradictions."
        )

        new_file = st.file_uploader("Upload new units (CSV or TXT)", type=["csv","txt"], key="new_units")
        sep2 = st.selectbox("Separator", [",",";","\\t"," "], key="sep2")
        sep2_actual = "\t" if sep2=="\\t" else sep2

        if new_file is not None:
            try:
                df_new_raw = pd.read_csv(new_file, sep=sep2_actual, engine="python")
            except Exception as e:
                st.error(f"Could not read file: {e}"); st.stop()

            new_names = None
            df_new = df_new_raw.copy()
            fc = df_new.columns[0]
            if pd.to_numeric(df_new[fc], errors='coerce').isna().sum() > len(df_new)*0.5:
                new_names = df_new[fc].astype(str).tolist()
                df_new = df_new.drop(columns=[fc])
            df_new = df_new.apply(pd.to_numeric, errors='coerce')
            if new_names is None:
                new_names = [f'x{i+1}' for i in range(len(df_new))]

            st.dataframe(df_new_raw, use_container_width=True, height=180)

            new_matrix = df_new.values.astype(float)
            res        = st.session_state['pipeline_result']
            increasing_0 = st.session_state['increasing_0']
            decreasing_0 = st.session_state['decreasing_0']
            al_texts   = st.session_state.get('al_texts', [])
            am_texts   = st.session_state.get('am_texts', [])

            if st.button("▶ Classify new units", type="primary", use_container_width=True):
                from drsa import classify_new_units

                prog_new = st.progress(0)
                stat_new = st.empty()

                stat_new.info("⏳ Step 1: Classifying new units with maximal rules (eq. 4)…")
                prog_new.progress(20)

                new_res = classify_new_units(
                    new_matrix,
                    res['matrix_s_minus'],
                    res['matrix_s_plus'],
                    res['step6_al_rules'],
                    res['al_match2'],
                    res['step6_am_rules'],
                    res['am_match2'],
                    increasing_0,
                    decreasing_0
                )

                if 'error' in new_res:
                    st.error(new_res['error']); st.stop()

                n_contra = new_res['n_contradictions']
                prog_new.progress(50)

                if n_contra == 0:
                    stat_new.success(f"✅ No contradictions found — applying MILP (8) for minimal rules")
                else:
                    stat_new.warning(f"⚠️ {n_contra} contradictions found — running MILP (6) and (7)…")

                prog_new.progress(80)

                if new_res.get('milp_success'):
                    stat_new.success(f"✅ Classification complete — {new_res.get('milp_message','')}")
                else:
                    stat_new.error(f"⚠️ {new_res.get('milp_message','')}")
                prog_new.progress(100)

                # ── Results table ──────────────────────────────────────────
                s_minus_f = new_res['final_s_minus']
                s_plus_f  = new_res['final_s_plus']

                rows = []
                for i, name in enumerate(new_names):
                    sm = int(s_minus_f[i])
                    sp = int(s_plus_f[i])
                    contra = sm > sp
                    if sm == sp:
                        assign = f"Class {sm}"
                    elif not contra:
                        assign = f"Class {sm} to {sp}"
                    else:
                        assign = f"CONTRADICTORY (s⁻={sm} > s⁺={sp})"
                    rows.append({"Unit":name,"s⁻":sm,"s⁺":sp,
                                 "Assignment":assign,
                                 "Status":"⚠️ Contradictory" if contra else "✅ OK"})

                df_nc = pd.DataFrame(rows)
                st.markdown("#### Classification results")
                st.dataframe(df_nc, use_container_width=True)

                # ── Unit explanation ───────────────────────────────────────
                al8 = new_res.get('step8_al_rules')
                am8 = new_res.get('step8_am_rules')
                al7 = new_res.get('step7_al_rules')
                am7 = new_res.get('step7_am_rules')
                al_final = al8 if (al8 is not None and len(al8) > 0) else al7
                am_final = am8 if (am8 is not None and len(am8) > 0) else am7

                if al_final is not None and len(al_final) > 0:
                    st.markdown("#### Minimal rules for A ∪ A_new")
                    al_texts_new = format_atleast_rules(al_final, increasing_0, decreasing_0,
                                                         st.session_state['crit_names'])
                    am_texts_new = format_atmost_rules(am_final, increasing_0, decreasing_0,
                                                        st.session_state['crit_names']) if am_final is not None and len(am_final)>0 else []

                    st.markdown("**R≥ At-Least Rules:**")
                    for txt in al_texts_new:
                        st.markdown(f'<div class="rule-box atleast">{txt}</div>', unsafe_allow_html=True)
                    if am_texts_new:
                        st.markdown("**R≤ At-Most Rules:**")
                        for txt in am_texts_new:
                            st.markdown(f'<div class="rule-box atmost">{txt}</div>', unsafe_allow_html=True)

                    # Unit by unit explanation
                    new_with_nan = np.hstack([new_matrix, np.full((len(new_matrix),1), np.nan)])
                    _, _, al_m_exp, am_m_exp = classify_units(
                        new_with_nan, al_final, am_final, increasing_0, decreasing_0)

                    st.markdown("#### Unit-by-unit explanation")
                    sel_new = st.selectbox("Select unit", new_names, key="sel_new")
                    idx_new = new_names.index(sel_new)
                    exp_new = explain_unit(idx_new, s_minus_f, s_plus_f,
                                          al_m_exp, am_m_exp,
                                          al_texts_new, am_texts_new, sel_new)
                    css = "class-ok" if not exp_new['contradictory'] and exp_new['s_minus']==exp_new['s_plus'] \
                          else "class-err" if exp_new['contradictory'] else "class-range"
                    st.markdown(f'<div class="class-box {css}"><b>{sel_new}</b> → <b>{exp_new["assignment"]}</b></div>',
                                unsafe_allow_html=True)
                    if exp_new['matched_atleast']:
                        st.markdown("**Satisfied at-least rules:**")
                        for r in exp_new['matched_atleast']:
                            st.markdown(f'<div class="rule-box atleast">{r}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No at-least rule satisfied → s⁻ = 1*")
                    if exp_new['matched_atmost']:
                        st.markdown("**Satisfied at-most rules:**")
                        for r in exp_new['matched_atmost']:
                            st.markdown(f'<div class="rule-box atmost">{r}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No at-most rule satisfied → s⁺ = p*")

                st.download_button("⬇ Download CSV", df_nc.to_csv(index=False),
                                   file_name="drsa_new_units.csv", mime="text/csv",
                                   use_container_width=True)