"""
DRSA Composite Indicator Tool
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
    classify_units, explain_unit,
    classify_new_units,
)

# ── Page config ────────────────────────────────────────────────────────────────
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
.changed-row{background:#fef9c3;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _nlen(x):
    return len(x) if x is not None and hasattr(x, '__len__') else 0

def show_rules(rules, texts, supps=None, units=None, rule_type="atleast"):
    for i, txt in enumerate(texts):
        extra = ""
        if supps is not None and units is not None and i < len(units):
            tags = "".join(f'<span class="tag">{u}</span>' for u in units[i])
            extra = f'<br><span style="color:#6b7280;font-size:0.75rem;">Support: {supps[i]:.3f} &nbsp;·&nbsp; </span>{tags}'
        st.markdown(f'<div class="rule-box {rule_type}">{txt}{extra}</div>', unsafe_allow_html=True)

def show_explanation(name, s_minus, s_plus, al_m, am_m, al_texts, am_texts, idx):
    exp = explain_unit(idx, s_minus, s_plus, al_m, am_m, al_texts, am_texts, name)
    css = "class-ok" if not exp['contradictory'] and exp['s_minus']==exp['s_plus'] \
          else "class-err" if exp['contradictory'] else "class-range"
    st.markdown(f'<div class="class-box {css}"><b>{name}</b> → <b>{exp["assignment"]}</b></div>',
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

def rules_to_df(al_texts, am_texts, label_al="at-least", label_am="at-most",
                al_supp=None, am_supp=None):
    rows = []
    for i, txt in enumerate(al_texts):
        r = {"type": label_al, "rule": txt}
        if al_supp is not None and i < len(al_supp): r["support"] = round(float(al_supp[i]), 4)
        rows.append(r)
    for i, txt in enumerate(am_texts):
        r = {"type": label_am, "rule": txt}
        if am_supp is not None and i < len(am_supp): r["support"] = round(float(am_supp[i]), 4)
        rows.append(r)
    return pd.DataFrame(rows)

def bibtex_omega():
    return """@article{corrente2026explainable,
  title     = {An explainable and interpretable composite indicator based on decision rules},
  author    = {Corrente, Salvatore and Greco, Salvatore and S{\\l}owi{\\'n}ski, Roman and Zappal{\\`a}, Silvano},
  journal   = {Omega},
  volume    = {142},
  pages     = {103513},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.omega.2026.103513}
}"""

def bibtex_softwarex():
    return """@article{zappala2026drsa,
  title   = {DRSA: A Python tool for explainable composite indicators},
  author  = {Zappal{\\`a}, Silvano and Corrente, Salvatore and Greco, Salvatore and S{\\l}owi{\\'n}ski, Roman},
  journal = {SoftwareX},
  year    = {2026},
  note    = {To appear}
}"""

def get_matching_units(rules, match_matrix, all_names, rule_type, inc, dec, crit_names):
    """For each rule, return list of unit names that match it."""
    if rules is None or len(rules) == 0 or match_matrix is None:
        return [[] for _ in rules] if rules is not None else []
    result = []
    for i in range(len(rules)):
        matched = [all_names[j] for j in range(len(all_names)) if match_matrix[j, i] == 1]
        result.append(matched)
    return result

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚖️ DRSA · Composite Indicator Tool")
st.markdown("""<div class="info-banner">
Dominance-based Rough Set Approach for explainable composite indicators.<br>
Implements <b>Algorithms 1, 2, 4</b> and full pipeline (Steps 1–7) from
Corrente, Greco, Słowiński, Zappalà — <i>Omega 142</i> (2026), 103513.
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload CSV or TXT", type=["csv","txt"],
        help="Last column = class label. Optional first column = unit names. NaN = non-reference unit.")
    sep = st.selectbox("Separator", [",",";","\\t"," "], index=0)
    sep_actual = "\t" if sep=="\\t" else sep

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    min_conf    = st.slider("Min confidence (c)", 0.0, 1.0, 1.0, 0.05)
    handle_miss = st.checkbox("Missing values (Algorithm 4)", False)
    random_seed = st.number_input("Random seed", value=1, min_value=0, step=1)

    st.markdown("---")
    st.markdown("### 📄 Cite")
    st.download_button("⬇ BibTeX — Omega 2026", bibtex_omega(),
                       file_name="corrente2026.bib", mime="text/plain",
                       use_container_width=True)
    st.download_button("⬇ BibTeX — SoftwareX", bibtex_softwarex(),
                       file_name="zappala2026drsa.bib", mime="text/plain",
                       use_container_width=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#9ca3af'>Corrente et al. (2026)<br>Omega 142, 103513</div>",
                unsafe_allow_html=True)

# ── Welcome ────────────────────────────────────────────────────────────────────
if uploaded is None:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
**File format**
- CSV or TXT
- Optional first column: unit names
- Criteria columns (numeric)
- Last column: class label
  - Integer → reference unit
  - Empty/NaN → non-reference unit
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
A3, A4 are non-reference.
""")
    with c3:
        st.markdown("""
**Workflow**
1. Load data & set directions
2. Choose: rule induction only OR full pipeline
3. Inspect classification
4. Classify new units
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

n_units    = len(df)
n_crit     = df.shape[1] - 1
crit_names = list(df.columns)[:-1]
matrix     = df.values.astype(float)
ref_mask   = ~np.isnan(matrix[:, -1])
ref_indices = np.where(ref_mask)[0].tolist()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Setup", "⚙️ Run", "🔍 Classification", "🆕 New Units"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### 📊 Dataset")
    st.dataframe(df_raw, use_container_width=True, height=250)

    st.markdown("#### 🎯 Reference units")
    selected_ref = st.multiselect(
        "Select reference units",
        options=unit_names,
        default=[unit_names[i] for i in ref_indices],
    )
    if len(selected_ref) == 0:
        st.error("Please select at least one reference unit."); st.stop()

    ref_indices_sel = [unit_names.index(n) for n in selected_ref]
    c1,c2,c3 = st.columns(3)
    c1.metric("Total units", n_units)
    c2.metric("Reference units", len(ref_indices_sel))
    c3.metric("Non-reference units", n_units - len(ref_indices_sel))

    st.markdown("#### 🔼 Criteria preference direction")
    dir_cols = st.columns(min(n_crit, 6))
    directions = {}
    for i, name in enumerate(crit_names):
        with dir_cols[i % len(dir_cols)]:
            directions[name] = st.selectbox(name, ["↑ Increasing","↓ Decreasing"], key=f"dir_{i}")

    inc = [i for i,n in enumerate(crit_names) if "Increasing" in directions[n]]
    dec = [i for i,n in enumerate(crit_names) if "Decreasing" in directions[n]]

    st.session_state.update({
        'inc': inc, 'dec': dec,
        'matrix': matrix, 'unit_names': unit_names,
        'crit_names': crit_names, 'ref_indices': ref_indices_sel,
        'n_units': n_units,
    })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if 'inc' not in st.session_state:
        st.info("Set up data and directions in the **Data & Setup** tab first.")
        st.stop()

    inc        = st.session_state['inc']
    dec        = st.session_state['dec']
    matrix     = st.session_state['matrix']
    unit_names = st.session_state['unit_names']
    crit_names = st.session_state['crit_names']
    ref_idx    = st.session_state['ref_indices']
    ref_matrix = matrix[ref_idx, :]
    ref_names  = [unit_names[i] for i in ref_idx]
    all_crit   = matrix[:, :-1]
    n_units    = st.session_state['n_units']

    mode = st.radio("Mode",
        ["🔬 Rule induction only (Steps 1-2)", "🚀 Full pipeline (Steps 1-7)"],
        horizontal=True)

    if "only" in mode:
        run_al = st.checkbox("At-least rules (R≥)", value=True, key="run_al")
        run_am = st.checkbox("At-most rules (R≤)",  value=True, key="run_am")
    else:
        run_al = run_am = True

    if st.button("▶ Run", type="primary", use_container_width=True):

        if "only" in mode:
            # ── Rule induction only ────────────────────────────────────────
            with st.spinner("Inducing rules…"):
                al_r = am_r = al_m = al_d = am_m = am_d = None
                if run_al:
                    al_r, al_m, al_d, _ = induce_atleast_rules(
                        ref_matrix, inc, dec, min_conf, handle_miss)
                if run_am:
                    am_r, am_m, am_d, _ = induce_atmost_rules(
                        ref_matrix, inc, dec, min_conf, handle_miss)

            n_al = _nlen(al_r); n_am = _nlen(am_r)
            al_texts = format_atleast_rules(al_r, inc, dec, crit_names) if n_al>0 else []
            am_texts = format_atmost_rules(am_r, inc, dec, crit_names) if n_am>0 else []
            al_supp  = compute_relative_support(al_r, al_m, al_d) if n_al>0 else []
            am_supp  = compute_relative_support(am_r, am_m, am_d) if n_am>0 else []
            al_units = get_supporting_units(al_m, al_d, ref_names) if n_al>0 else []
            am_units = get_supporting_units(am_m, am_d, ref_names) if n_am>0 else []

            st.session_state.update({
                'al_rules': al_r, 'am_rules': am_r,
                'al_texts': al_texts, 'am_texts': am_texts,
                'al_supp': al_supp, 'am_supp': am_supp,
                'al_units': al_units, 'am_units': am_units,
                'pipeline_result': None, 'mode': 'induction',
            })

            mc1,mc2,mc3 = st.columns(3)
            for col,val,lbl in zip([mc1,mc2,mc3],[n_al,n_am,n_al+n_am],
                                    ["At-least (R≥)","At-most (R≤)","Total"]):
                col.markdown(f'<div class="metric-card"><div class="value">{val}</div>'
                             f'<div class="label">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown("")
            if n_al > 0:
                st.markdown("### R≥ · At-Least Rules")
                show_rules(al_r, al_texts, al_supp, al_units, "atleast")
            if n_am > 0:
                st.markdown("### R≤ · At-Most Rules")
                show_rules(am_r, am_texts, am_supp, am_units, "atmost")
            if n_al + n_am > 0:
                st.markdown("### 💾 Export")
                df_dl = rules_to_df(al_texts, am_texts,
                                    al_supp=al_supp if n_al>0 else None,
                                    am_supp=am_supp if n_am>0 else None)
                st.download_button("⬇ Download rules CSV", df_dl.to_csv(index=False),
                                   file_name="drsa_rules.csv", mime="text/csv",
                                   use_container_width=True)

        else:
            # ── Full pipeline ──────────────────────────────────────────────
            prog = st.progress(0); status = st.empty()

            status.info("⏳ Step 2: Inducing rules from reference units…"); prog.progress(10)
            al_r, al_m, al_d, _ = induce_atleast_rules(ref_matrix, inc, dec, min_conf, handle_miss)
            am_r, am_m, am_d, _ = induce_atmost_rules(ref_matrix, inc, dec, min_conf, handle_miss)
            n_al2 = _nlen(al_r); n_am2 = _nlen(am_r)
            status.success(f"✅ Step 2: {n_al2} at-least, {n_am2} at-most"); prog.progress(20)

            status.info("⏳ Step 3-4: Greedy selection…"); prog.progress(30)
            from drsa.core.step_forward import step_forward as _sf
            sel_al, sel_am, _, _ = _sf(al_r, am_r, al_m, am_m, al_d, am_d,
                                        ref_matrix, all_crit, inc, dec,
                                        random_seed=int(random_seed))
            status.success(f"✅ Step 3-4: {_nlen(sel_al)} at-least, {_nlen(sel_am)} selected"); prog.progress(45)

            status.info("⏳ Step 5: Fixing classifications…"); prog.progress(50)
            from drsa.core.classifier import classify_units as _cu
            mat_nc = np.hstack([all_crit, np.full((n_units,1), np.nan)])
            s_minus, s_plus, _, _ = _cu(mat_nc, sel_al, sel_am, inc, dec)
            s_minus_f = s_minus.copy(); s_plus_f = s_plus.copy()
            for idx in ref_idx:
                dm_c = matrix[idx, -1]
                if not np.isnan(dm_c):
                    s_minus_f[idx] = dm_c; s_plus_f[idx] = dm_c
            mat_sm = np.hstack([all_crit, s_minus_f.reshape(-1,1)])
            mat_sp = np.hstack([all_crit, s_plus_f.reshape(-1,1)])
            status.success("✅ Step 5 done"); prog.progress(55)

            status.info("⏳ Step 6: Inducing rules from all units…"); prog.progress(60)
            al_r2, al_m2, al_d2, _ = induce_atleast_rules(mat_sm, inc, dec, min_conf, handle_miss)
            am_r2, am_m2, am_d2, _ = induce_atmost_rules(mat_sp, inc, dec, min_conf, handle_miss)
            n_al6 = _nlen(al_r2); n_am6 = _nlen(am_r2)
            status.success(f"✅ Step 6: {n_al6} at-least, {n_am6} at-most"); prog.progress(70)

            status.info("⏳ Step 7: MILP — minimal rule set…"); prog.progress(75)
            from drsa.core.milp import solve_minimal_rules as _smr
            al_min, am_min, _, _, milp_ok, milp_msg = _smr(mat_sm, al_m2, mat_sp, am_m2, al_r2, am_r2)
            al_final = al_min if (milp_ok and _nlen(al_min)>0) else al_r2
            am_final = am_min if (milp_ok and _nlen(am_min)>0) else am_r2
            if milp_ok:
                status.success(f"✅ Step 7: {_nlen(al_final)} at-least, {_nlen(am_final)} minimal rules")
            else:
                status.error(f"⚠️ Step 7: {milp_msg}")
            prog.progress(88)

            status.info("⏳ Final classification…")
            sm7, sp7, _, _ = _cu(mat_nc, al_final, am_final, inc, dec)
            prog.progress(100); status.success("🎉 Pipeline complete!")

            # Format texts
            al_texts_max = format_atleast_rules(al_r2, inc, dec, crit_names) if n_al6>0 else []
            am_texts_max = format_atmost_rules(am_r2, inc, dec, crit_names) if n_am6>0 else []
            al_texts_min = format_atleast_rules(al_final, inc, dec, crit_names) if _nlen(al_final)>0 else []
            am_texts_min = format_atmost_rules(am_final, inc, dec, crit_names) if _nlen(am_final)>0 else []

            # Units matching rules — for all units vs each rule
            # Maximal rules match matrices
            all_nc = np.hstack([all_crit, np.full((n_units,1), np.nan)])
            _, _, al_m_all_max, am_m_all_max = _cu(all_nc, al_r2, am_r2, inc, dec)
            _, _, al_m_all_min, am_m_all_min = _cu(all_nc, al_final, am_final, inc, dec)
            al_units_max = get_matching_units(al_r2, al_m_all_max, unit_names, 'atleast', inc, dec, crit_names)
            am_units_max = get_matching_units(am_r2, am_m_all_max, unit_names, 'atmost', inc, dec, crit_names)
            al_units_min = get_matching_units(al_final, al_m_all_min, unit_names, 'atleast', inc, dec, crit_names)
            am_units_min = get_matching_units(am_final, am_m_all_min, unit_names, 'atmost', inc, dec, crit_names)

            st.session_state.update({
                'al_rules': al_final, 'am_rules': am_final,
                'al_rules_max': al_r2, 'am_rules_max': am_r2,
                'al_texts': al_texts_min, 'am_texts': am_texts_min,
                'al_texts_max': al_texts_max, 'am_texts_max': am_texts_max,
                'al_units_min': al_units_min, 'am_units_min': am_units_min,
                'al_units_max': al_units_max, 'am_units_max': am_units_max,
                'al_match2': al_m2, 'am_match2': am_m2,
                'matrix_s_minus': mat_sm, 'matrix_s_plus': mat_sp,
                'classification_final': np.column_stack([sm7, sp7]),
                'mode': 'pipeline',
                'pipeline_result': {
                    'step2': (n_al2, n_am2), 'step3': (_nlen(sel_al), _nlen(sel_am)),
                    'step6': (n_al6, n_am6), 'step7': (_nlen(al_final), _nlen(am_final)),
                    'milp_ok': milp_ok, 'milp_msg': milp_msg,
                },
            })

            # Summary
            st.markdown("#### Pipeline summary")
            df_steps = pd.DataFrame([
                ("Step 2 – Reference units", n_al2, n_am2, n_al2+n_am2),
                ("Step 3 – Greedy selection", _nlen(sel_al), _nlen(sel_am), _nlen(sel_al)+_nlen(sel_am)),
                ("Step 6 – All units", n_al6, n_am6, n_al6+n_am6),
                ("Step 7 – Minimal (MILP)", _nlen(al_final), _nlen(am_final), _nlen(al_final)+_nlen(am_final)),
            ], columns=["Step","At-least","At-most","Total"])
            st.dataframe(df_steps, use_container_width=True, hide_index=True)

            # Maximal rules expander
            with st.expander(f"📂 Maximal rules — Step 6 ({n_al6} at-least, {n_am6} at-most)"):
                if al_texts_max:
                    st.markdown("**R≥ At-Least (maximal)**")
                    show_rules(al_r2, al_texts_max, units=al_units_max, rule_type="atleast")
                if am_texts_max:
                    st.markdown("**R≤ At-Most (maximal)**")
                    show_rules(am_r2, am_texts_max, units=am_units_max, rule_type="atmost")

            # Minimal rules
            if al_texts_min:
                st.markdown("### R≥ · Minimal At-Least Rules — Step 7")
                show_rules(al_final, al_texts_min, units=al_units_min, rule_type="atleast")
            if am_texts_min:
                st.markdown("### R≤ · Minimal At-Most Rules — Step 7")
                show_rules(am_final, am_texts_min, units=am_units_min, rule_type="atmost")

            # Download — two persistent buttons
            st.markdown("### 💾 Export rules")
            c1, c2 = st.columns(2)
            with c1:
                df_min = rules_to_df(al_texts_min, am_texts_min,
                                     "at-least (minimal)", "at-most (minimal)")
                st.download_button("⬇ Minimal rules CSV", df_min.to_csv(index=False),
                                   file_name="drsa_rules_minimal.csv", mime="text/csv",
                                   use_container_width=True)
            with c2:
                df_max = rules_to_df(al_texts_max, am_texts_max,
                                     "at-least (maximal)", "at-most (maximal)")
                st.download_button("⬇ Maximal rules CSV", df_max.to_csv(index=False),
                                   file_name="drsa_rules_maximal.csv", mime="text/csv",
                                   use_container_width=True)

    else:
        # ── Show previously computed results ───────────────────────────────
        al_texts = st.session_state.get('al_texts', [])
        am_texts = st.session_state.get('am_texts', [])
        if not al_texts and not am_texts:
            st.info("Press **▶ Run** to start.")
        else:
            mode_saved = st.session_state.get('mode', '')
            if mode_saved == 'pipeline':
                res = st.session_state.get('pipeline_result', {})
                if res:
                    df_steps = pd.DataFrame([
                        ("Step 2", *res['step2'], sum(res['step2'])),
                        ("Step 3", *res['step3'], sum(res['step3'])),
                        ("Step 6", *res['step6'], sum(res['step6'])),
                        ("Step 7", *res['step7'], sum(res['step7'])),
                    ], columns=["Step","At-least","At-most","Total"])
                    st.dataframe(df_steps, use_container_width=True, hide_index=True)

                al_r2     = st.session_state.get('al_rules_max')
                am_r2     = st.session_state.get('am_rules_max')
                al_final  = st.session_state.get('al_rules')
                am_final  = st.session_state.get('am_rules')
                al_texts_max  = st.session_state.get('al_texts_max', [])
                am_texts_max  = st.session_state.get('am_texts_max', [])
                al_units_max  = st.session_state.get('al_units_max', [])
                am_units_max  = st.session_state.get('am_units_max', [])
                al_units_min  = st.session_state.get('al_units_min', [])
                am_units_min  = st.session_state.get('am_units_min', [])

                if al_texts_max or am_texts_max:
                    with st.expander("📂 Maximal rules — Step 6"):
                        if al_texts_max:
                            st.markdown("**R≥ At-Least (maximal)**")
                            show_rules(al_r2, al_texts_max, units=al_units_max, rule_type="atleast")
                        if am_texts_max:
                            st.markdown("**R≤ At-Most (maximal)**")
                            show_rules(am_r2, am_texts_max, units=am_units_max, rule_type="atmost")

                if al_texts:
                    st.markdown("### R≥ · Minimal At-Least Rules")
                    show_rules(al_final, al_texts, units=al_units_min, rule_type="atleast")
                if am_texts:
                    st.markdown("### R≤ · Minimal At-Most Rules")
                    show_rules(am_final, am_texts, units=am_units_min, rule_type="atmost")

                # Persistent download buttons
                st.markdown("### 💾 Export rules")
                c1, c2 = st.columns(2)
                with c1:
                    df_min = rules_to_df(al_texts, am_texts,
                                         "at-least (minimal)", "at-most (minimal)")
                    st.download_button("⬇ Minimal rules CSV", df_min.to_csv(index=False),
                                       file_name="drsa_rules_minimal.csv", mime="text/csv",
                                       use_container_width=True)
                with c2:
                    df_max = rules_to_df(al_texts_max, am_texts_max,
                                         "at-least (maximal)", "at-most (maximal)")
                    st.download_button("⬇ Maximal rules CSV", df_max.to_csv(index=False),
                                       file_name="drsa_rules_maximal.csv", mime="text/csv",
                                       use_container_width=True)
            else:
                al_supp  = st.session_state.get('al_supp', [])
                am_supp  = st.session_state.get('am_supp', [])
                al_units = st.session_state.get('al_units', [])
                am_units = st.session_state.get('am_units', [])
                al_r     = st.session_state.get('al_rules')
                am_r     = st.session_state.get('am_rules')
                if al_texts:
                    st.markdown("### R≥ · At-Least Rules")
                    show_rules(al_r, al_texts, al_supp, al_units, "atleast")
                if am_texts:
                    st.markdown("### R≤ · At-Most Rules")
                    show_rules(am_r, am_texts, am_supp, am_units, "atmost")
                if al_texts or am_texts:
                    st.markdown("### 💾 Export")
                    df_dl = rules_to_df(al_texts, am_texts)
                    st.download_button("⬇ Download rules CSV", df_dl.to_csv(index=False),
                                       file_name="drsa_rules.csv", mime="text/csv",
                                       use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'al_rules' not in st.session_state:
        st.info("Run the analysis in the **Run** tab first.")
    else:
        al_rules   = st.session_state['al_rules']
        am_rules   = st.session_state['am_rules']
        al_texts   = st.session_state.get('al_texts', [])
        am_texts   = st.session_state.get('am_texts', [])
        unit_names = st.session_state['unit_names']
        matrix     = st.session_state['matrix']
        inc        = st.session_state['inc']
        dec        = st.session_state['dec']

        st.markdown("#### Classification of all units — equation (4)")

        if 'classification_final' in st.session_state:
            cl = st.session_state['classification_final']
            s_minus = cl[:, 0]; s_plus = cl[:, 1]
            mat_nc = np.hstack([matrix[:,:-1], np.full((len(matrix),1), np.nan)])
            _, _, al_m, am_m = classify_units(mat_nc, al_rules, am_rules, inc, dec)
        else:
            mat_nc = np.hstack([matrix[:,:-1], np.full((len(matrix),1), np.nan)])
            s_minus, s_plus, al_m, am_m = classify_units(mat_nc, al_rules, am_rules, inc, dec)

        rows = []
        for i, name in enumerate(unit_names):
            sm, sp = int(s_minus[i]), int(s_plus[i])
            contra = sm > sp
            assign = f"Class {sm}" if sm==sp else (f"Class {sm} to {sp}" if not contra
                     else f"CONTRADICTORY (s⁻={sm} > s⁺={sp})")
            rows.append({"Unit":name,"s⁻":sm,"s⁺":sp,"Assignment":assign,
                         "Status":"⚠️ Contradictory" if contra else "✅ OK"})

        df_class = pd.DataFrame(rows)
        st.dataframe(df_class, use_container_width=True, height=380)
        n_ok = df_class['Status'].str.contains('OK').sum()
        n_co = df_class['Status'].str.contains('Contr').sum()
        ca,cb = st.columns(2)
        ca.metric("Non-contradictory", n_ok)
        cb.metric("Contradictory", n_co)
        if n_co == 0:
            st.success("All units classified without contradictions.")
        else:
            st.warning(f"{n_co} unit(s) have contradictory assignments.")

        st.markdown("#### Unit-by-unit explanation")
        sel = st.selectbox("Select unit", unit_names, key="sel_tab3")
        show_explanation(sel, s_minus, s_plus, al_m, am_m,
                         al_texts, am_texts, unit_names.index(sel))

        st.download_button("⬇ Download classification CSV", df_class.to_csv(index=False),
                           file_name="drsa_classification.csv", mime="text/csv",
                           use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if st.session_state.get('mode') != 'pipeline':
        st.info("Run the **Full pipeline** (Steps 1-7) first to enable new unit classification.")
    else:
        st.markdown("#### 🆕 Classify new units")
        st.markdown("Upload new units **without** the class column. "
                    "The tool applies MILP (6), (7), (8) to handle contradictions.")

        new_file = st.file_uploader("Upload new units", type=["csv","txt"], key="new_file")
        sep2 = st.selectbox("Separator", [",",";","\\t"," "], key="sep2")
        sep2_act = "\t" if sep2=="\\t" else sep2

        if new_file is not None:
            try:
                df_new_raw = pd.read_csv(new_file, sep=sep2_act, engine="python")
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
            new_matrix = df_new.values.astype(float)

            st.dataframe(df_new_raw, use_container_width=True, height=180)

            if st.button("▶ Classify new units", type="primary", use_container_width=True):
                prog2 = st.progress(0); stat2 = st.empty()
                stat2.info("⏳ Running MILP (6), (7), (8)…"); prog2.progress(20)

                new_res = classify_new_units(
                    new_matrix,
                    st.session_state['matrix_s_minus'],
                    st.session_state['matrix_s_plus'],
                    st.session_state['al_rules_max'],
                    st.session_state['al_match2'],
                    st.session_state['am_rules_max'],
                    st.session_state['am_match2'],
                    st.session_state['inc'],
                    st.session_state['dec'],
                )
                prog2.progress(90)

                if 'error' in new_res:
                    st.error(new_res['error']); st.stop()

                n_co = new_res['n_contradictions']
                if n_co == 0:
                    stat2.success("✅ No contradictions — MILP (8) applied for minimal rules")
                elif new_res.get('milp_success'):
                    stat2.success(f"✅ {n_co} contradictions resolved via MILP (6) and (7)")
                else:
                    stat2.warning(f"⚠️ {n_co} contradictions — {new_res.get('milp_message','')}")
                prog2.progress(100)

                # Pick final rules
                al_fin = new_res.get('step8_al_rules')
                am_fin = new_res.get('step8_am_rules')
                if al_fin is None or _nlen(al_fin) == 0:
                    al_fin = new_res.get('step7_al_rules', st.session_state['al_rules'])
                if am_fin is None or _nlen(am_fin) == 0:
                    am_fin = new_res.get('step7_am_rules', st.session_state['am_rules'])

                al_texts_new = format_atleast_rules(al_fin, st.session_state['inc'],
                    st.session_state['dec'], st.session_state['crit_names']) if _nlen(al_fin)>0 else []
                am_texts_new = format_atmost_rules(am_fin, st.session_state['inc'],
                    st.session_state['dec'], st.session_state['crit_names']) if _nlen(am_fin)>0 else []

                st.session_state.update({
                    'new_s_minus':   new_res['final_s_minus'],
                    'new_s_plus':    new_res['final_s_plus'],
                    'new_names':     new_names,
                    'new_matrix':    new_matrix,
                    'new_al_rules':  al_fin,
                    'new_am_rules':  am_fin,
                    'new_al_texts':  al_texts_new,
                    'new_am_texts':  am_texts_new,
                    'new_res':       new_res,
                    'new_changed':   new_res.get('changed_units', []),
                })

        # ── Show results (always from session_state, safe) ─────────────────
        if 'new_s_minus' in st.session_state:
            s_minus_f  = st.session_state['new_s_minus']
            s_plus_f   = st.session_state['new_s_plus']
            new_names  = st.session_state['new_names']
            new_matrix = st.session_state['new_matrix']
            al_fin     = st.session_state['new_al_rules']
            am_fin     = st.session_state['new_am_rules']
            al_texts_new  = st.session_state['new_al_texts']
            am_texts_new  = st.session_state['new_am_texts']
            new_res    = st.session_state['new_res']
            changed    = st.session_state.get('new_changed', [])

            # ── Classification results: A + A_new ──────────────────────────
            st.markdown("#### Classification results")

            # Get original classifications of A
            mat_sm_orig = st.session_state['matrix_s_minus']
            mat_sp_orig = st.session_state['matrix_s_plus']
            all_unit_names = st.session_state['unit_names']
            cl_all = new_res.get('classification_all')

            rows_all = []
            # Existing units A
            for i, name in enumerate(all_unit_names):
                sm_orig = int(mat_sm_orig[i, -1])
                sp_orig = int(mat_sp_orig[i, -1])
                if cl_all is not None and i < len(cl_all):
                    sm_new = int(cl_all[i, 0])
                    sp_new = int(cl_all[i, 1])
                else:
                    sm_new = sm_orig; sp_new = sp_orig
                changed_flag = i in changed
                assign_orig = f"Class {sm_orig}" if sm_orig==sp_orig else f"Class {sm_orig}–{sp_orig}"
                assign_new  = f"Class {sm_new}"  if sm_new==sp_new   else f"Class {sm_new}–{sp_new}"
                rows_all.append({
                    "Unit": name,
                    "s⁻ (prev)": sm_orig, "s⁺ (prev)": sp_orig,
                    "s⁻ (new)": sm_new,  "s⁺ (new)": sp_new,
                    "Assignment": assign_new,
                    "Changed": "⚠️ Yes" if changed_flag else "",
                    "_changed": changed_flag,
                })
            # New units A_new
            n_existing = len(all_unit_names)
            for k, name in enumerate(new_names):
                sm = int(s_minus_f[k]); sp = int(s_plus_f[k])
                contra = sm > sp
                assign = f"Class {sm}" if sm==sp else (f"Class {sm}–{sp}" if not contra
                         else f"CONTRADICTORY")
                rows_all.append({
                    "Unit": name,
                    "s⁻ (prev)": "—", "s⁺ (prev)": "—",
                    "s⁻ (new)": sm, "s⁺ (new)": sp,
                    "Assignment": assign,
                    "Changed": "🆕 New" if not contra else "⚠️ Contradictory",
                    "_changed": True,
                })

            df_all = pd.DataFrame(rows_all)
            df_display = df_all.drop(columns=['_changed'])

            # Highlight changed rows
            def highlight_changed(row):
                orig_row = rows_all[row.name]
                if orig_row['_changed']:
                    return ['background-color: #fef9c3'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_display.style.apply(highlight_changed, axis=1),
                use_container_width=True, height=400
            )

            n_changed = len(changed)
            n_new_ok  = sum(1 for k in range(len(new_names))
                           if int(s_minus_f[k]) <= int(s_plus_f[k]))
            ca, cb, cc = st.columns(3)
            ca.metric("Changed in A", n_changed)
            cb.metric("New units classified", n_new_ok)
            cc.metric("New contradictions", len(new_names) - n_new_ok)

            # ── Summary result ─────────────────────────────────────────────
            with st.expander("🔍 Summary result"):
                changed_names = [all_unit_names[i] for i in changed if i < len(all_unit_names)]
                st.markdown(f"**Changed classification of previous units:** "
                            f"{', '.join(changed_names) if changed_names else 'None'}")
                def _nr(x): return _nlen(x) if x is not None else 0
                st.markdown(f"**MILP(7) rules selected:** "
                            f"{_nr(new_res.get('step7_al_rules'))} at-least, "
                            f"{_nr(new_res.get('step7_am_rules'))} at-most")
                st.markdown(f"**MILP(8) minimal rules:** "
                            f"{_nr(new_res.get('step8_al_rules'))} at-least, "
                            f"{_nr(new_res.get('step8_am_rules'))} at-most")
                st.markdown(f"**Status:** {new_res.get('milp_message','N/A')}")

            # ── Minimal rules for A ∪ A_new ────────────────────────────────
            if al_texts_new or am_texts_new:
                with st.expander(f"📂 Minimal rules for A ∪ A_new "
                                 f"({_nlen(al_fin)} at-least, {_nlen(am_fin)} at-most)"):
                    if al_texts_new:
                        st.markdown("**R≥ At-Least:**")
                        show_rules(al_fin, al_texts_new, rule_type="atleast")
                    if am_texts_new:
                        st.markdown("**R≤ At-Most:**")
                        show_rules(am_fin, am_texts_new, rule_type="atmost")

            # ── Unit explanation ───────────────────────────────────────────
            st.markdown("#### Unit-by-unit explanation")
            sel_new = st.selectbox("Select unit", new_names, key="sel_new")
            idx_new = new_names.index(sel_new)
            new_nc  = np.hstack([new_matrix, np.full((len(new_matrix),1), np.nan)])
            _, _, al_m_new, am_m_new = classify_units(
                new_nc, al_fin, am_fin,
                st.session_state['inc'], st.session_state['dec'])
            show_explanation(sel_new, s_minus_f, s_plus_f,
                             al_m_new, am_m_new, al_texts_new, am_texts_new, idx_new)

            st.download_button("⬇ Download classification CSV",
                               df_display.to_csv(index=False),
                               file_name="drsa_new_units.csv",
                               mime="text/csv", use_container_width=True)
