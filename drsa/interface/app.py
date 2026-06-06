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
                with st.spinner("Running full pipeline (Steps 1-7)… this may take a moment."):
                    res = run_pipeline(
                        matrix, ref_indices, increasing_0, decreasing_0,
                        min_confidence=min_conf,
                        handle_missing=handle_missing,
                        random_seed=int(random_seed)
                    )
                st.session_state['pipeline_result'] = res
                st.session_state['al_rules'] = res.get('step7_al_rules') or res.get('step6_al_rules')
                st.session_state['am_rules'] = res.get('step7_am_rules') or res.get('step6_am_rules')
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
    if 'al_rules' not in st.session_state:
        st.info("Run the pipeline first in the **Run Pipeline** tab.")
    else:
        st.markdown("#### Classify new units")
        st.markdown("Upload a file with the same criteria columns, **without** the class column.")

        new_file = st.file_uploader("Upload new units", type=["csv","txt"], key="new_units")
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

            matrix_new = np.hstack([df_new.values.astype(float),
                                    np.full((len(df_new),1), np.nan)])

            al_rules     = st.session_state['al_rules']
            am_rules     = st.session_state['am_rules']
            al_texts     = st.session_state.get('al_texts', [])
            am_texts     = st.session_state.get('am_texts', [])
            increasing_0 = st.session_state['increasing_0']
            decreasing_0 = st.session_state['decreasing_0']

            s_minus, s_plus, al_m, am_m = classify_units(
                matrix_new, al_rules, am_rules, increasing_0, decreasing_0)

            rows = []
            for i,name in enumerate(new_names):
                exp = explain_unit(i, s_minus, s_plus, al_m, am_m, al_texts, am_texts, name)
                rows.append({"Unit":name,"s⁻":int(exp['s_minus']),"s⁺":int(exp['s_plus']),
                             "Assignment":exp['assignment'],
                             "Status":"⚠️ Contradictory" if exp['contradictory'] else "✅ OK"})

            df_nc = pd.DataFrame(rows)
            st.markdown("#### Results")
            st.dataframe(df_nc, use_container_width=True)

            sel_new = st.selectbox("Select unit to explain", new_names, key="sel_new")
            idx_new = new_names.index(sel_new)
            exp_new = explain_unit(idx_new, s_minus, s_plus, al_m, am_m,
                                   al_texts, am_texts, sel_new)
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
