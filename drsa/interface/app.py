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
st.set_page_config(page_title="EI-SCORE", page_icon="📊", layout="wide")

# ── MathJax ───────────────────────────────────────────────────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$'],['\\(','\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
""", height=0)

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
<style>
/* Hide fullscreen button from images */
[data-testid="stImage"] button,
[data-testid="stImageContainer"] button {
    display: none !important;
}
/* Hide download button from all dataframe toolbars */
[data-testid="stDataFrameResizable"] button[title="Download as CSV"],
[data-testid="stDataFrameResizable"] button[aria-label="Download as CSV"],
[data-testid="stElementToolbar"] button[title="Download as CSV"],
[data-testid="stElementToolbar"] button[aria-label="Download as CSV"],
button[title="Download as CSV"],
button[aria-label="Download as CSV"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _nlen(x):
    return len(x) if x is not None and hasattr(x, '__len__') else 0

def show_rules(rules, texts, supps=None, units=None, rule_type="atleast"):
    for i, txt in enumerate(texts):
        extra = ""
        if units is not None and i < len(units) and len(units[i]) > 0:
            tags = "".join(f'<span class="tag">{u}</span>' for u in units[i])
            if supps is not None and i < len(supps):
                extra = f'<br><span style="color:#6b7280;font-size:0.75rem;">Support: {supps[i]:.3f} &nbsp;·&nbsp; </span>{tags}'
            else:
                extra = f'<br>{tags}'
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

def rules_to_csv(al_texts, am_texts, crit_names, inc, dec,
                 al_rules=None, am_rules=None, score_map=None):
    """
    Export rules in self-contained CSV format:
      #directions,increasing,decreasing,...
      #mode,class|score
      #labels,v1,v2,...  (only if score)
      type,class,crit1,crit2,...
      at-least,2,0.762,,...
    """
    import io
    buf = io.StringIO()
    dirs = ['increasing' if i in inc else 'decreasing' for i in range(len(crit_names))]
    buf.write('#directions,' + ','.join(dirs) + '\n')
    if score_map:
        buf.write('#mode,score\n')
    else:
        buf.write('#mode,class\n')
    buf.write('type,class,' + ','.join(crit_names) + '\n')
    def write_rules(rules, rtype):
        if rules is None or len(rules) == 0: return
        for rule in rules:
            cl = int(rule[-1])
            cl_out = score_map[cl] if score_map and cl in score_map else cl
            vals = [''] * len(crit_names)
            body = rule[:-1]
            pos = [int(body[k*2]) - 1 for k in range(len(body)//2) if body[k*2] != 0]
            thr = [body[k*2+1] for k in range(len(body)//2) if body[k*2] != 0]
            for p, t in zip(pos, thr):
                if 0 <= p < len(crit_names):
                    # Un-negate thresholds:
                    # at-least: decreasing criteria are negated internally
                    # at-most:  increasing criteria are negated internally
                    if rtype == 'at-least' and p in dec:
                        t = -t
                    elif rtype == 'at-most' and p in inc:
                        t = -t
                    vals[p] = str(round(float(t), 6))
            buf.write(rtype + ',' + str(cl_out) + ',' + ','.join(vals) + '\n')
    write_rules(al_rules, 'at-least')
    write_rules(am_rules, 'at-most')
    return buf.getvalue()

def _fmt_class(val, score_map):
    """Convert class index to score label for display."""
    if score_map and int(val) in score_map:
        return score_map[int(val)]
    return int(val)

def _assign_str(sm, sp, score_map):
    """Build assignment string using score or class labels."""
    lm = _fmt_class(sm, score_map)
    lp = _fmt_class(sp, score_map)
    mode = "Score" if score_map else "Class"
    if sm == sp:
        return f"{mode} {lm}"
    elif sm <= sp:
        return f"{mode} {lm} to {lp}"
    else:
        return f"CONTRADICTORY ({lm} > {lp})"

def bibtex():
    return """@article{corrente2026,
  title     = {An explainable and interpretable composite indicator based on decision rules},
  author    = {Corrente, Salvatore and Greco, Salvatore and S{\\l}owi{\\'n}ski, Roman and Zappal{\\`a}, Silvano},
  journal   = {Omega},
  volume    = {142},
  pages     = {103513},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.omega.2026.103513}
}
@article{greco2001,
  title     = {Rough sets theory for multicriteria decision analysis},
  author    = {Greco, Salvatore and Matarazzo, Benedetto and Slowinski, Roman},
  journal   = {European journal of operational research},
  volume    = {129},
  number    = {1},
  pages     = {1--47},
  year      = {2001},
  publisher = {Elsevier},
  doi       = {10.1016/S0377-2217(00)00167-3}
}
"""

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
# ── Logo + Title ──────────────────────────────────────────────────────────────
import os as _os
_logo_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'assets', 'logo.png')
_col_logo, _col_title = st.columns([1, 5])
with _col_logo:
    if _os.path.exists(_logo_path):
        st.image(_logo_path, width=150)
with _col_title:
    st.title("EI-SCORE")
    st.markdown("**Explainable-Interpretable and Simple Customized Overall Ranking Engine**")
st.markdown("""<div class="info-banner">
User-friendly GUI to build your customized composite indicator based on Decision Rules
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload CSV or TXT", type=["csv","txt"],
        help="Last column = class label. Optional first column = unit names. NaN = non-reference unit.")
    sep = st.selectbox("Separator", [",",";","\\t"," "], index=0)
    sep_actual = "\t" if sep=="\\t" else sep

    st.markdown("---")
    st.markdown("### 🚀 Quick start")
    st.markdown(
        "1. **Data & Setup** — upload dataset, set directions and settings\n"
        "2. **Run** — induce only rules or full pipeline\n"
        "3. **Assignment** — inspect assignment of all units\n"
        "4. **New Units** — assign new units with previous induced rules\n"
        "5. **Apply Rules** — load saved rules and assign units"
    )

    # ── User guide ────────────────────────────────────────────────────────────
    st.markdown("### 📖 User guide")
    _guide_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        'assets', 'user_guide.pdf')
    if _os.path.exists(_guide_path):
        with open(_guide_path, 'rb') as _f:
            st.download_button("⬇ User Guide (PDF)", _f.read(),
                               file_name="EI-SCORE_user_guide.pdf",
                               mime="application/pdf",
                               use_container_width=True)
    else:
        st.caption("User guide coming soon.")

    st.markdown("---")
    st.markdown("### 📄 Cite")
    st.download_button("⬇ BibTeX", bibtex(),
                       file_name="corrente2026.bib", mime="text/plain",
                       use_container_width=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af'>- S. Corrente, S. Greco, R. Słowiński, S. Zappalà<br>Omega 142, 103513, 2026<br><a href='https://doi.org/10.1016/j.omega.2026.103513' target='_blank' style='color:#9ca3af'>DOI:10.1016/j.omega.2026.103513</a></div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af'>- S. Greco, B. Matarazzo, R. Słowiński<br>European Journal of Operational Research 129(1), 1-47, 2001<br><a href='https://doi.org/10.1016/S0377-2217(00)00167-3' target='_blank' style='color:#9ca3af'>DOI:10.1016/S0377-2217(00)00167-3</a></div>",
        unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data & Setup", "⚙️ Run", "🔍 Assignment", "🆕 New Units", "📂 Apply Rules"
])


# ── Welcome ────────────────────────────────────────────────────────────────────
if uploaded is None:
    with tab1:
        c1,c2 = st.columns(2)
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
        sample = pd.DataFrame({
            'Name':['A1','A2','A3','A4','A5','A6','A7','A8'],
            'g1':[4,5,7,3,6,8,5,4],'g2':[3,4,6,2,5,7,4,3],
            'g3':[2,3,5,1,4,6,3,2],
            'class':[1,2,3,1,'','',2,'']
        })
        st.download_button("⬇ Download sample CSV", sample.to_csv(index=False),
                           file_name="drsa_sample.csv", mime="text/csv", key="dl_sample")
    with tab2: st.info("Upload a file in the sidebar to get started.")
    with tab3: st.info("Upload a file in the sidebar to get started.")
    with tab4: st.info("Upload a file in the sidebar to get started.")

if uploaded is not None:
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
    matrix_raw = df.values.astype(float)
    # Auto-detect missing values in criteria columns
    handle_miss = bool(np.isnan(matrix_raw[:, :-1]).any())
    ref_mask   = ~np.isnan(matrix_raw[:, -1])
    # Score map initialized — will be set by radio in Tab 2
    score_map     = st.session_state.get('score_map')
    score_map_inv = st.session_state.get('score_map_inv')
    # Apply score mapping to matrix if needed
    col_mode_val = st.session_state.get('col_mode_radio', 'Classification')
    if col_mode_val == 'Scoring' and score_map_inv:
        matrix = matrix_raw.copy()
        for i in range(len(matrix)):
            if not np.isnan(matrix[i, -1]) and matrix[i, -1] in score_map_inv:
                matrix[i, -1] = score_map_inv[matrix[i, -1]]
    else:
        matrix = matrix_raw.copy()
    ref_indices = np.where(ref_mask)[0].tolist()
    # score_map will be set in tab1 after user choice
    matrix = matrix_raw.copy()

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

        st.markdown("#### ⚙️ Settings")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            min_conf = st.slider("Min confidence (c)", 0.0, 1.0, 1.0, 0.05)
        with sc2:
            pass  # missing value detection is automatic
        with sc3:
            random_seed = st.number_input("Random seed", value=1, min_value=0, step=1)

        st.session_state.update({
            'inc': inc, 'dec': dec,
            'matrix': matrix, 'unit_names': unit_names,
            'crit_names': crit_names, 'ref_indices': ref_indices_sel,
            'n_units': n_units,
            'min_conf': min_conf, 'handle_miss': handle_miss, 'random_seed': random_seed,
            'score_map': score_map, 'score_map_inv': score_map_inv,
        })

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 2
    # ══════════════════════════════════════════════════════════════════════════════
    with tab2:
        if 'inc' not in st.session_state:
            st.info("Set up data and directions in the **Data & Setup** tab first.")
            st.stop()

        # ── Problem type ───────────────────────────────────────────────────────
        col_mode_val = st.radio(
            "**Problem type:**",
            ["Classification", "Scoring"], horizontal=True, key="col_mode_radio"
        )
        # Build score mapping
        _raw_sc = sorted(matrix_raw[:, -1][~np.isnan(matrix_raw[:, -1])].tolist())
        _uniq_sc = sorted(set(_raw_sc))
        if col_mode_val == "Scoring":
            score_map     = {i+1: v for i, v in enumerate(_uniq_sc)}
            score_map_inv = {v: i+1 for i, v in enumerate(_uniq_sc)}
            matrix = matrix_raw.copy()
            for _j in range(len(matrix)):
                if not np.isnan(matrix[_j, -1]):
                    matrix[_j, -1] = score_map_inv[matrix[_j, -1]]
        else:
            score_map = None; score_map_inv = None
            matrix = matrix_raw.copy()
        st.session_state["score_map"]     = score_map
        st.session_state["score_map_inv"] = score_map_inv
        st.session_state["matrix"]        = matrix
        st.markdown("---")

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
        score_map  = st.session_state.get('score_map')

        mode = st.radio("Mode",
            ["🔬 Rule induction only", "🚀 Full pipeline"],
            horizontal=True)

        if "induction only" in mode.lower():
            run_al = st.checkbox("At-least rules (R≥)", value=True, key="run_al")
            run_am = st.checkbox("At-most rules (R≤)",  value=True, key="run_am")
        else:
            run_al = run_am = True

        if st.button("▶ Run", type="primary", use_container_width=True):

            if "induction only" in mode.lower():
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
                al_texts = format_atleast_rules(al_r, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if n_al>0 else []
                am_texts = format_atmost_rules(am_r, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if n_am>0 else []
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
                    with st.expander(f"$\\mathcal{{R}}^{{\\geqslant}}$ · At-Least Rules ({n_al})", expanded=False):
                        show_rules(al_r, al_texts, al_supp, al_units, "atleast")
                if n_am > 0:
                    with st.expander(f"$\\mathcal{{R}}^{{\\leqslant}}$ · At-Most Rules ({n_am})", expanded=False):
                        show_rules(am_r, am_texts, am_supp, am_units, "atmost")
                if n_al + n_am > 0:
                    st.markdown("### 💾 Export")
                    csv_rules = rules_to_csv(al_texts, am_texts, crit_names, inc, dec,
                                             al_rules=al_r if n_al>0 else None,
                                             am_rules=am_r if n_am>0 else None,
                                       score_map=st.session_state.get('score_map'))
                    st.download_button("⬇ Download rules CSV", csv_rules,
                                       file_name="drsa_rules.csv", mime="text/csv",
                                       key="dl_rules_ind", use_container_width=True)

            else:
                # ── Full pipeline ──────────────────────────────────────────────
                prog = st.progress(0); status = st.empty()

                status.info("⏳ Step 1/4: Inducing rules from reference units…"); prog.progress(10)
                al_r, al_m, al_d, _ = induce_atleast_rules(ref_matrix, inc, dec, min_conf, handle_miss)
                am_r, am_m, am_d, _ = induce_atmost_rules(ref_matrix, inc, dec, min_conf, handle_miss)
                n_al2 = _nlen(al_r); n_am2 = _nlen(am_r)
                status.success(f"✅ Step 1/4: {n_al2} at-least, {n_am2} at-most"); prog.progress(20)

                status.info("⏳ Step 2/4: Greedy selection…"); prog.progress(30)
                from drsa.core.step_forward import step_forward as _sf
                sel_al, sel_am, _, _ = _sf(al_r, am_r, al_m, am_m, al_d, am_d,
                                            ref_matrix, all_crit, inc, dec,
                                            random_seed=int(random_seed))
                status.success(f"✅ Step 2/4: {_nlen(sel_al)} at-least, {_nlen(sel_am)} selected"); prog.progress(45)

                status.info("⏳ Step 3/4: Fixing assignments…"); prog.progress(50)
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
                status.success("✅ Step 3/4: done"); prog.progress(55)

                status.info("⏳ Step 4/5: Inducing rules from all units…"); prog.progress(60)
                al_r2, al_m2, al_d2, _ = induce_atleast_rules(mat_sm, inc, dec, min_conf, handle_miss)
                am_r2, am_m2, am_d2, _ = induce_atmost_rules(mat_sp, inc, dec, min_conf, handle_miss)
                n_al6 = _nlen(al_r2); n_am6 = _nlen(am_r2)
                status.success(f"✅ Step 4/5: {n_al6} at-least, {n_am6} at-most"); prog.progress(70)

                status.info("⏳ Step 5/5: Find minimal rule set…"); prog.progress(75)
                from drsa.core.milp import solve_minimal_rules as _smr
                al_min, am_min, _, _, milp_ok, milp_msg = _smr(mat_sm, al_m2, mat_sp, am_m2, al_r2, am_r2)
                al_final = al_min if (milp_ok and _nlen(al_min)>0) else al_r2
                am_final = am_min if (milp_ok and _nlen(am_min)>0) else am_r2
                if milp_ok:
                    status.success(f"✅ Step 5/5: {_nlen(al_final)} at-least, {_nlen(am_final)} minimal rules")
                else:
                    status.error(f"⚠️ Step 5/5: {milp_msg}")
                prog.progress(88)

                status.info("⏳ Final assignment…")
                sm7, sp7, _, _ = _cu(mat_nc, al_final, am_final, inc, dec)
                prog.progress(100); status.success("🎉 Pipeline complete!")

                # Format texts
                al_texts_max = format_atleast_rules(al_r2, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if n_al6>0 else []
                am_texts_max = format_atmost_rules(am_r2, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if n_am6>0 else []
                al_texts_min = format_atleast_rules(al_final, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if _nlen(al_final)>0 else []
                am_texts_min = format_atmost_rules(am_final, inc, dec, crit_names, score_map=st.session_state.get('score_map')) if _nlen(am_final)>0 else []

                # Compute match matrices for ALL units against maximal and minimal rules
                all_nc = np.hstack([all_crit, np.full((n_units,1), np.nan)])
                _, _, al_m_all_max, am_m_all_max = _cu(all_nc, al_r2,    am_r2,    inc, dec)
                _, _, al_m_all_min, am_m_all_min = _cu(all_nc, al_final, am_final, inc, dec)
                al_units_max = [[unit_names[j] for j in range(n_units) if al_m_all_max[j,i]==1]
                                for i in range(_nlen(al_r2))]
                am_units_max = [[unit_names[j] for j in range(n_units) if am_m_all_max[j,i]==1]
                                for i in range(_nlen(am_r2))]
                al_units_min = [[unit_names[j] for j in range(n_units) if al_m_all_min[j,i]==1]
                                for i in range(_nlen(al_final))]
                am_units_min = [[unit_names[j] for j in range(n_units) if am_m_all_min[j,i]==1]
                                for i in range(_nlen(am_final))]

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
                    ("Rules initially induced", n_al2, n_am2, n_al2+n_am2),
                    ("Greedy selection of rules", _nlen(sel_al), _nlen(sel_am), _nlen(sel_al)+_nlen(sel_am)),
                    ("Maximal set of rules", n_al6, n_am6, n_al6+n_am6),
                    ("Minimal set of rules", _nlen(al_final), _nlen(am_final), _nlen(al_final)+_nlen(am_final)),
                ], columns=["Step","At-least","At-most","Total"])
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

                # Maximal rules expander
                with st.expander(f"📂 Maximal rules ({n_al6} at-least, {n_am6} at-most)"):
                    if al_texts_max:
                        st.markdown("**$\\mathcal{R}^{\\geqslant}$ At-Least (maximal)**")
                        show_rules(al_r2, al_texts_max, units=al_units_max, rule_type="atleast")
                    if am_texts_max:
                        st.markdown("**$\\mathcal{R}^{\\leqslant}$ At-Most (maximal)**")
                        show_rules(am_r2, am_texts_max, units=am_units_max, rule_type="atmost")

                # Minimal rules
                if al_texts_min:
                    st.markdown("### $\\mathcal{R}^{\\geqslant}$ · Minimal At-Least Rules")
                    show_rules(al_final, al_texts_min, units=al_units_min, rule_type="atleast")
                if am_texts_min:
                    st.markdown("### $\\mathcal{R}^{\\leqslant}$ · Minimal At-Most Rules")
                    show_rules(am_final, am_texts_min, units=am_units_min, rule_type="atmost")

                # Download — two persistent buttons
                st.markdown("### 💾 Export rules")
                c1, c2 = st.columns(2)
                with c1:
                    csv_min = rules_to_csv(al_texts_min, am_texts_min, crit_names, inc, dec,
                                           al_rules=al_final, am_rules=am_final,
                                       score_map=st.session_state.get('score_map'))
                    st.download_button("⬇ Minimal rules CSV", csv_min,
                                       file_name="drsa_rules_minimal.csv", mime="text/csv",
                                       key="dl_min_run", use_container_width=True)
                with c2:
                    csv_max = rules_to_csv(al_texts_max, am_texts_max, crit_names, inc, dec,
                                           al_rules=al_r2, am_rules=am_r2,
                                       score_map=st.session_state.get('score_map'))
                    st.download_button("⬇ Maximal rules CSV", csv_max,
                                       file_name="drsa_rules_maximal.csv", mime="text/csv",
                                       key="dl_max_run", use_container_width=True)

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
                            ("Step 6 – Maximal set of rules", *res['step6'], sum(res['step6'])),
                            ("Step 7 – Minimal set of rules", *res['step7'], sum(res['step7'])),
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
                        with st.expander("📂 Maximal rules"):
                            if al_texts_max:
                                st.markdown("**$\\mathcal{R}^{\\geqslant}$ At-Least (maximal)**")
                                show_rules(al_r2, al_texts_max, units=al_units_max, rule_type="atleast")
                            if am_texts_max:
                                st.markdown("**$\\mathcal{R}^{\\leqslant}$ At-Most (maximal)**")
                                show_rules(am_r2, am_texts_max, units=am_units_max, rule_type="atmost")

                    if al_texts:
                        st.markdown("### $\\mathcal{R}^{\\geqslant}$ · Minimal At-Least Rules")
                        show_rules(al_final, al_texts, units=al_units_min, rule_type="atleast")
                    if am_texts:
                        st.markdown("### $\\mathcal{R}^{\\leqslant}$ · Minimal At-Most Rules")
                        show_rules(am_final, am_texts, units=am_units_min, rule_type="atmost")

                    # Persistent download buttons
                    st.markdown("### 💾 Export rules")
                    _inc = st.session_state.get('inc', []); _dec = st.session_state.get('dec', [])
                    _crit = st.session_state.get('crit_names', [])
                    c1, c2 = st.columns(2)
                    with c1:
                        csv_min_p = rules_to_csv(al_texts, am_texts, _crit, _inc, _dec,
                                                 al_rules=al_final, am_rules=am_final,
                                       score_map=st.session_state.get('score_map'))
                        st.download_button("⬇ Minimal rules CSV", csv_min_p,
                                           file_name="drsa_rules_minimal.csv", mime="text/csv",
                                           key="dl_min_prev", use_container_width=True)
                    with c2:
                        csv_max_p = rules_to_csv(al_texts_max, am_texts_max, _crit, _inc, _dec,
                                                 al_rules=al_r2, am_rules=am_r2,
                                       score_map=st.session_state.get('score_map'))
                        st.download_button("⬇ Maximal rules CSV", csv_max_p,
                                           file_name="drsa_rules_maximal.csv", mime="text/csv",
                                           key="dl_max_prev", use_container_width=True)
                else:
                    al_supp  = st.session_state.get('al_supp', [])
                    am_supp  = st.session_state.get('am_supp', [])
                    al_units = st.session_state.get('al_units', [])
                    am_units = st.session_state.get('am_units', [])
                    al_r     = st.session_state.get('al_rules')
                    am_r     = st.session_state.get('am_rules')
                    if al_texts:
                        with st.expander(f"$\\mathcal{{R}}^{{\\geqslant}}$ · At-Least Rules ({_nlen(al_r)})", expanded=False):
                            show_rules(al_r, al_texts, al_supp, al_units, "atleast")
                    if am_texts:
                        with st.expander(f"$\\mathcal{{R}}^{{\\leqslant}}$ · At-Most Rules ({_nlen(am_r)})", expanded=False):
                            show_rules(am_r, am_texts, am_supp, am_units, "atmost")
                    if al_texts or am_texts:
                        st.markdown("### 💾 Export")
                        _inc2 = st.session_state.get('inc',[]); _dec2 = st.session_state.get('dec',[])
                        _crit2 = st.session_state.get('crit_names',[])
                        csv_ind = rules_to_csv(al_texts, am_texts, _crit2, _inc2, _dec2,
                                               al_rules=al_r, am_rules=am_r,
                                       score_map=st.session_state.get('score_map'))
                        st.download_button("⬇ Download rules CSV", csv_ind,
                                           file_name="drsa_rules.csv", mime="text/csv",
                                           key="dl_ind_prev", use_container_width=True)

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

            st.markdown("#### Assignment of all units")

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
                contra  = sm > sp
                _smap3  = st.session_state.get('score_map')
                sm_lbl3 = _fmt_class(sm, _smap3)
                sp_lbl3 = _fmt_class(sp, _smap3)
                assign  = _assign_str(sm, sp, _smap3)
                rows.append({"Unit":name,"s⁻":sm_lbl3,"s⁺":sp_lbl3,"Assignment":assign,
                             "Status":"⚠️ Contradictory" if contra else "✅ OK"})

            df_class = pd.DataFrame(rows)
            df_class_csv = pd.DataFrame({
                "Unit":          [r["Unit"] for r in rows],
                "s-":            [r["s⁻"] for r in rows],
                "s+":            [r["s⁺"] for r in rows],  # already score labels
                "Assignment":    [f"{r['s⁻']}-{r['s⁺']}" if r['s⁻']!=r['s⁺'] else str(r['s⁻']) for r in rows],
                "Contradiction": ["Y" if r["Status"].startswith("⚠️") else "N" for r in rows],
            })
            st.dataframe(df_class, use_container_width=True, height=380)
            n_ok = df_class['Status'].str.contains('OK').sum()
            n_co = df_class['Status'].str.contains('Contr').sum()
            ca,cb = st.columns(2)
            ca.metric("Non-contradictory", n_ok)
            cb.metric("Contradictory", n_co)
            if n_co == 0:
                st.success("All units assigned without contradictions.")
            else:
                st.warning(f"{n_co} unit(s) have contradictory assignments.")

            st.markdown("#### Unit-by-unit explanation")
            sel = st.selectbox("Select unit", unit_names, key="sel_tab3")
            show_explanation(sel, s_minus, s_plus, al_m, am_m,
                             al_texts, am_texts, unit_names.index(sel))

            st.download_button("⬇ Download assignment CSV", df_class_csv.to_csv(index=False),
                               file_name="drsa_assignment.csv", mime="text/csv",
                               key="dl_classif", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 4
    # ══════════════════════════════════════════════════════════════════════════════
    with tab4:
        if st.session_state.get('mode') != 'pipeline':
            st.info("Run the **Full pipeline** first to enable new unit assignment.")
        else:
            st.markdown("#### 🆕 Assign new units")
            st.markdown("Upload new units **without** the class column. The tool tries to handle contradictions.")

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

                if st.button("▶ Assign new units", type="primary", use_container_width=True):
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
                        stat2.success("✅ No contradictions")
                    elif new_res.get('milp_success'):
                        stat2.success(f"✅ Done")
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
                        st.session_state['dec'], st.session_state['crit_names'],
                        score_map=st.session_state.get('score_map')) if _nlen(al_fin)>0 else []
                    am_texts_new = format_atmost_rules(am_fin, st.session_state['inc'],
                        st.session_state['dec'], st.session_state['crit_names'],
                        score_map=st.session_state.get('score_map')) if _nlen(am_fin)>0 else []
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
                st.markdown("#### Assignment results")

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
                    _smap4a = st.session_state.get('score_map')
                    sm_o = _fmt_class(sm_orig, _smap4a); sp_o = _fmt_class(sp_orig, _smap4a)
                    sm_n = _fmt_class(sm_new,  _smap4a); sp_n = _fmt_class(sp_new,  _smap4a)
                    assign_new = _assign_str(sm_new, sp_new, _smap4a)
                    rows_all.append({
                        "Unit": name,
                        "s⁻ (prev)": sm_o, "s⁺ (prev)": sp_o,
                        "s⁻ (new)": sm_n,  "s⁺ (new)": sp_n,
                        "Assignment": assign_new,
                        "Changed": "⚠️ Yes" if changed_flag else "",
                        "_changed": changed_flag,
                        "_sm_new": sm_n, "_sp_new": sp_n,
                        "_assign_csv": f"{sm_n}-{sp_n}" if sm_new!=sp_new else str(sm_n),
                        "_contra": False,
                    })
                # New units A_new
                n_existing = len(all_unit_names)
                for k, name in enumerate(new_names):
                    sm = int(s_minus_f[k]); sp = int(s_plus_f[k])
                    contra = sm > sp
                    assign = f"Class {sm}" if sm==sp else (f"Class {sm}–{sp}" if not contra
                             else f"CONTRADICTORY")
                    sm_prev = int(new_res['step1_s_minus'][k]) if new_res.get('step1_s_minus') is not None else sm
                    sp_prev = int(new_res['step1_s_plus'][k])  if new_res.get('step1_s_plus')  is not None else sp
                    smap4n = st.session_state.get('score_map')
                    sm_lbl4  = _fmt_class(sm, smap4n);    sp_lbl4  = _fmt_class(sp, smap4n)
                    smp_lbl4 = _fmt_class(sm_prev, smap4n); spp_lbl4 = _fmt_class(sp_prev, smap4n)
                    assign_new4 = _assign_str(sm, sp, smap4n)
                    rows_all.append({
                        "Unit": name,
                        "s⁻ (prev)": smp_lbl4, "s⁺ (prev)": spp_lbl4,
                        "s⁻ (new)": sm_lbl4,  "s⁺ (new)": sp_lbl4,
                        "Assignment": assign_new4,
                        "Changed": "🆕 New" if not contra else "⚠️ Contradictory",
                        "_changed": True,
                        "_sm_new": sm_lbl4, "_sp_new": sp_lbl4,
                        "_assign_csv": f"{sm_lbl4}-{sp_lbl4}" if sm!=sp else str(sm_lbl4),
                        "_contra": contra,
                    })

                df_all = pd.DataFrame(rows_all)
                df_display = df_all[[c for c in df_all.columns if not c.startswith('_')]]

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
                cb.metric("New units assigned", n_new_ok)
                cc.metric("New contradictions", len(new_names) - n_new_ok)

                # ── Summary result ─────────────────────────────────────────────
                with st.expander("🔍 Summary result"):
                    changed_names = [all_unit_names[i] for i in changed if i < len(all_unit_names)]
                    st.markdown(f"**Changed assignment of previous units:** "
                                f"{', '.join(changed_names) if changed_names else 'None'}")
                    def _nr(x): return _nlen(x) if x is not None else 0
                    st.markdown(f"**$\\mathcal{{R}}^{{\\geqslant/\\leqslant}}$ maximal rules selected:** "
                                f"{_nr(new_res.get('step7_al_rules'))} at-least, "
                                f"{_nr(new_res.get('step7_am_rules'))} at-most")
                    st.markdown(f"**$\\mathcal{{R}}^{{\\geqslant/\\leqslant}}$ minimal rules selected:** "
                                f"{_nr(new_res.get('step8_al_rules'))} at-least, "
                                f"{_nr(new_res.get('step8_am_rules'))} at-most")


                # ── Maximal rules MILP(7) expander ────────────────────────────
                al7 = new_res.get('step7_al_rules')
                am7 = new_res.get('step7_am_rules')

                # Compute match matrices for new units against MILP(7) and minimal rules
                new_nc = np.hstack([new_matrix, np.full((len(new_matrix),1), np.nan)])
                all_new_names = list(st.session_state['unit_names']) + list(new_names)
                all_units_matrix = np.vstack([st.session_state['matrix_s_minus'][:,:-1], new_matrix])
                all_nc_combined  = np.hstack([all_units_matrix, np.full((len(all_units_matrix),1), np.nan)])

                if _nlen(al7) > 0 or _nlen(am7) > 0:
                    al_texts7 = format_atleast_rules(al7, st.session_state['inc'],
                        st.session_state['dec'], st.session_state['crit_names'],
                        score_map=st.session_state.get('score_map')) if _nlen(al7)>0 else []
                    am_texts7 = format_atmost_rules(am7, st.session_state['inc'],
                        st.session_state['dec'], st.session_state['crit_names'],
                        score_map=st.session_state.get('score_map')) if _nlen(am7)>0 else []
                    _, _, al_m7_all, am_m7_all = classify_units(
                        all_nc_combined, al7 if _nlen(al7)>0 else np.empty((0,1)),
                        am7 if _nlen(am7)>0 else np.empty((0,1)),
                        st.session_state['inc'], st.session_state['dec'])
                    al_units7 = [[all_new_names[j] for j in range(len(all_new_names)) if al_m7_all[j,i]==1]
                                 for i in range(_nlen(al7))]
                    am_units7 = [[all_new_names[j] for j in range(len(all_new_names)) if am_m7_all[j,i]==1]
                                 for i in range(_nlen(am7))]
                    with st.expander(f"📂 Maximal rules selected ({_nlen(al7)} at-least, {_nlen(am7)} at-most)"
                                     f"({_nlen(al7)} at-least, {_nlen(am7)} at-most)"):
                        if al_texts7:
                            st.markdown("**$\\mathcal{R}^{\\geqslant}$ At-Least:**")
                            show_rules(al7, al_texts7, units=al_units7, rule_type="atleast")
                        if am_texts7:
                            st.markdown("**$\\mathcal{R}^{\\leqslant}$ At-Most:**")
                            show_rules(am7, am_texts7, units=am_units7, rule_type="atmost")

                # ── Minimal rules for A ∪ A_new ────────────────────────────────
                if al_texts_new or am_texts_new:
                    _, _, al_m_fin, am_m_fin = classify_units(
                        all_nc_combined, al_fin if _nlen(al_fin)>0 else np.empty((0,1)),
                        am_fin if _nlen(am_fin)>0 else np.empty((0,1)),
                        st.session_state['inc'], st.session_state['dec'])
                    al_units_fin = [[all_new_names[j] for j in range(len(all_new_names)) if al_m_fin[j,i]==1]
                                    for i in range(_nlen(al_fin))]
                    am_units_fin = [[all_new_names[j] for j in range(len(all_new_names)) if am_m_fin[j,i]==1]
                                    for i in range(_nlen(am_fin))]
                    with st.expander(f"📂 Minimal rules selected ({_nlen(al_fin)} at-least, {_nlen(am_fin)} at-most)"
                                     f"({_nlen(al_fin)} at-least, {_nlen(am_fin)} at-most)"):
                        if al_texts_new:
                            st.markdown("**$\\mathcal{R}^{\\geqslant}$ At-Least:**")
                            show_rules(al_fin, al_texts_new, units=al_units_fin, rule_type="atleast")
                        if am_texts_new:
                            st.markdown("**$\\mathcal{R}^{\\leqslant}$ At-Most:**")
                            show_rules(am_fin, am_texts_new, units=am_units_fin, rule_type="atmost")

                # ── Unit by unit explanation for ALL units (A + A_new) ─────
                st.markdown("#### Unit-by-unit explanation")
                all_names_expl = list(all_unit_names) + list(new_names)
                all_crit_expl  = np.vstack([
                    st.session_state['matrix_s_minus'][:,:-1],
                    new_matrix
                ])
                all_nc_expl = np.hstack([all_crit_expl, np.full((len(all_crit_expl),1), np.nan)])
                s_minus_all, s_plus_all, al_m_all, am_m_all = classify_units(
                    all_nc_expl, al_fin, am_fin,
                    st.session_state['inc'], st.session_state['dec'])
                sel_expl = st.selectbox("Select unit", all_names_expl, key="sel_new")
                idx_expl = all_names_expl.index(sel_expl)
                show_explanation(sel_expl, s_minus_all, s_plus_all,
                                 al_m_all, am_m_all, al_texts_new, am_texts_new, idx_expl)

                st.markdown("### 💾 Export")
                _inc_n = st.session_state['inc']; _dec_n = st.session_state['dec']
                _crit_n = st.session_state['crit_names']
                c1, c2 = st.columns(2)
                with c1:
                    csv_new_min = rules_to_csv(al_texts_new, am_texts_new, _crit_n, _inc_n, _dec_n,
                                               al_rules=al_fin, am_rules=am_fin,
                                       score_map=st.session_state.get('score_map'))
                    st.download_button("⬇ Minimal rules CSV", csv_new_min,
                                       file_name="drsa_newunits_rules_minimal.csv",
                                       mime="text/csv", key="dl_new_min", use_container_width=True)
                with c2:
                    _al7n = new_res.get('step7_al_rules'); _am7n = new_res.get('step7_am_rules')
                    _al7_txtn = format_atleast_rules(_al7n, _inc_n, _dec_n, _crit_n,
                        score_map=st.session_state.get('score_map')) if _nlen(_al7n)>0 else []
                    _am7_txtn = format_atmost_rules(_am7n, _inc_n, _dec_n, _crit_n,
                        score_map=st.session_state.get('score_map')) if _nlen(_am7n)>0 else []
                    csv_new_max = rules_to_csv(_al7_txtn, _am7_txtn, _crit_n, _inc_n, _dec_n,
                                               al_rules=_al7n, am_rules=_am7n,
                                       score_map=st.session_state.get('score_map'))
                    st.download_button("⬇ Maximal rules CSV", csv_new_max,
                                       file_name="drsa_newunits_rules_maximal.csv",
                                       mime="text/csv", key="dl_new_max", use_container_width=True)

                # Build clean CSV for download
                df_new_csv = pd.DataFrame({
                    "Unit":          df_all["Unit"],
                    "s- (prev)":     df_all["s⁻ (prev)"].apply(lambda x: "-" if x == "—" else x),
                    "s+ (prev)":     df_all["s⁺ (prev)"].apply(lambda x: "-" if x == "—" else x),
                    "s-":            df_all["s⁻ (new)"],
                    "s+":            df_all["s⁺ (new)"],
                    "Assignment":    df_all.apply(lambda r:
                                         f"{r['s⁻ (new)']}-{r['s⁺ (new)']}"
                                         if r['s⁻ (new)'] != r['s⁺ (new)']
                                         else str(r['s⁻ (new)']), axis=1),
                    "Contradiction": df_all.apply(lambda r:
                                         "Y" if isinstance(r['s⁻ (new)'], (int,float))
                                         and isinstance(r['s⁺ (new)'], (int,float))
                                         and int(r['s⁻ (new)']) > int(r['s⁺ (new)'])
                                         else "N", axis=1),
                    "Changed":       df_all["Changed"].apply(lambda x:
                                         "Y" if x in ["⚠️ Yes","🆕 New","⚠️ Contradictory"] else "N"),
                })
                st.download_button("⬇ Download assignment CSV",
                                   df_new_csv.to_csv(index=False),
                                   file_name="drsa_new_units.csv",
                                   mime="text/csv", key="dl_new_cl", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 5 — Apply Rules
    # ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 📂 Apply saved rules")

    # ── File uploader (always visible) ─────────────────────────────────────────
    col_r, col_s = st.columns([3, 1])
    with col_r:
        rules_file = st.file_uploader("Upload rules CSV", type=["csv","txt"], key="rules_file")
    with col_s:
        sep_r = st.selectbox("Separator", [",",";","\\t"," "], key="sep_rules")
        sep_r_act = "\t" if sep_r=="\\t" else sep_r

    # ── Welcome (only when no file loaded) ─────────────────────────────────────
    if rules_file is None:
        c1t5, c2t5, c3t5 = st.columns(3)
        with c1t5:
            st.markdown("""
**Workflow**
1. Load a rules CSV file
2. Load an units file (optional)
3. Visualize rules with matching units
4. Inspect assignment
""")
        with c2t5:
            st.markdown("""
**File format**
- CSV or TXT
- Two files:
  - **Rules**: `#directions`, `#mode` rows + type/class/criteria
  - **Units** (optional): name column + criteria (no class)
""")
        with c3t5:
            st.markdown("""
**Rules example**
```
#directions,increasing,increasing,decreasing
#mode,class
type,class,g1,g2,g3
at-least,2,4.5,,
at-most,1,,,2.5
```
**Units example**
```
Name,g1,g2,g3
x1,5.0,3.5,4.0
x2,2.0,4.5,1.5
```
""")
        sample_rules_class = (
            "#directions,increasing,increasing,decreasing\n"
            "#mode,class\n"
            "type,class,g1,g2,g3\n"
            "at-least,2,4.5,,\n"
            "at-least,3,,5.0,\n"
            "at-most,1,,,2.5\n"
            "at-most,2,,4.0,\n"
        )
        sample_rules_score = (
            "#directions,increasing,increasing,decreasing\n"
            "#mode,score\n"
            "type,class,g1,g2,g3\n"
            "at-least,16.67,4.5,,\n"
            "at-least,41.67,,5.0,\n"
            "at-most,0,,,2.5\n"
            "at-most,16.67,,4.0,\n"
        )
        sample_alts = (
            "Name,g1,g2,g3\n"
            "x1,5.0,3.5,4.0\n"
            "x2,2.0,4.5,1.5\n"
            "x3,6.0,6.0,5.5\n"
            "x4,3.5,2.0,3.0\n"
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.download_button("⬇ Sample rules (Class)", sample_rules_class,
                               file_name="sample_rules_class.csv", mime="text/csv",
                               key="dl_sample_rules_class", use_container_width=True)
        with sc2:
            st.download_button("⬇ Sample rules (Score)", sample_rules_score,
                               file_name="sample_rules_score.csv", mime="text/csv",
                               key="dl_sample_rules_score", use_container_width=True)
        with sc3:
            st.download_button("⬇ Sample units", sample_alts,
                               file_name="sample_units.csv", mime="text/csv",
                               key="dl_sample_alts", use_container_width=True)


    if rules_file is not None:
        try:
            raw_lines = rules_file.read().decode("utf-8").splitlines()
        except Exception as e:
            st.error(f"Could not read file: {e}"); st.stop()

        # Parse #directions line
        inc5 = []; dec5 = []; crit_names5 = []
        directions_line = None
        data_lines = []
        for line in raw_lines:
            if line.startswith("#directions"):
                directions_line = line
            else:
                data_lines.append(line)

        if directions_line is None:
            st.error("Missing #directions line in rules file."); st.stop()

        dir_parts = directions_line.split(sep_r_act)[1:]

        # Parse #mode
        mode_line = next((l for l in raw_lines if l.startswith('#mode')), None)
        file_mode = 'class'
        file_score_map = None
        file_score_map_inv = None
        if mode_line:
            file_mode = mode_line.split(sep_r_act)[1].strip()

        # Filter out metadata lines for parsing
        data_lines = [l for l in raw_lines if not l.startswith('#')]
        # Build score map from class column values if mode=score (after parsing)
        import io
        df_rules_raw = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=sep_r_act)
        crit_names5 = [c for c in df_rules_raw.columns if c not in ['type','class']]
        # Build score_map from unique class values if mode=score
        if file_mode == 'score':
            raw_vals = sorted(set(float(v) for v in df_rules_raw['class'].dropna()))
            file_score_map     = {i+1: v for i, v in enumerate(raw_vals)}
            file_score_map_inv = {v: i+1 for i, v in enumerate(raw_vals)}

        for i, d in enumerate(dir_parts):
            if i < len(crit_names5):
                if d.strip() == 'increasing':
                    inc5.append(i)
                else:
                    dec5.append(i)

        # Parse rules into arrays
        al_rules5 = []; am_rules5 = []
        n_crit5 = len(crit_names5)
        rule_width = n_crit5 * 2  # crit_idx, threshold pairs

        for _, row in df_rules_raw.iterrows():
            rtype = str(row.get('type','')).strip()
            rclass_raw = float(row.get('class', 0))
            # Convert score back to class index if needed
            if file_score_map_inv and rclass_raw in file_score_map_inv:
                rclass = int(file_score_map_inv[rclass_raw])
            else:
                rclass = int(rclass_raw)
            rule_vec = []
            for ci, cname in enumerate(crit_names5):
                val = row.get(cname, '')
                if pd.isna(val) or str(val).strip() == '':
                    rule_vec.extend([0, 0])
                else:
                    v = float(val)
                    # Re-negate to restore internal format:
                    # at-least: decreasing criteria are stored negated
                    # at-most:  increasing criteria are stored negated
                    if rtype == 'at-least' and ci in dec5:
                        v = -v
                    elif rtype == 'at-most' and ci in inc5:
                        v = -v
                    rule_vec.extend([ci + 1, v])
            rule_vec.append(rclass)
            if rtype == 'at-least':
                al_rules5.append(rule_vec)
            elif rtype == 'at-most':
                am_rules5.append(rule_vec)

        al_rules5 = np.array(al_rules5) if al_rules5 else np.empty((0, rule_width+1))
        am_rules5 = np.array(am_rules5) if am_rules5 else np.empty((0, rule_width+1))

        n_al5 = len(al_rules5); n_am5 = len(am_rules5)
        al_texts5 = format_atleast_rules(al_rules5, inc5, dec5, crit_names5, score_map=file_score_map) if n_al5>0 else []
        am_texts5 = format_atmost_rules(am_rules5, inc5, dec5, crit_names5, score_map=file_score_map) if n_am5>0 else []

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("At-least rules", n_al5)
        mc2.metric("At-most rules", n_am5)
        mc3.metric("Criteria", n_crit5)

        st.markdown(f"**Criteria directions:** " +
                    ", ".join(f"{crit_names5[i]} ({'↑' if i in inc5 else '↓'})"
                              for i in range(len(crit_names5))))

        # ── Load units (optional) ───────────────────────────────────
        st.markdown("---")
        col_a, col_sa = st.columns([3, 1])
        with col_a:
            alt_file5 = st.file_uploader("Upload units (optional, without class column). Note: column names must correspond to criteria names of rules",
                                          type=["csv","txt"], key="alt_file5")
        with col_sa:
            sep_a5 = st.selectbox("Separator", [",",";","\t"," "], key="sep_alt5")
            sep_a5_act = "\t" if sep_a5=="\t" else sep_a5

        alt_names5 = None; alt_matrix5 = None
        if alt_file5 is not None:
            try:
                df_alt5_raw = pd.read_csv(alt_file5, sep=sep_a5_act, engine="python")
            except Exception as e:
                st.error(f"Could not read units file: {e}"); st.stop()

            df_alt5 = df_alt5_raw.copy()
            fc5 = df_alt5.columns[0]
            if pd.to_numeric(df_alt5[fc5], errors='coerce').isna().sum() > len(df_alt5)*0.5:
                alt_names5 = df_alt5[fc5].astype(str).tolist()
                df_alt5 = df_alt5.drop(columns=[fc5])
            df_alt5 = df_alt5.apply(pd.to_numeric, errors='coerce')
            # Drop last column if it looks like a class column (all integers 1..p)
            # Keep only criteria columns that match the rules (by name)
            common_cols = [c for c in crit_names5 if c in df_alt5.columns]
            if len(common_cols) < n_crit5:
                st.warning(f"Units file has {len(common_cols)}/{n_crit5} matching criteria columns.")
            df_alt5 = df_alt5[[c for c in crit_names5 if c in df_alt5.columns]]
            if alt_names5 is None:
                alt_names5 = [f'a{i+1}' for i in range(len(df_alt5))]
            alt_matrix5 = df_alt5.values.astype(float)
            st.dataframe(df_alt5_raw, use_container_width=True, height=200)

        # ── Compute match units per rule ────────────────────────────────────
        al_units5 = [[] for _ in range(n_al5)]
        am_units5 = [[] for _ in range(n_am5)]
        s_minus5 = s_plus5 = al_m5 = am_m5 = None

        if alt_matrix5 is not None and n_crit5 == alt_matrix5.shape[1]:
            p5 = 3  # default, will be overridden by rules
            if n_al5 > 0:
                p5 = max(p5, int(np.max(al_rules5[:, -1])))
            if n_am5 > 0:
                p5 = max(p5, int(np.max(am_rules5[:, -1])))
            alt_nc5 = np.hstack([alt_matrix5, np.full((len(alt_matrix5),1), np.nan)])
            s_minus5, s_plus5, al_m5, am_m5 = classify_units(
                alt_nc5, al_rules5, am_rules5, inc5, dec5)
            al_units5 = [[alt_names5[j] for j in range(len(alt_names5)) if al_m5[j,i]==1]
                         for i in range(n_al5)]
            am_units5 = [[alt_names5[j] for j in range(len(alt_names5)) if am_m5[j,i]==1]
                         for i in range(n_am5)]

        # ── Display rules ───────────────────────────────────────────────────
        if n_al5 > 0:
            with st.expander(f"$\\mathcal{{R}}^{{\\geqslant}}$ · At-Least Rules ({n_al5})", expanded=False):
                show_rules(al_rules5, al_texts5,
                           units=al_units5 if alt_matrix5 is not None else None,
                           rule_type="atleast")
        if n_am5 > 0:
            with st.expander(f"$\\mathcal{{R}}^{{\\leqslant}}$ · At-Most Rules ({n_am5})", expanded=False):
                show_rules(am_rules5, am_texts5,
                           units=am_units5 if alt_matrix5 is not None else None,
                           rule_type="atmost")

        # ── Classification ──────────────────────────────────────────────────
        if alt_matrix5 is not None and s_minus5 is not None:
            st.markdown("---")
            st.markdown("#### Assignment")
            rows5 = []
            for i, name in enumerate(alt_names5):
                sm, sp = int(s_minus5[i]), int(s_plus5[i])
                contra = sm > sp
                sm_lbl5 = _fmt_class(sm, file_score_map); sp_lbl5 = _fmt_class(sp, file_score_map)
                assign_disp5 = _assign_str(sm, sp, file_score_map)
                assign_csv5  = f"{sm_lbl5}-{sp_lbl5}" if sm!=sp else str(sm_lbl5)
                rows5.append({"Unit":name,
                               "s⁻":sm_lbl5,"s⁺":sp_lbl5,
                               "Assignment":assign_disp5,
                               "Status":"⚠️ Contradictory" if contra else "✅ OK",
                               "_sm":sm_lbl5,"_sp":sp_lbl5,"_assign_csv":assign_csv5,"_contra":contra})
            df_class5 = pd.DataFrame(rows5)
            df_class5_disp = df_class5[["Unit","s⁻","s⁺","Assignment","Status"]]
            df_class5_csv  = pd.DataFrame({
                "Unit":         df_class5["Unit"],
                "s-":           df_class5["_sm"],
                "s+":           df_class5["_sp"],
                "Assignment":   df_class5["_assign_csv"],
                "Contradiction":df_class5["_contra"].map({True:"Y", False:"N"}),
            })
            st.dataframe(df_class5_disp, use_container_width=True, height=350)

            n_ok5 = df_class5_disp['Status'].str.contains('OK').sum()
            n_co5 = df_class5_disp['Status'].str.contains('Contr').sum()
            ca5, cb5 = st.columns(2)
            ca5.metric("Non-contradictory", n_ok5)
            cb5.metric("Contradictory", n_co5)

            st.markdown("#### Unit-by-unit explanation")
            sel5 = st.selectbox("Select unit", alt_names5, key="sel_tab5")
            show_explanation(sel5, s_minus5, s_plus5, al_m5, am_m5,
                             al_texts5, am_texts5, alt_names5.index(sel5))

            st.download_button("⬇ Download assignment CSV",
                               df_class5_csv.to_csv(index=False),
                               file_name="drsa_applied_assignment.csv",
                               mime="text/csv", key="dl_apply_class",
                               use_container_width=True)

        # ── New units section ───────────────────────────────────────────────
        if alt_matrix5 is not None and s_minus5 is not None:
            st.markdown("---")
            st.markdown("#### 🆕 Assign new units")
            st.markdown("Upload new units to classify them using the loaded rules (MILP 6).")

            col_nu, col_sep_nu = st.columns([3,1])
            with col_nu:
                new_file5 = st.file_uploader("Upload new units (without class column)",
                                              type=["csv","txt"], key="new_file5")
            with col_sep_nu:
                sep_nu5 = st.selectbox("Separator", [",",";","\t"," "], key="sep_nu5")
                sep_nu5_act = "\t" if sep_nu5=="\t" else sep_nu5

            if new_file5 is not None:
                try:
                    df_new5_raw = pd.read_csv(new_file5, sep=sep_nu5_act, engine="python")
                except Exception as e:
                    st.error(f"Could not read file: {e}"); st.stop()

                # Parse new units
                df_new5 = df_new5_raw.copy()
                fc_new5 = df_new5.columns[0]
                if pd.to_numeric(df_new5[fc_new5], errors='coerce').isna().sum() > len(df_new5)*0.5:
                    new_names5 = df_new5[fc_new5].astype(str).tolist()
                    df_new5 = df_new5.drop(columns=[fc_new5])
                else:
                    new_names5 = [f'x{i+1}' for i in range(len(df_new5))]
                df_new5 = df_new5.apply(pd.to_numeric, errors='coerce')
                df_new5 = df_new5[[c for c in crit_names5 if c in df_new5.columns]]
                new_matrix5 = df_new5.values.astype(float)

                st.dataframe(df_new5_raw, use_container_width=True, height=150)

                if st.button("▶ Assign new units", type="primary",
                             use_container_width=True, key="btn_assign_new5"):
                    from drsa.core.new_units import classify_new_units

                    # Build combined matrix: previous units + their classification
                    # s_minus5 and s_plus5 are already computed from eq.(4)
                    n_prev = len(alt_names5)
                    n_new  = len(new_names5)

                    from drsa.core.classifier import classify_units as _cu5
                    mat_sm5 = np.column_stack([alt_matrix5, s_minus5])
                    mat_sp5 = np.column_stack([alt_matrix5, s_plus5])
                    prev_nc5 = np.hstack([alt_matrix5, np.full((len(alt_matrix5),1), np.nan)])
                    _, _, match_al5, match_am5 = _cu5(prev_nc5, al_rules5, am_rules5, inc5, dec5)

                    try:
                        new_res5 = classify_new_units(
                            new_matrix5, mat_sm5, mat_sp5,
                            al_rules5, match_al5,
                            am_rules5, match_am5,
                            inc5, dec5)
                        s_minus_new5 = new_res5.get('s_minus_final',
                                       np.full(n_new, 1))
                        s_plus_new5  = new_res5.get('s_plus_final',
                                       np.full(n_new, 1))
                        al_fin5 = new_res5.get('al_rules_final', al_rules5)
                        am_fin5 = new_res5.get('am_rules_final', am_rules5)

                        st.session_state['new_res5']       = new_res5
                        st.session_state['new_names5']     = new_names5
                        st.session_state['new_matrix5']    = new_matrix5
                        st.session_state['s_minus_new5']   = s_minus_new5
                        st.session_state['s_plus_new5']    = s_plus_new5
                        st.session_state['al_fin5']        = al_fin5
                        st.session_state['am_fin5']        = am_fin5
                        st.session_state['alt_names5_prev'] = alt_names5
                        st.session_state['s_minus5_prev']   = s_minus5
                        st.session_state['s_plus5_prev']    = s_plus5
                        st.success("✅ Done")
                    except Exception as e:
                        st.error(f"Error: {e}")

                # Show results if available
                if st.session_state.get('new_res5') is not None:
                    new_res5     = st.session_state['new_res5']
                    new_names5   = st.session_state['new_names5']
                    new_matrix5  = st.session_state['new_matrix5']
                    s_minus_new5 = st.session_state['s_minus_new5']
                    s_plus_new5  = st.session_state['s_plus_new5']
                    al_fin5      = st.session_state['al_fin5']
                    am_fin5      = st.session_state['am_fin5']
                    alt_names5_p = st.session_state['alt_names5_prev']
                    s_minus5_p   = st.session_state['s_minus5_prev']
                    s_plus5_p    = st.session_state['s_plus5_prev']

                    st.markdown("#### Assignment results")
                    rows_new5 = []
                    all_names_new5 = list(alt_names5_p) + list(new_names5)

                    # Previous units — check if assignment changed
                    al_fin5 = new_res5.get('al_rules_final', al_rules5)
                    am_fin5 = new_res5.get('am_rules_final', am_rules5)
                    step1_sm5 = new_res5.get('step1_s_minus', None)
                    step1_sp5 = new_res5.get('step1_s_plus',  None)
                    changed5  = new_res5.get('changed_units', [])

                    for i, name in enumerate(alt_names5_p):
                         sm_o = int(s_minus5_p[i]); sp_o = int(s_plus5_p[i])
                         # Recompute with new rules
                         all_m5_tmp = np.vstack([alt_matrix5, new_matrix5])
                         all_nc5_tmp = np.hstack([all_m5_tmp, np.full((len(all_m5_tmp),1), np.nan)])
                         sm_new5_all, sp_new5_all, _, _ = classify_units(
                             all_nc5_tmp, al_fin5, am_fin5, inc5, dec5)
                         sm_n = int(sm_new5_all[i]); sp_n = int(sp_new5_all[i])
                         changed_flag = (sm_n != sm_o or sp_n != sp_o)
                         rows_new5.append({
                             "Unit": name,
                             "s⁻ (prev)": _fmt_class(sm_o, file_score_map),
                             "s⁺ (prev)": _fmt_class(sp_o, file_score_map),
                             "s⁻ (new)":  _fmt_class(sm_n, file_score_map),
                             "s⁺ (new)":  _fmt_class(sp_n, file_score_map),
                             "Assignment": _assign_str(sm_n, sp_n, file_score_map),
                             "Changed": "⚠️ Yes" if changed_flag else "",
                             "_changed": changed_flag,
                         })

                    # New units
                    for k, name in enumerate(new_names5):
                         sm1 = int(step1_sm5[k]) if step1_sm5 is not None else int(s_minus_new5[k])
                         sp1 = int(step1_sp5[k]) if step1_sp5 is not None else int(s_plus_new5[k])
                         sm  = int(s_minus_new5[k]); sp = int(s_plus_new5[k])
                         contra = sm > sp
                         rows_new5.append({
                             "Unit": name,
                             "s⁻ (prev)": _fmt_class(sm1, file_score_map),
                             "s⁺ (prev)": _fmt_class(sp1, file_score_map),
                             "s⁻ (new)":  _fmt_class(sm,  file_score_map),
                             "s⁺ (new)":  _fmt_class(sp,  file_score_map),
                             "Assignment": _assign_str(sm, sp, file_score_map),
                             "Changed": "🆕 New" if not contra else "⚠️ Contradictory",
                             "_changed": True,
                         })

                    df_n5 = pd.DataFrame(rows_new5)
                    df_n5_show = df_n5[[c for c in df_n5.columns if not c.startswith('_')]]
                    def _hl5(row):
                         return (['background-color: #fef9c3']*len(row)
                                 if rows_new5[row.name]['_changed'] else ['']*len(row))
                    st.dataframe(df_n5_show.style.apply(_hl5, axis=1),
                                  use_container_width=True, height=400)

                    # ── Summary ───────────────────────────────────────────────
                    n_ch5   = sum(1 for r in rows_new5 if r["Changed"]=="⚠️ Yes")
                    n_ok5n  = sum(1 for r in rows_new5[-len(new_names5):]
                                   if "Contr" not in r["Assignment"])
                    n_co5n  = len(new_names5) - n_ok5n
                    ca5, cb5, cc5 = st.columns(3)
                    ca5.metric("Changed in A", n_ch5)
                    cb5.metric("New units assigned", n_ok5n)
                    cc5.metric("New contradictions", n_co5n)
                    ch_names5 = [r["Unit"] for r in rows_new5 if r["Changed"]=="⚠️ Yes"]
                    if ch_names5:
                         st.markdown(f"**Changed assignment of previous units:** "
                                     f"{', '.join(ch_names5)}")

                    # ── Maximal rules expander ────────────────────────────────
                    al7_5 = new_res5.get('step7_al_rules')
                    am7_5 = new_res5.get('step7_am_rules')
                    all_m5v = np.vstack([alt_matrix5, new_matrix5])
                    all_nc5v = np.hstack([all_m5v, np.full((len(all_m5v),1), np.nan)])
                    if _nlen(al7_5)>0 or _nlen(am7_5)>0:
                         al_t7_5 = format_atleast_rules(al7_5, inc5, dec5, crit_names5,
                             score_map=file_score_map) if _nlen(al7_5)>0 else []
                         am_t7_5 = format_atmost_rules(am7_5, inc5, dec5, crit_names5,
                             score_map=file_score_map) if _nlen(am7_5)>0 else []
                         _, _, al_m7_5, am_m7_5 = classify_units(
                             all_nc5v,
                             al7_5 if _nlen(al7_5)>0 else np.empty((0,1)),
                             am7_5 if _nlen(am7_5)>0 else np.empty((0,1)),
                             inc5, dec5)
                         al_u7_5 = [[all_names_new5[j] for j in range(len(all_names_new5))
                                     if al_m7_5[j,i]==1] for i in range(_nlen(al7_5))]
                         am_u7_5 = [[all_names_new5[j] for j in range(len(all_names_new5))
                                     if am_m7_5[j,i]==1] for i in range(_nlen(am7_5))]
                         with st.expander(f"📂 Maximal rules selected ({_nlen(al7_5)} at-least, {_nlen(am7_5)} at-most)", expanded=False):
                             if al_t7_5:
                                 st.markdown("**R≥ At-Least:**")
                                 show_rules(al7_5, al_t7_5, units=al_u7_5, rule_type="atleast")
                             if am_t7_5:
                                 st.markdown("**R≤ At-Most:**")
                                 show_rules(am7_5, am_t7_5, units=am_u7_5, rule_type="atmost")

                    # ── Minimal rules expander ────────────────────────────────
                    al_tf5 = format_atleast_rules(al_fin5, inc5, dec5, crit_names5,
                         score_map=file_score_map) if _nlen(al_fin5)>0 else []
                    am_tf5 = format_atmost_rules(am_fin5, inc5, dec5, crit_names5,
                         score_map=file_score_map) if _nlen(am_fin5)>0 else []
                    _, _, al_mf5, am_mf5 = classify_units(
                         all_nc5v, al_fin5, am_fin5, inc5, dec5)
                    al_uf5 = [[all_names_new5[j] for j in range(len(all_names_new5))
                                if al_mf5[j,i]==1] for i in range(_nlen(al_fin5))]
                    am_uf5 = [[all_names_new5[j] for j in range(len(all_names_new5))
                                if am_mf5[j,i]==1] for i in range(_nlen(am_fin5))]
                    with st.expander(f"📂 Minimal rules selected ({_nlen(al_fin5)} at-least, {_nlen(am_fin5)} at-most)", expanded=False):
                         if al_tf5:
                             st.markdown("**R≥ At-Least:**")
                             show_rules(al_fin5, al_tf5, units=al_uf5, rule_type="atleast")
                         if am_tf5:
                             st.markdown("**R≤ At-Most:**")
                             show_rules(am_fin5, am_tf5, units=am_uf5, rule_type="atmost")

                    # ── Unit-by-unit explanation ──────────────────────────────
                    st.markdown("#### Unit-by-unit explanation")
                    sm_a5, sp_a5, al_m_a5, am_m_a5 = classify_units(
                         all_nc5v, al_fin5, am_fin5, inc5, dec5)
                    sel5n = st.selectbox("Select unit", all_names_new5, key="sel_new5")
                    show_explanation(sel5n, sm_a5, sp_a5, al_m_a5, am_m_a5,
                                      al_tf5, am_tf5, all_names_new5.index(sel5n))

                    # ── Downloads ─────────────────────────────────────────────
                    st.markdown("### 💾 Export")
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                         csv_max5 = rules_to_csv(al_t7_5 if _nlen(al7_5)>0 else [],
                             am_t7_5 if _nlen(am7_5)>0 else [],
                             crit_names5, inc5, dec5,
                             al_rules=al7_5, am_rules=am7_5,
                             score_map=file_score_map)
                         st.download_button("⬇ Maximal rules CSV", csv_max5,
                             file_name="drsa_new5_maximal.csv", mime="text/csv",
                             key="dl_new5_max", use_container_width=True)
                    with dc2:
                         csv_min5 = rules_to_csv(al_tf5, am_tf5, crit_names5, inc5, dec5,
                             al_rules=al_fin5, am_rules=am_fin5,
                             score_map=file_score_map)
                         st.download_button("⬇ Minimal rules CSV", csv_min5,
                             file_name="drsa_new5_minimal.csv", mime="text/csv",
                             key="dl_new5_min", use_container_width=True)
                    with dc3:
                         df_n5_csv = pd.DataFrame({
                             "Unit":       [r["Unit"] for r in rows_new5],
                             "s- (prev)":  [r["s⁻ (prev)"] for r in rows_new5],
                             "s+ (prev)":  [r["s⁺ (prev)"] for r in rows_new5],
                             "s-":         [r["s⁻ (new)"] for r in rows_new5],
                             "s+":         [r["s⁺ (new)"] for r in rows_new5],
                             "Assignment": [r["Assignment"] for r in rows_new5],
                             "Changed":    [r["Changed"] for r in rows_new5],
                         })
                         st.download_button("⬇ Assignment CSV", df_n5_csv.to_csv(index=False),
                             file_name="drsa_new5_assignment.csv", mime="text/csv",
                             key="dl_new5_cl", use_container_width=True)

