"""
DRSA Rule Induction Tool - Streamlit Interface
Implements Algorithms 1, 2, 4 from Corrente et al. (Omega, 2026)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Path setup
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_HERE)
_ROOT        = os.path.dirname(_PACKAGE_DIR)
sys.path.insert(0, _ROOT)

from drsa import (
    induce_atleast_rules,
    induce_atmost_rules,
    format_atleast_rules,
    format_atmost_rules,
    compute_relative_support,
    get_supporting_units,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DRSA Rule Induction",
    page_icon="⚖️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }
.rule-box {
    background: #f8f9fa; border-left: 4px solid #1a1a2e;
    padding: 12px 16px; margin: 6px 0; border-radius: 0 6px 6px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; line-height: 1.6;
}
.rule-box.atleast { border-left-color: #2563eb; }
.rule-box.atmost  { border-left-color: #dc2626; }
.metric-card {
    background: #1a1a2e; color: white; border-radius: 8px;
    padding: 16px 20px; text-align: center;
}
.metric-card .value { font-size: 2.2rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.metric-card .label { font-size: 0.8rem; opacity: 0.7; margin-top: 2px; }
.info-banner {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 14px 18px; font-size: 0.88rem; color: #1e3a5f; margin-bottom: 18px;
}
.tag {
    display: inline-block; background: #dbeafe; color: #1e40af;
    border-radius: 4px; padding: 1px 8px; font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace; margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚖️ DRSA Rule Induction")
st.markdown("""
<div class="info-banner">
Dominance-based Rough Set Approach for composite indicator explanation.<br>
Implements <b>Algorithms 1, 2 and 4</b> from Corrente, Greco, Słowiński, Zappalà —
<i>"An explainable and interpretable composite indicator based on decision rules"</i>,
<b>Omega 142</b> (2026), 103513.
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Input")
    uploaded = st.file_uploader(
        "Upload CSV or TXT file",
        type=["csv", "txt"],
        help="Last column = class label. Optional first column = alternative names."
    )
    sep = st.selectbox("Column separator", [",", ";", "\\t", " "], index=0)
    sep_actual = "\t" if sep == "\\t" else sep

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    min_conf = st.slider("Minimum confidence (c)", 0.0, 1.0, 1.0, 0.05,
                         help="c=1.0 → exact rules only.")
    handle_missing = st.checkbox("Handle missing values (Algorithm 4)", False)
    run_atleast = st.checkbox("Induce at-least rules (R≥)", value=True)
    run_atmost  = st.checkbox("Induce at-most rules (R≤)",  value=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af'>"
        "Corrente et al. (2026)<br>Omega 142, 103513</div>",
        unsafe_allow_html=True
    )

# ── Welcome screen ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("#### Getting started")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**File format**
- CSV or TXT
- Optional first column: alternative names (text)
- Criteria columns (numeric)
- Last column: class label (integer)
- Missing values: empty cell or `NaN`
""")
    with c2:
        st.markdown("""
**Example with alternative names**
```
Alternative,g1,g2,g3,class
A1,4,3,2,1
A2,5,4,3,2
A3,7,6,5,3
```
""")
    with c3:
        st.markdown("""
**Output**
- At-least rules (R≥)
- At-most rules (R≤)
- Relative support per rule
- Supporting alternatives
- Download TXT / CSV
""")
    sample = pd.DataFrame({
        'Alternative': ['A1','A2','A3','A4','A5','A6'],
        'g1': [4,5,7,3,6,8], 'g2': [3,4,6,2,5,7],
        'g3': [2,3,5,1,4,6], 'class': [1,2,3,1,2,3]
    })
    st.download_button("⬇ Download sample CSV", sample.to_csv(index=False),
                       file_name="drsa_sample.csv", mime="text/csv")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded, sep=sep_actual, engine="python")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Detect alternative names column
unit_names = None
df = df_raw.copy()
first_col = df.columns[0]
if pd.to_numeric(df[first_col], errors='coerce').isna().sum() > len(df) * 0.5:
    unit_names = df[first_col].astype(str).tolist()
    df = df.drop(columns=[first_col])
    st.info(f"Column **'{first_col}'** detected as alternative names.")

df = df.apply(pd.to_numeric, errors='coerce')
if unit_names is None:
    unit_names = [f'a{i+1}' for i in range(len(df))]

n_units, n_cols = df.shape
n_criteria = n_cols - 1
crit_names = list(df.columns)[:-1]

st.markdown("#### 📊 Loaded data")
st.dataframe(df_raw, use_container_width=True, height=220)
st.markdown(f"**{n_units} units · {n_criteria} criteria**")

# ── Criteria direction ─────────────────────────────────────────────────────────
st.markdown("#### 🔼 Criteria preference direction")
dir_cols = st.columns(min(n_criteria, 6))
directions = {}
for i, name in enumerate(crit_names):
    with dir_cols[i % len(dir_cols)]:
        directions[name] = st.selectbox(
            name, ["↑ Increasing", "↓ Decreasing"], key=f"dir_{i}"
        )

increasing_0 = [i for i, name in enumerate(crit_names) if "Increasing" in directions[name]]
decreasing_0 = [i for i, name in enumerate(crit_names) if "Decreasing" in directions[name]]

# ── Run ────────────────────────────────────────────────────────────────────────
if st.button("▶ Induce rules", type="primary", use_container_width=True):
    matrix = df.values.astype(float)

    with st.spinner("Inducing rules…"):
        al_rules = am_rules = None
        al_match = al_dec = am_match = am_dec = None

        if run_atleast:
            al_rules, al_match, al_dec, _ = induce_atleast_rules(
                matrix, increasing_0, decreasing_0, min_conf, handle_missing)
        if run_atmost:
            am_rules, am_match, am_dec, _ = induce_atmost_rules(
                matrix, increasing_0, decreasing_0, min_conf, handle_missing)

    n_al = len(al_rules) if al_rules is not None and len(al_rules) > 0 else 0
    n_am = len(am_rules) if am_rules is not None and len(am_rules) > 0 else 0

    # Metrics
    mc1, mc2, mc3 = st.columns(3)
    for col, val, lbl in zip(
        [mc1, mc2, mc3], [n_al, n_am, n_al + n_am],
        ["At-least rules (R≥)", "At-most rules (R≤)", "Total rules"]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="value">{val}</div>'
            f'<div class="label">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    # At-least rules
    if run_atleast and n_al > 0:
        st.markdown("### R≥ · At-Least Rules")
        al_texts = format_atleast_rules(al_rules, increasing_0, decreasing_0, crit_names)
        al_supp  = compute_relative_support(al_rules, al_match, al_dec)
        al_units = get_supporting_units(al_match, al_dec, unit_names)
        for txt, supp, units in zip(al_texts, al_supp, al_units):
            tags = "".join(f'<span class="tag">{u}</span>' for u in units)
            st.markdown(
                f'<div class="rule-box atleast">{txt}'
                f'<br><span style="color:#6b7280;font-size:0.78rem;">Support: {supp:.3f} &nbsp;·&nbsp; </span>'
                f'{tags}</div>', unsafe_allow_html=True)
    elif run_atleast:
        st.info("No at-least rules induced with current settings.")

    # At-most rules
    if run_atmost and n_am > 0:
        st.markdown("### R≤ · At-Most Rules")
        am_texts = format_atmost_rules(am_rules, increasing_0, decreasing_0, crit_names)
        am_supp  = compute_relative_support(am_rules, am_match, am_dec)
        am_units = get_supporting_units(am_match, am_dec, unit_names)
        for txt, supp, units in zip(am_texts, am_supp, am_units):
            tags = "".join(f'<span class="tag">{u}</span>' for u in units)
            st.markdown(
                f'<div class="rule-box atmost">{txt}'
                f'<br><span style="color:#6b7280;font-size:0.78rem;">Support: {supp:.3f} &nbsp;·&nbsp; </span>'
                f'{tags}</div>', unsafe_allow_html=True)
    elif run_atmost:
        st.info("No at-most rules induced with current settings.")

    # Export
    if n_al + n_am > 0:
        st.markdown("### 💾 Export results")

        lines = [
            "DRSA RULE INDUCTION RESULTS",
            "=" * 60,
            f"Dataset        : {uploaded.name}",
            f"Units          : {n_units}",
            f"Criteria       : {n_criteria}",
            f"Min confidence : {min_conf}",
            f"Missing values : {handle_missing}",
            "",
        ]
        if run_atleast and n_al > 0:
            lines += ["AT-LEAST RULES (R≥)", "-" * 40]
            for txt, supp, units in zip(al_texts, al_supp, al_units):
                lines.append(txt)
                lines.append(f"  Support: {supp:.4f}  |  Units: {{{', '.join(units)}}}")
                lines.append("")
        if run_atmost and n_am > 0:
            lines += ["AT-MOST RULES (R≤)", "-" * 40]
            for txt, supp, units in zip(am_texts, am_supp, am_units):
                lines.append(txt)
                lines.append(f"  Support: {supp:.4f}  |  Units: {{{', '.join(units)}}}")
                lines.append("")

        rows = []
        if run_atleast and n_al > 0:
            for txt, supp, units in zip(al_texts, al_supp, al_units):
                rows.append({"type": "at-least", "rule": txt,
                             "support": round(supp, 4),
                             "supporting_units": "; ".join(units)})
        if run_atmost and n_am > 0:
            for txt, supp, units in zip(am_texts, am_supp, am_units):
                rows.append({"type": "at-most", "rule": txt,
                             "support": round(supp, 4),
                             "supporting_units": "; ".join(units)})
        df_out = pd.DataFrame(rows)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇ Download rules as TXT", "\n".join(lines),
                               file_name="drsa_rules.txt", mime="text/plain",
                               use_container_width=True)
        with dl2:
            st.download_button("⬇ Download rules as CSV", df_out.to_csv(index=False),
                               file_name="drsa_rules.csv", mime="text/csv",
                               use_container_width=True)
