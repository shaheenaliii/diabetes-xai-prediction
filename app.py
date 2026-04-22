# ============================================================
# app.py
# Streamlit interactive app.
# Usage: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import ks_2samp
from evaluation import (
    generate_gaussian_noise,
    generate_copula,
    statistical_evaluation,
    get_ks_per_feature,
    ml_utility_evaluation,
    privacy_evaluation,
    compute_final_score,
)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="Synthetic Data Evaluator", layout="wide")

st.title("Synthetic Data Evaluation Framework")
st.write("Upload your dataset, generate synthetic data, and evaluate it across three dimensions.")
st.divider()

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
target_col    = st.sidebar.text_input("Target column name", value="target")

st.sidebar.subheader("Generators")
use_noise  = st.sidebar.checkbox("Gaussian Noise (baseline)", value=True)
use_copula = st.sidebar.checkbox("Gaussian Copula",           value=True)

st.sidebar.subheader("Score weights")
w_stat = st.sidebar.number_input("Statistical weight", 0.0, 1.0, 0.35, 0.05)
w_ml   = st.sidebar.number_input("ML Utility weight",  0.0, 1.0, 0.40, 0.05)
w_priv = st.sidebar.number_input("Privacy weight",     0.0, 1.0, 0.25, 0.05)

run = st.sidebar.button("Run Evaluation")

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.stop()

# ── LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

if target_col not in df.columns:
    st.error(f"Column '{target_col}' not found. Please check the target column name.")
    st.stop()

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Preview", "Statistical", "ML Utility", "Privacy", "Final Score"
])

# ── TAB 1: DATA PREVIEW ──────────────────────────────────────
with tab1:
    st.subheader("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",           df.shape[0])
    c2.metric("Columns",        df.shape[1])
    c3.metric("Target balance", f"{df[target_col].mean():.1%} positive")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.subheader("Feature distributions (real data)")
    cols_per_row = 4
    rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, 3 * rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='white', linewidth=0.5)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── STOP HERE IF RUN NOT CLICKED ─────────────────────────────
if not run:
    for t in [tab2, tab3, tab4, tab5]:
        with t:
            st.info("Click 'Run Evaluation' in the sidebar.")
    st.stop()

if not use_noise and not use_copula:
    st.sidebar.error("Select at least one generator.")
    st.stop()

# ── GENERATE SYNTHETIC DATA ──────────────────────────────────
generators = {}
with st.spinner("Generating synthetic datasets..."):
    if use_noise:
        generators["Gaussian Noise"]  = generate_gaussian_noise(df, num_cols)
    if use_copula:
        generators["Gaussian Copula"] = generate_copula(df, len(df), target_col)

# ── RUN ALL EVALUATIONS ──────────────────────────────────────
stat_scores  = {}
stat_subs    = {}
ml_scores    = {}
ml_details   = {}
priv_scores  = {}
priv_details = {}

with st.spinner("Running evaluations — this may take a minute..."):
    for name, syn in generators.items():
        stat_scores[name], stat_subs[name]          = statistical_evaluation(df, syn, num_cols)
        ml_scores[name],   ml_details[name]         = ml_utility_evaluation(df, syn, target_col)
        priv_scores[name], *priv_rest                = privacy_evaluation(df, syn, num_cols, target_col)
        priv_details[name]                           = priv_rest   # [dcr, attack_auc]

# ── TAB 2: STATISTICAL ───────────────────────────────────────
with tab2:
    st.subheader("Statistical fidelity scores")
    cols = st.columns(len(generators))
    for i, (name, score) in enumerate(stat_scores.items()):
        cols[i].metric(name, f"{score:.4f}")

    st.write("---")
    st.subheader("Sub-scores breakdown")
    sub_df = pd.DataFrame(stat_subs).T
    st.dataframe(sub_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)

    st.write("---")
    st.subheader("KS p-values per feature")
    st.write("Green = p > 0.05 (distributions are similar). Red = p < 0.05 (significant difference).")

    ks_data = {name: get_ks_per_feature(df, syn, num_cols)
               for name, syn in generators.items()}
    ks_df = pd.DataFrame(ks_data)
    st.dataframe(
        ks_df.style.format("{:.4f}")
             .highlight_between(left=0.05, right=1.0, color='#d4edda')
             .highlight_between(left=0.0,  right=0.05, color='#f8d7da'),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(num_cols))
    bar_w = 0.35
    bar_colors = ['steelblue', 'darkorange']
    for i, (name, scores) in enumerate(ks_data.items()):
        offset = (i - len(generators)/2 + 0.5) * bar_w
        ax.bar(x + offset, list(scores.values()), bar_w,
               label=name, color=bar_colors[i % len(bar_colors)], alpha=0.8)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("KS p-value")
    ax.set_title("Feature-wise KS test")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.write("---")
    st.subheader("Correlation matrices")
    n_plots = 1 + len(generators)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    sns.heatmap(df[num_cols].corr(), ax=axes[0], annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.3, annot_kws={"size": 6}, cbar=False)
    axes[0].set_title("Real data", fontsize=10)
    for i, (name, syn) in enumerate(generators.items()):
        sns.heatmap(syn[num_cols].corr(), ax=axes[i+1], annot=True, fmt=".2f",
                    cmap="Blues", linewidths=0.3, annot_kws={"size": 6}, cbar=False)
        axes[i+1].set_title(name, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 3: ML UTILITY ────────────────────────────────────────
with tab3:
    st.subheader("ML utility — TSTR (Train on Synthetic, Test on Real)")
    st.write("The model trains on synthetic data and is tested on real data. Scores close to 1 mean the synthetic data is a strong substitute for training.")

    for name in generators:
        d = ml_details[name]
        st.write(f"**{name}** — ML Utility Score: `{ml_scores[name]:.4f}`")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Real Accuracy", f"{d['acc_real']:.4f}")
        c2.metric("Real F1",       f"{d['f1_real']:.4f}")
        c3.metric("Real AUC",      f"{d['auc_real']:.4f}")
        c4.metric("TSTR Accuracy", f"{d['acc_syn']:.4f}", delta=f"{d['acc_syn']-d['acc_real']:+.4f}")
        c5.metric("TSTR F1",       f"{d['f1_syn']:.4f}",  delta=f"{d['f1_syn']-d['f1_real']:+.4f}")
        c6.metric("TSTR AUC",      f"{d['auc_syn']:.4f}", delta=f"{d['auc_syn']-d['auc_real']:+.4f}")
        st.write("")

    st.write("---")
    st.subheader("Performance comparison chart")
    metrics      = ["Accuracy", "F1", "AUC"]
    metric_keys  = [('acc_real','acc_syn'), ('f1_real','f1_syn'), ('auc_real','auc_syn')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (metric, (rk, sk)) in enumerate(zip(metrics, metric_keys)):
        real_vals = [ml_details[n][rk] for n in generators]
        syn_vals  = [ml_details[n][sk] for n in generators]
        x = np.arange(len(generators))
        axes[i].bar(x - 0.2, real_vals, 0.35, label="Real (baseline)", color="steelblue",  alpha=0.85)
        axes[i].bar(x + 0.2, syn_vals,  0.35, label="TSTR (synthetic)", color="darkorange", alpha=0.85)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(list(generators.keys()), fontsize=8)
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1.15)
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 4: PRIVACY ───────────────────────────────────────────
with tab4:
    st.subheader("Privacy risk evaluation")
    st.write("Higher scores = better privacy. DCR measures how far synthetic records are from real ones. MI Attack measures how easily an attacker can identify real records.")

    for name in generators:
        dcr, mi_auc = priv_details[name]
        risk = ("Low risk"      if priv_scores[name] > 0.7 else
                "Moderate risk" if priv_scores[name] > 0.4 else
                "High risk")
        st.write(f"**{name}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Privacy Score",          f"{priv_scores[name]:.4f}")
        c2.metric("DCR (avg min distance)", f"{dcr:.4f}")
        c3.metric("MI Attack AUC",          f"{mi_auc:.4f}")
        st.write(f"Risk level: **{risk}**")
        st.write("")

    st.write("---")
    st.subheader("What these numbers mean")
    st.write("""
- **DCR (Distance to Closest Record):** How far each synthetic patient record is from the nearest real patient. Higher = safer.
- **MI Attack AUC:** An attacker tries to detect if a record came from the real dataset. AUC = 0.5 means they are just guessing (good). AUC = 1.0 means they can perfectly identify real records (bad).
- **Privacy Score:** Combined measure. Closer to 1 is better.
""")

# ── TAB 5: FINAL SCORE ───────────────────────────────────────
with tab5:
    st.subheader("Final quality scores")
    st.write(f"Formula: `({w_stat} × Statistical + {w_ml} × ML Utility + {w_priv} × Privacy) / {w_stat+w_ml+w_priv}`")

    final = {
        name: compute_final_score(stat_scores[name], ml_scores[name], priv_scores[name],
                                  w_stat, w_ml, w_priv)
        for name in generators
    }
    best = max(final, key=final.get)

    cols = st.columns(len(generators))
    for i, (name, score) in enumerate(final.items()):
        label = f"{name} ★ Best" if name == best else name
        cols[i].metric(label, f"{score:.4f}")

    st.write("---")
    summary = pd.DataFrame({
        "Generator":   list(generators.keys()),
        "Statistical": [stat_scores[n]  for n in generators],
        "ML Utility":  [ml_scores[n]    for n in generators],
        "Privacy":     [priv_scores[n]  for n in generators],
        "Final Score": [final[n]        for n in generators],
    }).set_index("Generator")

    st.dataframe(
        summary.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
        use_container_width=True
    )

    st.write("---")
    st.subheader("Score breakdown charts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x    = np.arange(len(generators))
    w    = 0.22
    axes[0].bar(x - w, [stat_scores[n] for n in generators], w, label="Statistical", color="steelblue",  alpha=0.85)
    axes[0].bar(x,     [ml_scores[n]   for n in generators], w, label="ML Utility",  color="darkorange", alpha=0.85)
    axes[0].bar(x + w, [priv_scores[n] for n in generators], w, label="Privacy",     color="seagreen",   alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(generators.keys()), fontsize=9)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Pillar scores by generator")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    bar_colors = ['steelblue', 'darkorange', 'seagreen', 'tomato']
    bars = axes[1].bar(list(final.keys()), list(final.values()),
                       color=bar_colors[:len(final)], alpha=0.85)
    for bar, val in zip(bars, final.values()):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Final quality score")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"Best generator: {best} with a final score of {final[best]:.4f}")
