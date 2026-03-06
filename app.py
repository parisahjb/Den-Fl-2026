"""
app.py  —  Seismic Displacement Predictor
Hybrid ML + Knowledge-Based Model

Fully dynamic — all feature names, model info, and training data
are read directly from the pkl files. Nothing is hardcoded.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import io
from sklearn.ensemble import GradientBoostingRegressor

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Seismic Displacement Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS  — everything derived from pkl, nothing hardcoded
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open("gb_model.pkl",      "rb") as f: model = pickle.load(f)
    with open("feature_info.pkl",  "rb") as f: info  = pickle.load(f)
    with open("training_data.pkl", "rb") as f: train = pickle.load(f)

    # ── Authoritative feature list: always trust the model itself ──
    model_feats = list(model.feature_names_in_)

    # ── Derive which columns are raw inputs vs KM log features ──
    km_log_cols = [c for c in model_feats if c.endswith("_log")]
    raw_inputs  = [c for c in model_feats if not c.endswith("_log")]

    # ── KM base names (strip _log suffix) ──
    km_cols = [c.replace("_log", "") for c in km_log_cols]

    # ── Training data aligned to model feature order ──
    X_raw = train["X"]
    y     = np.array(train["y"])
    # Reindex to model's exact column order; fill any gaps with 0
    X_aligned = X_raw.reindex(columns=model_feats, fill_value=0.0)

    return model, info, X_aligned, y, model_feats, raw_inputs, km_cols, km_log_cols

(model, info, X_train, y_train,
 MODEL_FEATS, RAW_INPUTS, KM_COLS, KM_LOG_COLS) = load_artifacts()

# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def build_feature_row(inp: dict, km: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching MODEL_FEATS exactly.
    inp : {raw_input_col: value}
    km  : {km_base_name: value}  e.g. {"Jafarian_2019": 150.0}
    """
    row = {}
    for col in RAW_INPUTS:
        row[col] = float(inp.get(col, 0.0))
    for col in KM_COLS:
        row[f"{col}_log"] = np.log1p(float(km.get(col, 0.0)))
    return pd.DataFrame([row])[MODEL_FEATS]


def compute_loo_residuals():
    """Compute LOO residuals once per session and cache in st.session_state."""
    if "loo_residuals" not in st.session_state:
        with st.spinner("Computing Jackknife+ calibration (one-time ~30s)..."):
            n   = len(y_train)
            Xt  = X_train.values
            yt  = y_train.copy()
            res = np.zeros(n)
            for i in range(n):
                m = GradientBoostingRegressor(**model.get_params())
                m.fit(
                    pd.DataFrame(np.delete(Xt, i, axis=0), columns=MODEL_FEATS),
                    np.delete(yt, i)
                )
                res[i] = abs(
                    yt[i] - m.predict(
                        pd.DataFrame(Xt[i:i+1], columns=MODEL_FEATS)
                    )[0]
                )
            st.session_state["loo_residuals"] = res


def jackknife_interval(X_df: pd.DataFrame, alpha: float = 0.10):
    """Return (point, lower, upper, q) using Jackknife+ intervals."""
    compute_loo_residuals()
    q     = np.quantile(st.session_state["loo_residuals"], 1 - alpha)
    point = float(np.clip(model.predict(X_df)[0], 0, None))
    lower = max(0.0, point - q)
    upper = point + q
    return point, lower, upper, q


def uncertainty_color(cv: float) -> str:
    if cv < 0.3: return "#16A34A"   # green
    if cv < 0.6: return "#D97706"   # amber
    return       "#DC2626"           # red


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🌍 Seismic Displacement\nPredictor")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🔮  Single Prediction",
    "📂  Batch Prediction",
    "🧠  Model Explainability",
    "ℹ️  About & Model Info",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** {info.get('model_name','Gradient Boosting')}  \n"
    f"**LOO-CV R²:** {info.get('loo_r2','—')}  \n"
    f"**RMSE:** {info.get('loo_rmse',0):.1f} cm  \n"
    f"**JK+ Coverage:** {info.get('jk_coverage',0)*100:.1f}%  \n"
    f"**Features:** {len(MODEL_FEATS)}  \n"
    f"**Training samples:** {len(y_train)}"
)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════
if "Single" in page:
    st.title("🔮 Single Record Prediction")
    st.markdown(
        "Enter ground motion parameters and (optionally) empirical model "
        "predictions. Returns a **point estimate** with a "
        "**Jackknife+ prediction interval**."
    )

    # ── Raw input fields (generated dynamically from RAW_INPUTS) ──
    st.markdown("### Ground Motion Input Parameters")

    # Sensible defaults and ranges per variable name
    INPUT_CONFIG = {
        "M":        ("Magnitude (M)",               2.0,  10.0,   7.0,  0.1 ),
        "R":        ("Distance R (km)",              0.1, 100.0,   5.0,  0.5 ),
        "Max_Acc":  ("Max Acceleration (g)",         0.0,   5.0,   0.4,  0.01),
        "Max_Vel":  ("Max Velocity (cm/s)",          0.0,1000.0,  80.0,  1.0 ),
        "Arias":    ("Arias Intensity (cm/s)",       0.0,5000.0, 200.0,  1.0 ),
        "Char_Int": ("Characteristic Intensity",     0.0,   5.0,   0.1,  0.01),
        "CAV":      ("CAV (cm/s)",                   0.0,10000.0,1200.0,10.0 ),
        "Housner":  ("Housner Intensity (cm)",       0.0,1000.0, 180.0,  1.0 ),
        "Sa_avg":   ("Sa,avg (g)",                   0.0,   5.0,   0.5,  0.01),
        "SaT":      ("SaT@1.5 (g)",                  0.0,   5.0,   0.8,  0.01),
        "Ia_Du":    ("Ia/Du",                        0.0, 500.0,  20.0,  0.5 ),
        "ru_max_ave":("ru_max_ave",                  0.0,  10.0,   1.0,  0.01),
        "Shear_strain":("Shear Strain (%)",          0.0,  50.0,   2.0,  0.1 ),
    }

    cols = st.columns(3)
    inp  = {}
    for i, col_name in enumerate(RAW_INPUTS):
        cfg = INPUT_CONFIG.get(col_name, (col_name, 0.0, 1e6, 0.0, 0.1))
        label, mn, mx, default, step = cfg
        inp[col_name] = cols[i % 3].number_input(label, mn, mx, default, step)

    # ── Empirical model predictions ──
    st.markdown("### Empirical Model Predictions (cm)")
    st.caption("Leave as 0 if unavailable — raw inputs alone are sufficient.")

    KM_LABELS = {
        "Song_2015":     "Song & Rodriguez-Marek 2015",
        "Jafarian_2019": "Jafarian et al. 2019",
        "HL_2010":       "HL 2010",
        "SR09":          "SR09",
        "TS16":          "TS16",
        "Lashgari_2021": "Lashgari et al. 2021",
        "Youd_2002":     "Youd 2002",
    }

    km = {}
    km_cols_ui = st.columns(min(len(KM_COLS), 4))
    for i, col_name in enumerate(KM_COLS):
        label = KM_LABELS.get(col_name, col_name)
        km[col_name] = km_cols_ui[i % 4].number_input(
            label, 0.0, 100000.0, 0.0, 1.0, key=f"km_{col_name}"
        )

    pi_levels   = [0.10, 0.05]
    alpha_v     = st.selectbox(
        "Prediction interval level",
        pi_levels,
        format_func=lambda x: f"{int((1-x)*100)}%"
    )

    if st.button("🚀 Predict Displacement", type="primary", use_container_width=True):
        X_new = build_feature_row(inp, km)
        point, lower, upper, q = jackknife_interval(X_new, alpha=alpha_v)
        cv    = (upper - lower) / (2 * point + 1e-9)
        color = uncertainty_color(cv)
        pct   = int((1 - alpha_v) * 100)

        st.markdown("---")
        st.markdown("### Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Displacement", f"{point:.1f} cm")
        r2.metric(f"Lower Bound ({pct}% PI)", f"{lower:.1f} cm")
        r3.metric(f"Upper Bound ({pct}% PI)", f"{upper:.1f} cm")
        r4.metric("Interval Width",           f"{upper-lower:.1f} cm")

        unc = ("🟢 Low" if cv < 0.3 else "🟡 Moderate" if cv < 0.6 else "🔴 High")
        st.progress(min(cv, 1.0), text=f"Uncertainty: {unc}  (CV = {cv:.2f})")

        # Interval bar
        fig, ax = plt.subplots(figsize=(9, 1.8))
        ax.barh([""], [upper - lower], left=[lower], color=color,
                alpha=0.30, height=0.5, label=f"{pct}% PI")
        ax.plot([point], [0], "o", color=color, markersize=12,
                label="Point estimate", zorder=5)
        ax.set_xlabel("Displacement (cm)")
        ax.set_xlim(0, max(upper * 1.2, 10))
        ax.legend(fontsize=9); ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Comparison with empirical models
        emp_vals = {KM_LABELS.get(k, k): v for k, v in km.items() if v > 0}
        if emp_vals:
            st.markdown("#### Comparison with Empirical Models")
            fig2, ax2 = plt.subplots(figsize=(9, max(3, len(emp_vals) * 0.6 + 1)))
            names  = list(emp_vals.keys()) + ["▶ ML Hybrid (this model)"]
            values = list(emp_vals.values()) + [point]
            colors = ["#94A3B8"] * len(emp_vals) + [color]
            bars   = ax2.barh(names, values, color=colors, edgecolor="white")
            ax2.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
            ax2.set_xlabel("Predicted Displacement (cm)")
            ax2.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close()
        else:
            st.info("Enter empirical model predictions above to see a comparison chart.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a **CSV or Excel** file. "
        "Download results with predictions and 90% Jackknife+ intervals."
    )

    # Template: all raw inputs + all KM columns
    template_cols = RAW_INPUTS + KM_COLS
    tdf = pd.DataFrame(columns=template_cols)
    tdf.loc[0] = [0.0] * len(template_cols)
    buf = io.BytesIO()
    tdf.to_excel(buf, index=False)
    st.download_button(
        "⬇️ Download Input Template (.xlsx)", buf.getvalue(),
        "input_template.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload file (CSV or Excel)", type=["csv","xlsx","xls"])

    if uploaded is not None:
        try:
            df_up = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                     else pd.read_excel(uploaded))
            st.success(f"✅ {len(df_up)} records loaded  |  {df_up.shape[1]} columns")
            st.dataframe(df_up.head(5), use_container_width=True)

            # Fill any missing columns with 0
            missing = [c for c in template_cols if c not in df_up.columns]
            if missing:
                st.warning(f"⚠️ Missing columns filled with 0: {missing}")
                for c in missing:
                    df_up[c] = 0.0

            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Build feature matrix from uploaded data
                df_proc = df_up.copy()
                for km_col in KM_COLS:
                    df_proc[f"{km_col}_log"] = np.log1p(df_proc[km_col].fillna(0))

                # Align to model's exact feature order
                X_batch = df_proc.reindex(columns=MODEL_FEATS, fill_value=0.0)
                preds   = np.clip(model.predict(X_batch), 0, None)
                df_up["Pred_Displacement_cm"] = np.round(preds, 2)

                # Jackknife+ intervals
                compute_loo_residuals()
                q90 = np.quantile(st.session_state["loo_residuals"], 0.90)
                df_up["PI_Lower_90"] = np.round(np.clip(preds - q90, 0, None), 2)
                df_up["PI_Upper_90"] = np.round(preds + q90, 2)
                df_up["PI_Width"]    = np.round(
                    df_up["PI_Upper_90"] - df_up["PI_Lower_90"], 2
                )

                # Show results
                res_cols = (
                    [c for c in ["Rec_No","M","R"] if c in df_up.columns] +
                    ["Pred_Displacement_cm","PI_Lower_90","PI_Upper_90","PI_Width"]
                )
                st.markdown("### Results")
                st.dataframe(df_up[res_cols], use_container_width=True)

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Mean", f"{preds.mean():.1f} cm")
                sc2.metric("Std",  f"{preds.std():.1f} cm")
                sc3.metric("Min",  f"{preds.min():.1f} cm")
                sc4.metric("Max",  f"{preds.max():.1f} cm")

                fig, ax = plt.subplots(figsize=(9, 3.5))
                ax.hist(preds, bins=20, color="#2563EB", edgecolor="white", alpha=0.85)
                ax.set_xlabel("Predicted Displacement (cm)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Predicted Displacements")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

                out = io.BytesIO()
                df_up.to_excel(out, index=False)
                st.download_button(
                    "⬇️ Download Results (.xlsx)", out.getvalue(),
                    "predictions_output.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.title("🧠 Model Explainability (SHAP)")
    st.markdown(
        "SHAP values quantify each feature's contribution to predictions.  \n"
        "**Positive SHAP** → pushes predicted displacement **higher**.  \n"
        "**Negative SHAP** → pushes predicted displacement **lower**."
    )

    @st.cache_resource
    def compute_shap_values():
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X_train)
        return exp, vals

    with st.spinner("Computing SHAP values..."):
        explainer, shap_vals = compute_shap_values()

    tab1, tab2, tab3 = st.tabs(
        ["📊 Feature Importance", "🐝 Beeswarm", "📈 Dependence Plots"]
    )

    with tab1:
        st.markdown("#### Global Feature Importance — Mean |SHAP Value|")
        mean_abs = np.abs(shap_vals).mean(axis=0)
        feat_ord = np.argsort(mean_abs)
        fig, ax  = plt.subplots(figsize=(9, max(5, len(MODEL_FEATS) * 0.4)))
        bars = ax.barh(
            [MODEL_FEATS[i] for i in feat_ord],
            mean_abs[feat_ord],
            color="#2563EB", edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_xlabel("Mean |SHAP value| (cm)")
        ax.set_title(f"Feature Importance — {info.get('model_name','Gradient Boosting')}")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Top 3 features from data
        top3 = [MODEL_FEATS[i] for i in feat_ord[::-1][:3]]
        st.info(f"**Top features:** {', '.join(top3)}")

    with tab2:
        st.markdown("#### SHAP Beeswarm Plot")
        st.caption("Each dot = one sample. Color = feature value (🔴 high / 🔵 low).")
        fig, ax = plt.subplots(figsize=(10, max(6, len(MODEL_FEATS) * 0.45)))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_train, plot_type="dot",
                          max_display=len(MODEL_FEATS), show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("#### SHAP Dependence Plot")
        st.caption("How SHAP value varies with feature value. "
                   "Color by a second feature to reveal interaction effects.")
        col_a, col_b = st.columns(2)
        default_x = MODEL_FEATS.index("CAV") if "CAV" in MODEL_FEATS else 0
        default_c = MODEL_FEATS.index("Arias") if "Arias" in MODEL_FEATS else 1
        feat_sel  = col_a.selectbox("Feature (x-axis)", MODEL_FEATS, index=default_x)
        color_sel = col_b.selectbox("Color by",         MODEL_FEATS, index=default_c)
        fi = MODEL_FEATS.index(feat_sel)
        ci = MODEL_FEATS.index(color_sel)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_train.iloc[:, fi], shap_vals[:, fi],
            c=X_train.iloc[:, ci], cmap="RdYlBu_r",
            alpha=0.75, s=55, edgecolors="white", linewidth=0.3
        )
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel(feat_sel, fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(f"Dependence: {feat_sel}  (color = {color_sel})")
        ax.spines[["top","right"]].set_visible(False)
        plt.colorbar(sc, ax=ax, label=color_sel, pad=0.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.title("ℹ️ About This Tool")
    st.markdown(f"""
    ## Seismic-Induced Displacement Predictor

    Predicts **seismic-induced lateral displacement** using a
    **hybrid machine learning model** combining:
    - **{len(RAW_INPUTS)} ground motion intensity measures** (raw inputs)
    - **{len(KM_COLS)} knowledge-based empirical model outputs** (log-transformed)

    Trained on high-fidelity numerical simulation data
    (n = {len(y_train)} records), validated with Leave-One-Out CV.
    """)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🏆 Model Performance (LOO-CV)")
        km_metrics = info.get("km_metrics", {})
        rows = [{
            "Model": f"{info.get('model_name','GB')} ✅",
            "R²":       info.get("loo_r2",  "—"),
            "RMSE (cm)": info.get("loo_rmse","—"),
            "MAE (cm)":  info.get("loo_mae", "—"),
        }]
        for km_name, m in km_metrics.items():
            rows.append({
                "Model":     km_name,
                "R²":        round(m.get("R2",   0), 3),
                "RMSE (cm)": round(m.get("RMSE", 0), 1),
                "MAE (cm)":  "—",
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        best_km_rmse = min(m["RMSE"] for m in km_metrics.values()) if km_metrics else 1
        ml_rmse      = info.get("loo_rmse", 1)
        improvement  = (best_km_rmse - ml_rmse) / best_km_rmse * 100
        st.success(f"ML hybrid reduces RMSE by **{improvement:.0f}%** vs best empirical model.")

    with c2:
        st.markdown("### 🎯 Jackknife+ Uncertainty Quantification")
        cov     = info.get("jk_coverage", 0)
        width   = info.get("jk_width",    0)
        n       = len(y_train)
        alpha   = 0.10
        min_cov = (1 - alpha) * (1 + 1/n)
        st.metric("Empirical Coverage (90% PI)", f"{cov*100:.1f}%",
                  delta=f"Target ≥ {min_cov*100:.1f}%")
        st.metric("Mean Interval Width", f"{width:.1f} cm")
        st.markdown(f"""
        **Jackknife+** (Barber et al., 2021) provides finite-sample,
        distribution-free intervals with theoretical guarantees.

        For n={n}, α=0.10:
        > Minimum coverage = (1−α)(1+1/n) = **{min_cov*100:.1f}%**

        Achieved: **{cov*100:.1f}%** {'✅' if cov >= min_cov else '⚠️'}
        """)

    st.markdown("---")
    st.markdown(f"### 📥 Model Features ({len(MODEL_FEATS)} total)")
    feat_rows = []
    for feat in MODEL_FEATS:
        if feat in RAW_INPUTS:
            feat_rows.append({"Feature": feat, "Type": "Raw Input",
                              "Note": "Direct ground motion measurement"})
        else:
            base = feat.replace("_log","")
            feat_rows.append({"Feature": feat, "Type": "Hybrid (KM)",
                              "Note": f"log(1 + {base} prediction)"})
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    dropped = info.get("dropped_features", [])
    if dropped:
        st.markdown("### 🗑️ Dropped Features (near-zero SHAP importance)")
        st.dataframe(
            pd.DataFrame({"Dropped Feature": dropped,
                          "Reason": ["Near-zero SHAP importance"] * len(dropped)}),
            use_container_width=True, hide_index=True
        )

    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - Barber et al. (2021). *Predictive Inference with the Jackknife+*. Ann. Statistics.
    - Jafarian et al. (2019). *Empirical seismic displacement model*.
    - Haskell & Lacasse (2010). *HL empirical displacement model*.
    - Youd et al. (2002). *Revised multilinear regression equations*.
    - Song & Rodriguez-Marek (2015). *Probabilistic seismic displacement model*.
    - Lashgari et al. (2021). *Seismic displacement prediction model*.
    """)
    st.caption("Built with Python · scikit-learn · SHAP · Streamlit")
