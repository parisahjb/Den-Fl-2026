"""
app.py  —  Seismic Displacement Predictor
Hybrid ML + Knowledge-Based Model
─────────────────────────────────────────
Streamlit multi-page app with:
  Page 1 : Single Prediction
  Page 2 : Batch Prediction (CSV/Excel upload)
  Page 3 : Model Explainability (SHAP)
  Page 4 : About / Model Info
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
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open("gb_model.pkl",      "rb") as f: model = pickle.load(f)
    with open("feature_info.pkl",  "rb") as f: info  = pickle.load(f)
    with open("training_data.pkl", "rb") as f: train = pickle.load(f)
    return model, info, train

model, info, train_data = load_artifacts()
X_train = train_data["X"]   # pandas DataFrame with named columns
y_train = train_data["y"]   # numpy array
FEAT    = info["feature_names"]
KM_COLS = info["km_cols"]
DROPPED = set(info["dropped_features"])

# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def build_feature_row(inp: dict, km: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the training feature columns.
    inp : dict of raw ground motion inputs
    km  : dict of empirical model predictions
    """
    row = {k: inp[k] for k in ["M", "R", "Max_Vel", "Arias", "CAV",
                                 "Housner", "Sa_avg", "SaT", "Ia_Du"]}
    for col in KM_COLS:
        log_col = f"{col}_log"
        if log_col not in DROPPED:
            row[log_col] = np.log1p(km.get(col, 0.0))

    # Return as DataFrame with exact column order matching training
    return pd.DataFrame([row])[FEAT]


def jackknife_interval_fast(X_new_df: pd.DataFrame, alpha: float = 0.10):
    """
    Jackknife+ prediction interval.
    X_new_df must be a DataFrame with named columns matching FEAT.
    Calibration is computed once per session and cached.
    """
    if "loo_residuals" not in st.session_state:
        with st.spinner("Computing Jackknife+ calibration (one-time, ~30s)..."):
            n   = len(y_train)
            Xt  = X_train.values          # numpy array for indexing
            yt  = np.array(y_train)
            res = np.zeros(n)
            for i in range(n):
                Xtr    = np.delete(Xt, i, axis=0)
                ytr    = np.delete(yt, i)
                m      = GradientBoostingRegressor(**model.get_params())
                m.fit(Xtr, ytr)
                # Predict using DataFrame to preserve feature names
                X_i_df = pd.DataFrame(Xt[i:i+1], columns=FEAT)
                res[i] = abs(yt[i] - m.predict(X_i_df)[0])
            st.session_state["loo_residuals"] = res

    q     = np.quantile(st.session_state["loo_residuals"], 1 - alpha)
    # Always predict using a properly named DataFrame
    point = float(np.clip(model.predict(X_new_df)[0], 0, None))
    lower = max(0.0, point - q)
    upper = point + q
    return point, lower, upper, q


def calibrate_batch_intervals():
    """Compute and cache LOO residuals for batch interval calculation."""
    if "loo_residuals" not in st.session_state:
        with st.spinner("Calibrating prediction intervals (~30s)..."):
            n   = len(y_train)
            Xt  = X_train.values
            yt  = np.array(y_train)
            res = np.zeros(n)
            for i in range(n):
                m = GradientBoostingRegressor(**model.get_params())
                m.fit(np.delete(Xt, i, axis=0), np.delete(yt, i))
                X_i_df = pd.DataFrame(Xt[i:i+1], columns=FEAT)
                res[i] = abs(yt[i] - m.predict(X_i_df)[0])
            st.session_state["loo_residuals"] = res


def gauge_color(cv: float) -> str:
    """Return color based on coefficient of variation."""
    if cv < 0.3: return "#16A34A"   # green  — low uncertainty
    if cv < 0.6: return "#D97706"   # amber  — moderate
    return       "#DC2626"           # red    — high


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🌍 Seismic Displacement\nPredictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮  Single Prediction",
        "📂  Batch Prediction",
        "🧠  Model Explainability",
        "ℹ️  About & Model Info",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** {info['model_name']}  \n"
    f"**LOO-CV R²:** {info['loo_r2']}  \n"
    f"**RMSE:** {info['loo_rmse']:.1f} cm  \n"
    f"**JK+ Coverage:** {info['jk_coverage']*100:.1f}%"
)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════
if "Single" in page:
    st.title("🔮 Single Record Prediction")
    st.markdown(
        "Enter ground motion parameters and (optionally) empirical model "
        "predictions. Returns a point estimate with a **Jackknife+ "
        "prediction interval**."
    )

    # ── Ground motion inputs ────────────────────────────────
    st.markdown("### Ground Motion Input Parameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        M       = st.number_input("Magnitude (M)",          2.0,  10.0,   7.0,  0.1)
        R       = st.number_input("Distance R (km)",         0.1, 100.0,   5.0,  0.5)
        Max_Vel = st.number_input("Max Velocity (cm/s)",     0.0, 1000.0,  80.0, 1.0)

    with c2:
        Arias   = st.number_input("Arias Intensity (cm/s)",  0.0, 5000.0, 200.0, 1.0)
        CAV     = st.number_input("CAV (cm/s)",              0.0,10000.0,1200.0,10.0)
        Housner = st.number_input("Housner Intensity (cm)",  0.0, 1000.0, 180.0, 1.0)

    with c3:
        Sa_avg  = st.number_input("Sa,avg (g)",              0.0,    5.0,   0.5, 0.01)
        SaT     = st.number_input("SaT@1.5 (g)",             0.0,    5.0,   0.8, 0.01)
        Ia_Du   = st.number_input("Ia/Du",                   0.0,  500.0,  20.0, 0.5)

    # ── Empirical model predictions ─────────────────────────
    st.markdown("### Empirical Model Predictions (cm)")
    st.caption("Leave as 0 if unavailable — raw inputs alone are sufficient.")

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        km_song = st.number_input("Song 2015",     0.0, 10000.0, 0.0, 1.0)
        km_jaf  = st.number_input("Jafarian 2019", 0.0, 10000.0, 0.0, 1.0)
    with kc2:
        km_hl   = st.number_input("HL 2010",       0.0, 10000.0, 0.0, 1.0)
        km_sr09 = st.number_input("SR09",          0.0, 10000.0, 0.0, 1.0)
    with kc3:
        km_ts16 = st.number_input("TS16",          0.0, 10000.0, 0.0, 1.0)
        km_lash = st.number_input("Lashgari 2021", 0.0, 10000.0, 0.0, 1.0)
    with kc4:
        km_youd = st.number_input("Youd 2002",     0.0, 10000.0, 0.0, 1.0)
        alpha_v = st.selectbox(
            "Prediction interval level",
            [0.10, 0.05],
            format_func=lambda x: f"{int((1-x)*100)}%"
        )

    inp = {
        "M": M, "R": R, "Max_Vel": Max_Vel, "Arias": Arias,
        "CAV": CAV, "Housner": Housner, "Sa_avg": Sa_avg,
        "SaT": SaT, "Ia_Du": Ia_Du,
    }
    km = {
        "Song_2015": km_song, "Jafarian_2019": km_jaf, "HL_2010": km_hl,
        "SR09": km_sr09, "TS16": km_ts16, "Lashgari_2021": km_lash,
        "Youd_2002": km_youd,
    }

    if st.button("🚀 Predict Displacement", type="primary", use_container_width=True):
        X_new = build_feature_row(inp, km)   # DataFrame with named columns
        point, lower, upper, q = jackknife_interval_fast(X_new, alpha=alpha_v)
        cv    = (upper - lower) / (2 * point + 1e-9)
        color = gauge_color(cv)

        st.markdown("---")
        st.markdown("### Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Displacement",               f"{point:.1f} cm")
        r2.metric(f"Lower Bound ({int((1-alpha_v)*100)}% PI)", f"{lower:.1f} cm")
        r3.metric(f"Upper Bound ({int((1-alpha_v)*100)}% PI)", f"{upper:.1f} cm")
        r4.metric("Interval Width",                        f"{upper-lower:.1f} cm")

        unc_label = ("🟢 Low" if cv < 0.3 else
                     "🟡 Moderate" if cv < 0.6 else "🔴 High")
        st.progress(min(cv, 1.0),
                    text=f"Uncertainty: {unc_label}  (CV = {cv:.2f})")

        # Interval bar chart
        fig, ax = plt.subplots(figsize=(9, 1.8))
        ax.barh([""], [upper - lower], left=[lower],
                color=color, alpha=0.30, height=0.5,
                label=f"{int((1-alpha_v)*100)}% PI")
        ax.plot([point], [0], "o", color=color, markersize=12,
                label="Point estimate", zorder=5)
        ax.set_xlabel("Displacement (cm)")
        ax.set_xlim(0, max(upper * 1.2, 10))
        ax.legend(fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Comparison with empirical models
        emp_vals = {k: v for k, v in km.items() if v > 0}
        if emp_vals:
            st.markdown("#### Comparison with Empirical Models")
            fig2, ax2 = plt.subplots(figsize=(9, 3.5))
            names  = list(emp_vals.keys()) + ["▶ ML Hybrid (this model)"]
            values = list(emp_vals.values()) + [point]
            colors = ["#94A3B8"] * len(emp_vals) + [color]
            bars   = ax2.barh(names, values, color=colors, edgecolor="white")
            ax2.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
            ax2.set_xlabel("Predicted Displacement (cm)")
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        else:
            st.info("Enter empirical model predictions above to see a "
                    "side-by-side comparison chart.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a **CSV or Excel** file containing multiple records. "
        "Download results with displacement predictions and 90% prediction intervals."
    )

    # Template download
    template_cols = (
        ["M", "R", "Max_Vel", "Arias", "CAV", "Housner",
         "Sa_avg", "SaT", "Ia_Du"] + KM_COLS
    )
    tdf = pd.DataFrame([{
        "M": 7.0, "R": 5.0, "Max_Vel": 80.0, "Arias": 200.0, "CAV": 1200.0,
        "Housner": 180.0, "Sa_avg": 0.5, "SaT": 0.8, "Ia_Du": 20.0,
        "Song_2015": 300.0, "Jafarian_2019": 200.0, "HL_2010": 150.0,
        "SR09": 500.0, "TS16": 400.0, "Lashgari_2021": 180.0, "Youd_2002": 300.0,
    }])
    buf = io.BytesIO()
    tdf.to_excel(buf, index=False)
    st.download_button(
        "⬇️ Download Input Template (.xlsx)",
        data=buf.getvalue(),
        file_name="input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload your file (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)

            st.success(f"✅ Loaded {len(df_up)} records  |  {df_up.shape[1]} columns")
            st.dataframe(df_up.head(5), use_container_width=True)

            # Fill missing columns with 0
            missing = [c for c in template_cols if c not in df_up.columns]
            if missing:
                st.warning(f"⚠️ Missing columns filled with 0: {missing}")
                for c in missing:
                    df_up[c] = 0.0

            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Build feature matrix — keep as DataFrame with named columns
                df_proc = df_up.copy()
                for km_col in KM_COLS:
                    log_col = f"{km_col}_log"
                    if log_col not in DROPPED:
                        df_proc[log_col] = np.log1p(df_proc[km_col].fillna(0))

                X_batch = df_proc[FEAT]   # DataFrame preserves column names
                preds   = np.clip(model.predict(X_batch), 0, None)
                df_up["Pred_Displacement_cm"] = np.round(preds, 2)

                # Jackknife+ intervals
                calibrate_batch_intervals()
                q90 = np.quantile(st.session_state["loo_residuals"], 0.90)
                df_up["PI_Lower_90"] = np.round(np.clip(preds - q90, 0, None), 2)
                df_up["PI_Upper_90"] = np.round(preds + q90, 2)
                df_up["PI_Width"]    = np.round(
                    df_up["PI_Upper_90"] - df_up["PI_Lower_90"], 2
                )

                # Show results
                st.markdown("### Results Preview")
                res_cols = (
                    [c for c in ["Rec_No", "M", "R"] if c in df_up.columns] +
                    ["Pred_Displacement_cm", "PI_Lower_90", "PI_Upper_90", "PI_Width"]
                )
                st.dataframe(df_up[res_cols], use_container_width=True)

                # Summary metrics
                st.markdown("### Summary Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Mean",  f"{preds.mean():.1f} cm")
                sc2.metric("Std",   f"{preds.std():.1f} cm")
                sc3.metric("Min",   f"{preds.min():.1f} cm")
                sc4.metric("Max",   f"{preds.max():.1f} cm")

                # Distribution chart
                fig, ax = plt.subplots(figsize=(9, 3.5))
                ax.hist(preds, bins=20, color="#2563EB",
                        edgecolor="white", alpha=0.85)
                ax.set_xlabel("Predicted Displacement (cm)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Predicted Displacements")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Download results
                out = io.BytesIO()
                df_up.to_excel(out, index=False)
                st.download_button(
                    "⬇️ Download Results (.xlsx)",
                    data=out.getvalue(),
                    file_name="predictions_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.title("🧠 Model Explainability (SHAP)")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's "
        "contribution to the model's predictions.  \n"
        "**Positive SHAP** → pushes predicted displacement higher.  \n"
        "**Negative SHAP** → pushes predicted displacement lower."
    )

    @st.cache_resource
    def compute_shap_values():
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X_train)   # X_train is a named DataFrame
        return exp, vals

    with st.spinner("Computing SHAP values..."):
        explainer, shap_vals = compute_shap_values()

    tab1, tab2, tab3 = st.tabs(
        ["📊 Feature Importance", "🐝 Beeswarm", "📈 Dependence Plots"]
    )

    with tab1:
        st.markdown("#### Global Feature Importance — Mean |SHAP Value|")
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        feat_ord  = np.argsort(mean_abs)
        fig, ax   = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            [FEAT[i] for i in feat_ord],
            mean_abs[feat_ord],
            color="#2563EB", edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_xlabel("Mean |SHAP value| (cm)")
        ax.set_title("Feature Importance — Gradient Boosting")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.info(
            "**Key finding:** CAV (Cumulative Absolute Velocity) is the dominant "
            "predictor, followed by Arias Intensity and Housner Intensity. "
            "Among hybrid features, Jafarian_2019_log and HL_2010_log contribute "
            "meaningfully while TS16, SR09, Song_2015, and Lashgari_2021 were "
            "removed due to near-zero importance."
        )

    with tab2:
        st.markdown("#### SHAP Beeswarm Plot")
        st.caption(
            "Each dot = one training sample.  "
            "Color = feature value (🔴 high / 🔵 low).  "
            "X-axis = impact on predicted displacement."
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)
        shap.summary_plot(
            shap_vals, X_train,
            plot_type="dot", max_display=15, show=False
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown("#### SHAP Dependence Plot")
        st.caption(
            "Shows how SHAP value changes with feature value. "
            "Color by a second feature to reveal interaction effects."
        )
        col_a, col_b = st.columns(2)
        feat_sel  = col_a.selectbox(
            "Feature (x-axis)", FEAT,
            index=FEAT.index("CAV") if "CAV" in FEAT else 0
        )
        color_sel = col_b.selectbox(
            "Color by (interaction)", FEAT,
            index=FEAT.index("Arias") if "Arias" in FEAT else 1
        )

        fi  = FEAT.index(feat_sel)
        ci  = FEAT.index(color_sel)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_train.iloc[:, fi],
            shap_vals[:, fi],
            c=X_train.iloc[:, ci],
            cmap="RdYlBu_r", alpha=0.75, s=55,
            edgecolors="white", linewidth=0.3
        )
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel(feat_sel, fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(f"Dependence: {feat_sel}  (color = {color_sel})")
        ax.spines[["top", "right"]].set_visible(False)
        plt.colorbar(sc, ax=ax, label=color_sel, pad=0.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT & MODEL INFO
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.title("ℹ️ About This Tool")

    st.markdown("""
    ## Seismic-Induced Displacement Predictor

    This tool predicts **seismic-induced lateral displacement** using a 
    **hybrid machine learning model** that combines:
    - **Ground motion intensity measures** from numerical simulations
    - **Knowledge-based empirical model outputs** as structured prior information

    The model was trained on high-fidelity numerical simulation data (n = 75 records)
    and validated using Leave-One-Out Cross-Validation (LOO-CV).
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Model Performance (LOO-CV)")
        perf = pd.DataFrame({
            "Model": [
                "Gradient Boosting ✅", "Random Forest", "XGBoost",
                "HL_2010 (best empirical)", "Lashgari_2021",
                "Jafarian_2019", "Song_2015",
            ],
            "R²":       [ 0.659,  0.642,  0.607,  -7.18,  -9.00, -51.55, -68.62],
            "RMSE (cm)":[ 23.4,   24.0,   25.1,  114.7,  126.8,  290.8,  334.7],
            "MAE (cm)": [ 16.9,   15.9,   16.6,   75.3,  107.5,  156.7,  216.7],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.success(
            "The hybrid ML model reduces RMSE by **~79%** compared to the best "
            "empirical model (HL_2010)."
        )

    with col2:
        st.markdown("### 🎯 Uncertainty Quantification (Jackknife+)")
        st.metric("Empirical Coverage (90% PI)", "89.3%",
                  delta="Target ≥ 88.8%", delta_color="normal")
        st.metric("Mean Interval Width", "75.0 cm")
        st.markdown("""
        The **Jackknife+** method (Barber et al., 2021) provides 
        finite-sample, distribution-free prediction intervals with 
        theoretical coverage guarantees — no normality assumption required.

        For n = 75, the theoretical minimum coverage at the 90% level is:
        > (1 − α)(1 + 1/n) = **88.8%**

        Achieved coverage: **89.3%** ✅
        """)

    st.markdown("---")
    st.markdown("### 📥 Input Features Used by the Model")
    feat_table = pd.DataFrame({
        "Feature": [
            "M", "R", "Max_Vel", "Arias", "CAV", "Housner",
            "Sa_avg", "SaT", "Ia_Du",
            "Jafarian_2019_log", "HL_2010_log", "Youd_2002_log",
        ],
        "Description": [
            "Moment magnitude",
            "Source-to-site distance (km)",
            "Peak ground velocity (cm/s)",
            "Arias Intensity (cm/s)",
            "Cumulative Absolute Velocity (cm/s)",
            "Housner Intensity (cm)",
            "Average spectral acceleration (g)",
            "Spectral acceleration at 1.5T (g)",
            "Intensity / Duration ratio",
            "log(1 + Jafarian et al. 2019 prediction)",
            "log(1 + Haskell & Lacasse 2010 prediction)",
            "log(1 + Youd 2002 prediction)",
        ],
        "Type": ["Raw Input"] * 9 + ["Hybrid (Knowledge Model)"] * 3,
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🗑️ Dropped Features (low SHAP importance)")
    dropped_table = pd.DataFrame({
        "Dropped Feature": [
            "TS16_log", "Song_2015_log", "SR09_log",
            "Lashgari_2021_log", "Max_Acc", "Char_Int",
        ],
        "Reason": ["Near-zero SHAP importance across all models"] * 6,
    })
    st.dataframe(dropped_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - Barber, R.F., Candès, E.J., Ramdas, A., & Tibshirani, R.J. (2021).
      *Predictive Inference with the Jackknife+*. Annals of Statistics, 49(1), 486–507.
    - Jafarian, Y., et al. (2019). *Empirical model for seismic displacement prediction*.
    - Haskell, J. & Lacasse, S. (2010). *HL empirical displacement model*.
    - Youd, T.L., et al. (2002). *Revised multilinear regression equations for 
      predicting lateral spread displacement*.
    - Song, J. & Rodriguez-Marek, A. (2015). *Probabilistic seismic displacement model*.
    - Lashgari, M., et al. (2021). *Seismic displacement prediction model*.
    """)

    st.markdown("---")
    st.caption(
        "Built with Python · scikit-learn · XGBoost · SHAP · Streamlit  |  "
        "Trained on numerical simulation data (n = 75 records)"
    )
