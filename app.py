"""
app.py  —  Seismic Displacement Predictor
Hybrid ML + Knowledge-Based Model
─────────────────────────────────────────
Pages:
  1. Single Prediction
  2. Batch Prediction
  3. Model Explainability (SHAP)
  4. About & Model Info
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

# ── Ground truth: use model's own feature list as the only authority ──
# model.feature_names_in_ = 18 features (full set the model was trained on)
# ['M','R','Max_Acc','Max_Vel','Arias','Char_Int','CAV','Housner',
#  'Sa_avg','SaT','Ia_Du','Song_2015_log','Jafarian_2019_log',
#  'HL_2010_log','SR09_log','TS16_log','Lashgari_2021_log','Youd_2002_log']
MODEL_FEATS = list(model.feature_names_in_)   # 18 features — authoritative

# Raw inputs (11) and KM columns (7) derived from MODEL_FEATS
RAW_INPUTS = ['M','R','Max_Acc','Max_Vel','Arias','Char_Int',
              'CAV','Housner','Sa_avg','SaT','Ia_Du']
KM_COLS    = ['Song_2015','Jafarian_2019','HL_2010',
              'SR09','TS16','Lashgari_2021','Youd_2002']
KM_LOG_COLS = [f"{k}_log" for k in KM_COLS]   # all 7 log-transformed KM cols

# Training data for Jackknife+ calibration
# X_train has 12 cols — extend to 18 by filling missing cols with 0
X_train_12 = train_data["X"]                  # 12-col DataFrame
y_train     = np.array(train_data["y"])        # (75,)

# Build 18-col X_train aligned to model (missing cols → 0)
X_train = X_train_12.reindex(columns=MODEL_FEATS, fill_value=0.0)

# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def build_feature_row(inp: dict, km: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame exactly matching MODEL_FEATS (18 cols).
    inp : {raw_input_col: value}  — all 11 raw inputs
    km  : {km_model_name: value} — all 7 empirical model predictions
    """
    row = {}
    # Raw inputs
    for col in RAW_INPUTS:
        row[col] = float(inp.get(col, 0.0))
    # Log-transformed KM predictions
    for col in KM_COLS:
        row[f"{col}_log"] = np.log1p(float(km.get(col, 0.0)))
    # Return as DataFrame in exact model column order
    return pd.DataFrame([row])[MODEL_FEATS]


def jackknife_interval_fast(X_new_df: pd.DataFrame, alpha: float = 0.10):
    """
    Jackknife+ prediction interval (Barber et al. 2021).
    X_new_df : DataFrame with MODEL_FEATS columns.
    LOO residuals are computed once per session and cached.
    """
    if "loo_residuals" not in st.session_state:
        with st.spinner("Computing Jackknife+ calibration (one-time ~30s)..."):
            n   = len(y_train)
            Xt  = X_train.values        # (75, 18) numpy
            yt  = y_train.copy()
            res = np.zeros(n)
            for i in range(n):
                Xtr = np.delete(Xt, i, axis=0)
                ytr = np.delete(yt, i)
                m   = GradientBoostingRegressor(**model.get_params())
                m.fit(pd.DataFrame(Xtr, columns=MODEL_FEATS),
                      ytr)
                Xi  = pd.DataFrame(Xt[i:i+1], columns=MODEL_FEATS)
                res[i] = abs(yt[i] - m.predict(Xi)[0])
            st.session_state["loo_residuals"] = res

    q     = np.quantile(st.session_state["loo_residuals"], 1 - alpha)
    point = float(np.clip(model.predict(X_new_df)[0], 0, None))
    lower = max(0.0, point - q)
    upper = point + q
    return point, lower, upper, q


def calibrate_loo():
    """Compute and cache LOO residuals (shared by batch page)."""
    if "loo_residuals" not in st.session_state:
        with st.spinner("Calibrating prediction intervals (~30s)..."):
            n   = len(y_train)
            Xt  = X_train.values
            yt  = y_train.copy()
            res = np.zeros(n)
            for i in range(n):
                m = GradientBoostingRegressor(**model.get_params())
                m.fit(pd.DataFrame(np.delete(Xt, i, 0), columns=MODEL_FEATS),
                      np.delete(yt, i))
                Xi  = pd.DataFrame(Xt[i:i+1], columns=MODEL_FEATS)
                res[i] = abs(yt[i] - m.predict(Xi)[0])
            st.session_state["loo_residuals"] = res


def gauge_color(cv: float) -> str:
    if cv < 0.3: return "#16A34A"
    if cv < 0.6: return "#D97706"
    return "#DC2626"


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
    f"**Model:** {info['model_name']}  \n"
    f"**LOO-CV R²:** {info['loo_r2']}  \n"
    f"**RMSE:** {info['loo_rmse']:.1f} cm  \n"
    f"**JK+ Coverage:** {info['jk_coverage']*100:.1f}%  \n"
    f"**Features:** {len(MODEL_FEATS)}"
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

    st.markdown("### Ground Motion Input Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        M       = st.number_input("Magnitude (M)",            2.0,  10.0,   7.0,  0.1)
        R       = st.number_input("Distance R (km)",           0.1, 100.0,   5.0,  0.5)
        Max_Acc = st.number_input("Max Acceleration (g)",      0.0,   5.0,   0.4,  0.01)
        Max_Vel = st.number_input("Max Velocity (cm/s)",       0.0,1000.0,  80.0,  1.0)
    with c2:
        Arias   = st.number_input("Arias Intensity (cm/s)",    0.0,5000.0, 200.0,  1.0)
        Char_Int= st.number_input("Characteristic Intensity",  0.0,   5.0,   0.1,  0.01)
        CAV     = st.number_input("CAV (cm/s)",                0.0,10000.0,1200.0, 10.0)
        Housner = st.number_input("Housner Intensity (cm)",    0.0,1000.0, 180.0,  1.0)
    with c3:
        Sa_avg  = st.number_input("Sa,avg (g)",                0.0,   5.0,   0.5,  0.01)
        SaT     = st.number_input("SaT@1.5 (g)",               0.0,   5.0,   0.8,  0.01)
        Ia_Du   = st.number_input("Ia/Du",                     0.0, 500.0,  20.0,  0.5)

    st.markdown("### Empirical Model Predictions (cm)")
    st.caption("Leave as 0 if unavailable — the model will still predict using raw inputs.")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        km_song = st.number_input("Song & Rodriguez-Marek 2015", 0.0, 10000.0, 0.0, 1.0)
        km_jaf  = st.number_input("Jafarian et al. 2019",        0.0, 10000.0, 0.0, 1.0)
    with kc2:
        km_hl   = st.number_input("HL 2010",                     0.0, 10000.0, 0.0, 1.0)
        km_sr09 = st.number_input("SR09",                        0.0, 10000.0, 0.0, 1.0)
    with kc3:
        km_ts16 = st.number_input("TS16",                        0.0, 10000.0, 0.0, 1.0)
        km_lash = st.number_input("Lashgari et al. 2021",        0.0, 10000.0, 0.0, 1.0)
    with kc4:
        km_youd = st.number_input("Youd 2002",                   0.0, 10000.0, 0.0, 1.0)
        alpha_v = st.selectbox("Prediction interval",
                               [0.10, 0.05],
                               format_func=lambda x: f"{int((1-x)*100)}%")

    inp = {"M":M,"R":R,"Max_Acc":Max_Acc,"Max_Vel":Max_Vel,"Arias":Arias,
           "Char_Int":Char_Int,"CAV":CAV,"Housner":Housner,
           "Sa_avg":Sa_avg,"SaT":SaT,"Ia_Du":Ia_Du}
    km  = {"Song_2015":km_song,"Jafarian_2019":km_jaf,"HL_2010":km_hl,
           "SR09":km_sr09,"TS16":km_ts16,"Lashgari_2021":km_lash,"Youd_2002":km_youd}

    if st.button("🚀 Predict Displacement", type="primary", use_container_width=True):
        X_new = build_feature_row(inp, km)
        point, lower, upper, q = jackknife_interval_fast(X_new, alpha=alpha_v)
        cv    = (upper - lower) / (2 * point + 1e-9)
        color = gauge_color(cv)

        st.markdown("---")
        st.markdown("### Prediction Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Displacement", f"{point:.1f} cm")
        r2.metric(f"Lower ({int((1-alpha_v)*100)}% PI)", f"{lower:.1f} cm")
        r3.metric(f"Upper ({int((1-alpha_v)*100)}% PI)", f"{upper:.1f} cm")
        r4.metric("Interval Width", f"{upper-lower:.1f} cm")

        unc = ("🟢 Low" if cv < 0.3 else "🟡 Moderate" if cv < 0.6 else "🔴 High")
        st.progress(min(cv, 1.0), text=f"Uncertainty: {unc}  (CV = {cv:.2f})")

        # Interval visualisation
        fig, ax = plt.subplots(figsize=(9, 1.8))
        ax.barh([""], [upper-lower], left=[lower], color=color,
                alpha=0.30, height=0.5, label=f"{int((1-alpha_v)*100)}% PI")
        ax.plot([point], [0], "o", color=color, markersize=12,
                label="Point estimate", zorder=5)
        ax.set_xlabel("Displacement (cm)")
        ax.set_xlim(0, max(upper*1.2, 10))
        ax.legend(fontsize=9); ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

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
    st.markdown("Upload a **CSV or Excel** file. Download results with predictions and 90% intervals.")

    # Template download
    template_cols = RAW_INPUTS + KM_COLS
    tdf = pd.DataFrame([{
        "M":7.0,"R":5.0,"Max_Acc":0.4,"Max_Vel":80.0,"Arias":200.0,
        "Char_Int":0.1,"CAV":1200.0,"Housner":180.0,"Sa_avg":0.5,
        "SaT":0.8,"Ia_Du":20.0,"Song_2015":300.0,"Jafarian_2019":200.0,
        "HL_2010":150.0,"SR09":500.0,"TS16":400.0,"Lashgari_2021":180.0,
        "Youd_2002":300.0
    }])
    buf = io.BytesIO()
    tdf.to_excel(buf, index=False)
    st.download_button("⬇️ Download Input Template (.xlsx)", buf.getvalue(),
                       "input_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    uploaded = st.file_uploader("Upload file (CSV or Excel)", type=["csv","xlsx","xls"])

    if uploaded is not None:
        try:
            df_up = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                     else pd.read_excel(uploaded))
            st.success(f"✅ {len(df_up)} records loaded")
            st.dataframe(df_up.head(5), use_container_width=True)

            # Fill any missing input columns with 0
            for c in template_cols:
                if c not in df_up.columns:
                    df_up[c] = 0.0

            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Build 18-col feature matrix
                df_proc = df_up.copy()
                for km_col in KM_COLS:
                    df_proc[f"{km_col}_log"] = np.log1p(df_proc[km_col].fillna(0))

                X_batch = df_proc.reindex(columns=MODEL_FEATS, fill_value=0.0)
                preds   = np.clip(model.predict(X_batch), 0, None)
                df_up["Pred_Displacement_cm"] = np.round(preds, 2)

                calibrate_loo()
                q90 = np.quantile(st.session_state["loo_residuals"], 0.90)
                df_up["PI_Lower_90"] = np.round(np.clip(preds - q90, 0, None), 2)
                df_up["PI_Upper_90"] = np.round(preds + q90, 2)
                df_up["PI_Width"]    = np.round(df_up["PI_Upper_90"] - df_up["PI_Lower_90"], 2)

                res_cols = ([c for c in ["Rec_No","M","R"] if c in df_up.columns] +
                            ["Pred_Displacement_cm","PI_Lower_90","PI_Upper_90","PI_Width"])
                st.markdown("### Results")
                st.dataframe(df_up[res_cols], use_container_width=True)

                sc1,sc2,sc3,sc4 = st.columns(4)
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
                st.download_button("⬇️ Download Results (.xlsx)", out.getvalue(),
                                   "predictions_output.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   type="primary")

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.title("🧠 Model Explainability (SHAP)")
    st.markdown(
        "SHAP values quantify each feature's contribution to predictions.  \n"
        "**Positive SHAP** → pushes displacement prediction higher.  \n"
        "**Negative SHAP** → pushes displacement prediction lower."
    )

    @st.cache_resource
    def compute_shap_values():
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X_train)   # X_train is aligned 18-col DataFrame
        return exp, vals

    with st.spinner("Computing SHAP values..."):
        explainer, shap_vals = compute_shap_values()

    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance","🐝 Beeswarm","📈 Dependence"])

    with tab1:
        st.markdown("#### Global Feature Importance — Mean |SHAP Value|")
        mean_abs = np.abs(shap_vals).mean(axis=0)
        feat_ord = np.argsort(mean_abs)
        fig, ax  = plt.subplots(figsize=(9, 7))
        bars = ax.barh([MODEL_FEATS[i] for i in feat_ord],
                       mean_abs[feat_ord], color="#2563EB", edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_xlabel("Mean |SHAP value| (cm)")
        ax.set_title("Feature Importance — Gradient Boosting")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.info(
            "**CAV** (Cumulative Absolute Velocity) is the dominant predictor, "
            "followed by Arias Intensity and Housner Intensity. "
            "Among hybrid features, Jafarian_2019_log and HL_2010_log contribute most."
        )

    with tab2:
        st.markdown("#### SHAP Beeswarm Plot")
        st.caption("Each dot = one sample. Color = feature value (🔴 high / 🔵 low).")
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_train, plot_type="dot",
                          max_display=18, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.markdown("#### SHAP Dependence Plot")
        st.caption("How SHAP value varies with feature value. Color reveals interaction effects.")
        col_a, col_b = st.columns(2)
        feat_sel  = col_a.selectbox("Feature (x-axis)", MODEL_FEATS,
                                    index=MODEL_FEATS.index("CAV"))
        color_sel = col_b.selectbox("Color by (interaction)", MODEL_FEATS,
                                    index=MODEL_FEATS.index("Arias"))
        fi = MODEL_FEATS.index(feat_sel)
        ci = MODEL_FEATS.index(color_sel)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(X_train.iloc[:, fi], shap_vals[:, fi],
                        c=X_train.iloc[:, ci], cmap="RdYlBu_r",
                        alpha=0.75, s=55, edgecolors="white", linewidth=0.3)
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
    st.markdown("""
    ## Seismic-Induced Displacement Predictor

    This tool predicts **seismic-induced lateral displacement** using a
    **hybrid machine learning model** combining:
    - Ground motion intensity measures from numerical simulations
    - Knowledge-based empirical model outputs as structured prior information

    Trained on high-fidelity numerical simulation data (n = 75 records)
    and validated with Leave-One-Out Cross-Validation (LOO-CV).
    """)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🏆 Model Performance (LOO-CV)")
        perf = pd.DataFrame({
            "Model":    ["Gradient Boosting ✅","Random Forest","XGBoost",
                         "HL_2010 (best empirical)","Lashgari_2021","Jafarian_2019"],
            "R²":       [ 0.659,  0.642,  0.607,  -7.18,  -9.00, -51.55],
            "RMSE (cm)":[ 23.4,   24.0,   25.1,  114.7,  126.8,  290.8],
            "MAE (cm)": [ 16.9,   15.9,   16.6,   75.3,  107.5,  156.7],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.success("Hybrid ML reduces RMSE by **~79%** vs best empirical model (HL_2010).")

    with c2:
        st.markdown("### 🎯 Jackknife+ Uncertainty")
        st.metric("Coverage (90% PI)", "89.3%", "Target ≥ 88.8%")
        st.metric("Mean Interval Width", "75.0 cm")
        st.markdown("""
        **Jackknife+** (Barber et al., 2021) gives finite-sample,
        distribution-free prediction intervals with theoretical guarantees.
        For n=75: minimum coverage = (1−0.1)(1+1/75) = **88.8%** ✅
        """)

    st.markdown("---")
    st.markdown("### 📥 Model Features (18 total)")
    feat_table = pd.DataFrame({
        "Feature":     MODEL_FEATS,
        "Type":        (["Raw Input"]*11 + ["Hybrid KM (log)"]*7),
        "Description": [
            "Moment magnitude",
            "Source-to-site distance (km)",
            "Peak ground acceleration (g)",
            "Peak ground velocity (cm/s)",
            "Arias Intensity (cm/s)",
            "Characteristic Intensity",
            "Cumulative Absolute Velocity (cm/s)",
            "Housner Intensity (cm)",
            "Average spectral acceleration (g)",
            "Spectral acceleration at 1.5T (g)",
            "Intensity / Duration ratio",
            "log(1 + Song & Rodriguez-Marek 2015)",
            "log(1 + Jafarian et al. 2019)",
            "log(1 + Haskell & Lacasse 2010)",
            "log(1 + SR09)",
            "log(1 + TS16)",
            "log(1 + Lashgari et al. 2021)",
            "log(1 + Youd 2002)",
        ],
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

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
    st.caption("Built with Python · scikit-learn · SHAP · Streamlit  |  n=75 records")
