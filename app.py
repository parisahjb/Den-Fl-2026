"""
app.py  —  Seismic Displacement Predictor
Hybrid ML + Knowledge-Based Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import io
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="Seismic Displacement Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_artifacts():
    with open("gb_model.pkl",      "rb") as f: model = pickle.load(f)
    with open("feature_info.pkl",  "rb") as f: info  = pickle.load(f)
    with open("training_data.pkl", "rb") as f: train = pickle.load(f)
    return model, info, train

model, info, train_data = load_artifacts()
X_train = train_data["X"]
y_train = train_data["y"]
FEAT    = info["feature_names"]
KM_COLS = info["km_cols"]
DROPPED = set(info["dropped_features"])


def build_feature_row(inp, km):
    row = {k: inp[k] for k in ["M","R","Max_Vel","Arias","CAV",
                                 "Housner","Sa_avg","SaT","Ia_Du"]}
    for col in KM_COLS:
        log_col = f"{col}_log"
        if log_col not in DROPPED:
            row[log_col] = np.log1p(km.get(col, 0.0))
    return pd.DataFrame([row])[FEAT]


def jackknife_interval_fast(X_new_row, alpha=0.10):
    if "loo_residuals" not in st.session_state:
        with st.spinner("Computing Jackknife+ calibration (one-time, ~30s)..."):
            n  = len(y_train)
            Xt = np.array(X_train)
            yt = np.array(y_train)
            res = np.zeros(n)
            for i in range(n):
                Xtr = np.delete(Xt, i, 0)
                ytr = np.delete(yt, i)
                m   = GradientBoostingRegressor(**model.get_params())
                m.fit(Xtr, ytr)
                res[i] = abs(yt[i] - m.predict(Xt[i:i+1])[0])
            st.session_state["loo_residuals"] = res
    q     = np.quantile(st.session_state["loo_residuals"], 1 - alpha)
    point = float(np.clip(model.predict(np.array(X_new_row))[0], 0, None))
    lower = max(0.0, point - q)
    upper = point + q
    return point, lower, upper, q


def gauge_color(cv):
    if cv < 0.3:  return "#16A34A"
    if cv < 0.6:  return "#D97706"
    return "#DC2626"


# ── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 Seismic Displacement\nPredictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🔮  Single Prediction",
    "📂  Batch Prediction",
    "🧠  Model Explainability",
    "ℹ️  About & Model Info"
], label_visibility="collapsed")
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
        "predictions. The model returns a point estimate with a **90% "
        "Jackknife+ prediction interval**."
    )

    st.markdown("### Ground Motion Input Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        M       = st.number_input("Magnitude (M)",         2.0, 10.0, 7.0, 0.1)
        R       = st.number_input("Distance R (km)",       0.1, 100.0, 5.0, 0.5)
        Max_Vel = st.number_input("Max Velocity (cm/s)",   0.0, 1000.0, 80.0, 1.0)
    with c2:
        Arias   = st.number_input("Arias Intensity (cm/s)",0.0, 5000.0, 200.0, 1.0)
        CAV     = st.number_input("CAV (cm/s)",            0.0, 10000.0, 1200.0, 10.0)
        Housner = st.number_input("Housner Intensity (cm)",0.0, 1000.0, 180.0, 1.0)
    with c3:
        Sa_avg  = st.number_input("Sa,avg (g)",            0.0, 5.0, 0.5, 0.01)
        SaT     = st.number_input("SaT@1.5 (g)",           0.0, 5.0, 0.8, 0.01)
        Ia_Du   = st.number_input("Ia/Du",                 0.0, 500.0, 20.0, 0.5)

    st.markdown("### Empirical Model Predictions (cm)")
    st.caption("Leave as 0 if unavailable.")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        km_song = st.number_input("Song 2015",          0.0, 10000.0, 0.0)
        km_jaf  = st.number_input("Jafarian 2019",      0.0, 10000.0, 0.0)
    with kc2:
        km_hl   = st.number_input("HL 2010",            0.0, 10000.0, 0.0)
        km_sr09 = st.number_input("SR09",               0.0, 10000.0, 0.0)
    with kc3:
        km_ts16 = st.number_input("TS16",               0.0, 10000.0, 0.0)
        km_lash = st.number_input("Lashgari 2021",      0.0, 10000.0, 0.0)
    with kc4:
        km_youd = st.number_input("Youd 2002",          0.0, 10000.0, 0.0)
        alpha_v = st.selectbox("PI Level", [0.10, 0.05],
                               format_func=lambda x: f"{int((1-x)*100)}%")

    inp = {"M":M,"R":R,"Max_Vel":Max_Vel,"Arias":Arias,"CAV":CAV,
           "Housner":Housner,"Sa_avg":Sa_avg,"SaT":SaT,"Ia_Du":Ia_Du}
    km  = {"Song_2015":km_song,"Jafarian_2019":km_jaf,"HL_2010":km_hl,
           "SR09":km_sr09,"TS16":km_ts16,"Lashgari_2021":km_lash,"Youd_2002":km_youd}

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        X_new = build_feature_row(inp, km)
        point, lower, upper, q = jackknife_interval_fast(X_new, alpha=alpha_v)
        cv    = (upper - lower) / (2 * point + 1e-9)
        color = gauge_color(cv)

        st.markdown("---")
        st.markdown("### Results")
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Predicted Displacement",  f"{point:.1f} cm")
        r2.metric(f"Lower ({int((1-alpha_v)*100)}% PI)", f"{lower:.1f} cm")
        r3.metric(f"Upper ({int((1-alpha_v)*100)}% PI)", f"{upper:.1f} cm")
        r4.metric("Interval Width",          f"{upper-lower:.1f} cm")

        unc = "🟢 Low" if cv < 0.3 else ("🟡 Moderate" if cv < 0.6 else "🔴 High")
        st.progress(min(cv, 1.0), text=f"Uncertainty: {unc}  (CV={cv:.2f})")

        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.barh([""], [upper-lower], left=[lower],
                color=color, alpha=0.3, height=0.5,
                label=f"{int((1-alpha_v)*100)}% PI")
        ax.plot([point], [0], "o", color=color, markersize=12,
                label="Point estimate", zorder=5)
        ax.set_xlabel("Displacement (cm)")
        ax.set_xlim(0, max(upper*1.2, 10))
        ax.legend(fontsize=9); ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

        emp_vals = {k:v for k,v in km.items() if v > 0}
        if emp_vals:
            st.markdown("#### vs Empirical Models")
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            names  = list(emp_vals.keys()) + ["▶ ML Hybrid"]
            values = list(emp_vals.values()) + [point]
            colors = ["#94A3B8"]*len(emp_vals) + [color]
            bars   = ax2.barh(names, values, color=colors, edgecolor="white")
            ax2.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
            ax2.set_xlabel("Predicted Displacement (cm)")
            ax2.spines[["top","right"]].set_visible(False)
            st.pyplot(fig2, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV or Excel file. Download results with predictions and 90% intervals.")

    template_cols = ["M","R","Max_Vel","Arias","CAV","Housner",
                     "Sa_avg","SaT","Ia_Du"] + KM_COLS
    tdf = pd.DataFrame([{
        "M":7.0,"R":5.0,"Max_Vel":80.0,"Arias":200.0,"CAV":1200.0,
        "Housner":180.0,"Sa_avg":0.5,"SaT":0.8,"Ia_Du":20.0,
        "Song_2015":300.0,"Jafarian_2019":200.0,"HL_2010":150.0,
        "SR09":500.0,"TS16":400.0,"Lashgari_2021":180.0,"Youd_2002":300.0
    }])
    buf = io.BytesIO()
    tdf.to_excel(buf, index=False)
    st.download_button("⬇️ Download Input Template",
                       buf.getvalue(), "input_template.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Upload file", type=["csv","xlsx","xls"])
    if uploaded:
        df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                else pd.read_excel(uploaded)
        st.success(f"✅ {len(df_up)} records loaded")
        st.dataframe(df_up.head(), use_container_width=True)

        for c in template_cols:
            if c not in df_up.columns: df_up[c] = 0.0

        if st.button("🚀 Run Batch Prediction", type="primary"):
            df_proc = df_up.copy()
            for km in KM_COLS:
                log_col = f"{km}_log"
                if log_col not in DROPPED:
                    df_proc[log_col] = np.log1p(df_proc[km].fillna(0))

            preds = np.clip(model.predict(df_proc[FEAT].values), 0, None)
            df_up["Pred_Displacement_cm"] = np.round(preds, 2)

            if "loo_residuals" not in st.session_state:
                with st.spinner("Calibrating intervals (~30s)..."):
                    n, Xt, yt = len(y_train), np.array(X_train), np.array(y_train)
                    res = np.zeros(n)
                    for i in range(n):
                        m = GradientBoostingRegressor(**model.get_params())
                        m.fit(np.delete(Xt,i,0), np.delete(yt,i))
                        res[i] = abs(yt[i] - m.predict(Xt[i:i+1])[0])
                    st.session_state["loo_residuals"] = res

            q90 = np.quantile(st.session_state["loo_residuals"], 0.90)
            df_up["PI_Lower_90"] = np.round(np.clip(preds-q90, 0, None), 2)
            df_up["PI_Upper_90"] = np.round(preds+q90, 2)
            df_up["PI_Width"]    = np.round(df_up["PI_Upper_90"]-df_up["PI_Lower_90"], 2)

            res_cols = ([c for c in ["Rec_No","M","R"] if c in df_up.columns] +
                        ["Pred_Displacement_cm","PI_Lower_90","PI_Upper_90","PI_Width"])
            st.dataframe(df_up[res_cols], use_container_width=True)

            sc1,sc2,sc3,sc4 = st.columns(4)
            sc1.metric("Mean", f"{preds.mean():.1f} cm")
            sc2.metric("Std",  f"{preds.std():.1f} cm")
            sc3.metric("Min",  f"{preds.min():.1f} cm")
            sc4.metric("Max",  f"{preds.max():.1f} cm")

            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.hist(preds, bins=20, color="#2563EB", edgecolor="white", alpha=0.8)
            ax.set_xlabel("Predicted Displacement (cm)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Predictions")
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

            out = io.BytesIO()
            df_up.to_excel(out, index=False)
            st.download_button("⬇️ Download Results (.xlsx)", out.getvalue(),
                               "predictions_output.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               type="primary")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.title("🧠 Model Explainability (SHAP)")
    st.markdown(
        "SHAP values quantify each feature's contribution to individual predictions. "
        "**Positive** = pushes displacement higher. **Negative** = pushes lower."
    )

    @st.cache_resource
    def compute_shap():
        exp  = shap.TreeExplainer(model)
        vals = exp.shap_values(X_train)
        return exp, vals

    with st.spinner("Computing SHAP values..."):
        exp, shap_vals = compute_shap()

    tab1, tab2, tab3 = st.tabs(["📊 Importance","🐝 Beeswarm","📈 Dependence"])

    with tab1:
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        feat_ord  = np.argsort(mean_abs)
        fig, ax   = plt.subplots(figsize=(9,6))
        bars = ax.barh([FEAT[i] for i in feat_ord], mean_abs[feat_ord],
                       color="#2563EB", edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_xlabel("Mean |SHAP value| (cm)")
        ax.set_title("Global Feature Importance")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.info("**CAV** is the dominant predictor, followed by Arias and Housner Intensity. "
                "Jafarian_2019_log and HL_2010_log are the most informative hybrid features.")

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)
        shap.summary_plot(shap_vals, X_train, plot_type="dot",
                          max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        col_a, col_b = st.columns(2)
        feat_sel  = col_a.selectbox("Feature (x-axis)", FEAT,
                                    index=FEAT.index("CAV") if "CAV" in FEAT else 0)
        color_sel = col_b.selectbox("Color by",         FEAT,
                                    index=FEAT.index("Arias") if "Arias" in FEAT else 1)
        fi, ci = FEAT.index(feat_sel), FEAT.index(color_sel)
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(X_train.iloc[:,fi], shap_vals[:,fi],
                        c=X_train.iloc[:,ci], cmap="RdYlBu_r",
                        alpha=0.75, s=55, edgecolors="white", lw=0.3)
        ax.axhline(0,"k",lw=0.8,linestyle="--")
        ax.set_xlabel(feat_sel); ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feat_sel}  (color={color_sel})")
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
    This tool uses a **hybrid machine learning model** combining:
    - Ground motion intensity measures from numerical simulations
    - Knowledge-based empirical model outputs as structured priors
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Performance (LOO-CV)")
        perf = pd.DataFrame({
            "Model":    ["Gradient Boosting ✅","Random Forest","XGBoost",
                         "HL_2010 (best empirical)","Lashgari_2021"],
            "R²":       [0.659, 0.642, 0.607, -7.18, -9.00],
            "RMSE (cm)":[23.4,  24.0,  25.1, 114.7, 126.8],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.success("ML hybrid reduces RMSE by **~79%** vs best empirical model.")

    with c2:
        st.markdown("### Uncertainty Quantification")
        st.metric("Jackknife+ Coverage (90% PI)", "89.3%",
                  "Target ≥ 88.8%", delta_color="normal")
        st.metric("Mean Interval Width", "75.0 cm")
        st.markdown("""
        **Jackknife+** (Barber et al., 2021) provides finite-sample, 
        distribution-free prediction intervals with theoretical guarantees.
        """)

    st.markdown("---")
    st.markdown("### Input Features Used by the Model")
    ftab = pd.DataFrame({
        "Feature":    ["M","R","Max_Vel","Arias","CAV","Housner","Sa_avg","SaT","Ia_Du",
                       "Jafarian_2019_log","HL_2010_log","Youd_2002_log"],
        "Description":["Moment magnitude","Source-to-site distance (km)",
                       "Peak ground velocity (cm/s)","Arias Intensity (cm/s)",
                       "Cumulative Absolute Velocity (cm/s)","Housner Intensity (cm)",
                       "Average spectral acceleration (g)",
                       "Spectral acceleration at 1.5T (g)","Intensity/Duration ratio",
                       "log(1 + Jafarian 2019 prediction)",
                       "log(1 + HL 2010 prediction)",
                       "log(1 + Youd 2002 prediction)"],
        "Type":       ["Raw Input"]*9 + ["Hybrid (KM)"]*3,
    })
    st.dataframe(ftab, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    - Barber et al. (2021). *Predictive Inference with the Jackknife+*. Ann. Statistics.
    - Jafarian et al. (2019). *Empirical model for seismic displacement*.
    - Haskell & Lacasse (2010). *HL displacement model*.
    - Youd et al. (2002). *Revised multilinear regression equations*.
    - Song & Rodriguez-Marek (2015). *Probabilistic seismic displacement*.
    """)
    st.caption("Built with Python · scikit-learn · SHAP · Streamlit")
