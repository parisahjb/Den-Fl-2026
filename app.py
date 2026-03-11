# ── app.py — Seismic Lateral Displacement Prediction (Three-Tier Hybrid ML)
# ── Parisa Hajibabaee | Florida Polytechnic University

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FILE_PATH  = "DATA-ML v03-Parisa.xlsx"
FC         = 0.007
Dr         = 500
g          = 9.81

RAW_FEATURES = [
    "M", "R", "Max_Acc", "Max_Vel", "Arias", "Char_Int",
    "CAV", "Housner", "Sa_avg", "SaT", "Ia_Du",
    "ru_max_ave", "Shear_strain"
]

TOP6 = ["CAV", "Arias", "Shear_strain", "Housner", "Sa_avg", "M"]

FEATURE_LABELS = {
    "M":           "Magnitude (M)",
    "R":           "Distance R (km)",
    "Max_Acc":     "Peak Ground Acceleration (g)",
    "Max_Vel":     "Peak Ground Velocity (cm/s)",
    "Arias":       "Arias Intensity (m/s)",
    "Char_Int":    "Characteristic Intensity",
    "CAV":         "Cumul. Abs. Velocity — CAV (cm/s)",
    "Housner":     "Housner Intensity (cm)",
    "Sa_avg":      "Average Spectral Accel. Sa_avg (g)",
    "SaT":         "Spectral Accel. at T — SaT (g)",
    "Ia_Du":       "Arias Intensity Duration Ia_Du",
    "ru_max_ave":  "Max Excess Pore Pressure Ratio ru",
    "Shear_strain":"Maximum Shear Strain (%)",
}

FEATURE_DEFAULTS = {
    "M": 6.5, "R": 10.0, "Max_Acc": 0.3, "Max_Vel": 30.0,
    "Arias": 1.5, "Char_Int": 0.5, "CAV": 800.0, "Housner": 80.0,
    "Sa_avg": 0.4, "SaT": 0.5, "Ia_Du": 0.8,
    "ru_max_ave": 0.7, "Shear_strain": 2.5,
}

FEATURE_RANGES = {
    "M":           (4.0, 9.0, 0.01),
    "R":           (0.1, 200.0, 0.1),
    "Max_Acc":     (0.01, 2.0, 0.001),
    "Max_Vel":     (0.1, 300.0, 0.1),
    "Arias":       (0.001, 20.0, 0.001),
    "Char_Int":    (0.001, 5.0, 0.001),
    "CAV":         (10.0, 5000.0, 1.0),
    "Housner":     (1.0, 500.0, 0.1),
    "Sa_avg":      (0.01, 3.0, 0.001),
    "SaT":         (0.01, 3.0, 0.001),
    "Ia_Du":       (0.001, 5.0, 0.001),
    "ru_max_ave":  (0.0, 1.0, 0.001),
    "Shear_strain":(0.001, 20.0, 0.001),
}

# ─────────────────────────────────────────────
# EMPIRICAL FORMULAS
# ─────────────────────────────────────────────
def compute_song2015(M, R, Max_Acc, Max_Vel, Arias, Char_Int,
                     CAV, Housner, Sa_avg, SaT, Ia_Du, ru_max_ave, Shear_strain):
    try:
        val = np.exp(
            -8.436 - 3.195*np.log(FC) - 0.346*np.log(FC)**2
            + 1.579*np.log(SaT) + 0.393*np.log(FC)*np.log(SaT)
            - 0.136*np.log(SaT)**2 + 1.542*np.log(Max_Vel) + 0.112*0.1
        )
        return max(val, 0)
    except:
        return 0.0

def compute_all_kms(row):
    M_v  = row["M"];       R_v  = row["R"]
    D    = row["Max_Acc"]; E    = row["Max_Vel"]
    F    = row["Arias"];   K    = row["SaT"]
    out  = {}
    # Song_2015
    try:
        out["Song_2015"] = np.exp(
            -8.436 - 3.195*np.log(FC) - 0.346*np.log(FC)**2
            + 1.579*np.log(K) + 0.393*np.log(FC)*np.log(K)
            - 0.136*np.log(K)**2 + 1.542*np.log(E) + 0.112*0.1)
    except: out["Song_2015"] = 0.0
    # Jafarian_2019
    try:
        A1 = ((0.41-0.34*FC)/(0.02+FC)) + ((0.002+1.42*FC)/(0.003+FC))*Dr
        A2 = ((0.24+0.381*FC)/(FC+FC)) + Dr
        A3 = ((0.55-2.77*FC)/(0.005+FC)) + ((0.55-3.28*FC)/(0.11+FC))*Dr
        A4 = ((0.55-3.28*FC)/(0.11+FC)) + Dr
        out["Jafarian_2019"] = np.exp((A1/A2)*np.log(F/100) + (A3/A4))
    except: out["Jafarian_2019"] = 0.0
    # HL_2010
    try:
        out["HL_2010"] = 10**(0.788*np.log10(F/100) - 10.166*FC
                              + 5.95*FC*np.log10(F/100) + 1.779 + 0.224)
    except: out["HL_2010"] = 0.0
    # SR09
    try:
        ratio = FC/D
        out["SR09"] = np.exp(-1.56 - 4.58*FC - 20.84*ratio**2
                             + 44.75*ratio**3 - 30.5*ratio**4
                             - 0.64*np.log(D) + 1.55*np.log(E)
                             + 0.405 + 0.524*ratio)
    except: out["SR09"] = 0.0
    # TS16
    try:
        ratio = FC/D
        out["TS16"] = np.exp(6.4 - 8.374*ratio - 0.419*ratio**2
                             + 6.366*ratio**3 - 7.031*ratio**4
                             + 0.767*np.log(D) + 0.67*np.log(F/100))
    except: out["TS16"] = 0.0
    # Lashgari_2021
    try:
        c1 = 6.31*Dr**0.06; c2 = 9.89*Dr**0.04; c3 = 1.27*Dr**0.05
        out["Lashgari_2021"] = np.exp(c1 - c2*np.exp(np.log(FC/D)/c3)
                                      + np.log(FC*g))
    except: out["Lashgari_2021"] = 0.0
    # Youd_2002
    try:
        R_dist = R_v
        out["Youd_2002"] = 10**(-16.213 + 1.532*M_v
            - 1.406*np.log10(R_dist + 10**(0.89*M_v - 5.64))
            - 0.012*R_dist + 0.338*np.log10(6*100/30.9)
            + 3.413*np.log10(100 - 0.7)
            - 0.795*np.log10(0.18 + 0.1)) * 100
    except: out["Youd_2002"] = 0.0
    return out

# ─────────────────────────────────────────────
# PINN (pure numpy, no pkl)
# ─────────────────────────────────────────────
class PhysicsInformedNN:
    def __init__(self, hidden_layers=(64,32,16), lr=0.0005,
                 epochs=600, batch_size=16, lambda_ts16=0.2, seed=42):
        self.hidden_layers = hidden_layers
        self.lr = lr; self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ts16 = lambda_ts16; self.seed = seed
        self.weights = []; self.biases = []
        self.mu = None; self.sigma = None

    def _km_ts16(self, X):
        D = X[:, RAW_FEATURES.index("Max_Acc")]
        F = X[:, RAW_FEATURES.index("Arias")]
        r = FC / np.clip(D, 1e-9, None)
        val = np.exp(6.4 - 8.374*r - 0.419*r**2 + 6.366*r**3
                     - 7.031*r**4 + 0.767*np.log(np.clip(D,1e-9,None))
                     + 0.67*np.log(np.clip(F/100,1e-9,None)))
        return np.log1p(np.clip(val, 0, None))

    def _init_weights(self, n_in, rng):
        sizes = [n_in] + list(self.hidden_layers) + [1]
        self.weights = []
        self.biases  = []
        for i in range(len(sizes)-1):
            fan_in = sizes[i]
            self.weights.append(
                rng.standard_normal((fan_in, sizes[i+1])) * np.sqrt(2/fan_in))
            self.biases.append(np.zeros(sizes[i+1]))

    def _forward(self, X):
        activations = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = np.maximum(0, z) if i < len(self.weights)-1 else z
            activations.append(a)
        return a.flatten(), activations

    def _backward(self, X, y, activations):
        n = len(y)
        y_pred = activations[-1].flatten()
        delta  = 2*(y_pred - y) / n
        grads_w = []; grads_b = []
        for i in range(len(self.weights)-1, -1, -1):
            a_in  = activations[i]
            a_out = activations[i+1]
            if i < len(self.weights)-1:
                delta = delta * (a_out.flatten() > 0)
            gw = a_in.T @ delta.reshape(-1,1)
            gb = delta.sum(axis=0)
            grads_w.insert(0, gw); grads_b.insert(0, gb)
            delta = (delta.reshape(-1,1) @ self.weights[i].T).flatten()
        return grads_w, grads_b

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        self._init_weights(Xs.shape[1], rng)
        n = len(y)
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b_idx = idx[start:start+self.batch_size]
                Xb    = Xs[b_idx]; yb = y[b_idx]
                y_pred, acts = self._forward(Xb)
                # Data loss gradients
                gw, gb = self._backward(Xb, yb, acts)
                # Physics loss (TS16)
                if self.lambda_ts16 > 0:
                    km_target = self._km_ts16(X[b_idx])
                    y_log     = np.log1p(np.clip(y_pred, 0, None))
                    p_delta   = 2*(y_log - km_target) / len(b_idx)
                    p_delta  *= 1.0 / (np.clip(y_pred, 0, None) + 1)
                    p_delta  *= self.lambda_ts16
                    pgw, pgb  = self._backward(Xb, y_pred - p_delta, acts)
                    gw = [g + pg for g, pg in zip(gw, pgw)]
                    gb = [g + pg for g, pg in zip(gb, pgb)]
                # Update with gradient clipping
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * np.clip(gw[i], -1, 1)
                    self.biases[i]  -= self.lr * np.clip(gb[i], -1, 1)
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sigma
        pred, _ = self._forward(Xs)
        return np.clip(pred, 0, None)


# ─────────────────────────────────────────────
# DATA + MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data and training models...")
def load_and_train():
    # ── Load data
    df_raw = pd.read_excel(FILE_PATH, header=3)
    col_map = {df_raw.columns[i]: name for i, name in enumerate(
        ["Rec_No"] + RAW_FEATURES + ["Displacement"])}
    df = df_raw.rename(columns=col_map).iloc[:, :15].dropna(
        subset=["Displacement"]).reset_index(drop=True)

    # ── Compute KM features
    kms = df.apply(compute_all_kms, axis=1, result_type="expand")
    for km in kms.columns:
        df[f"KM_{km}"]     = kms[km].clip(lower=0)
        df[f"KM_{km}_log"] = np.log1p(df[f"KM_{km}"])

    X_raw = df[RAW_FEATURES].values
    X_t1  = df[TOP6].values
    X_t2  = df[TOP6 + ["KM_Song_2015_log"]].values
    y     = df["Displacement"].values

    # ── LOO-CV for Jackknife+ residuals
    def get_loo_preds(model, X):
        loo = LeaveOneOut()
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            m = type(model)(**model.get_params())
            m.fit(X[tr], y[tr])
            preds[te[0]] = m.predict(X[te])[0]
        return preds

    # Tier 1: GB Top 6
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3,
         learning_rate=0.05, subsample=0.8, random_state=42)
    loo_t1 = get_loo_preds(gb, X_t1)
    gb.fit(X_t1, y)
    q_t1 = np.quantile(np.abs(y - loo_t1), 0.90)

    # Tier 2: XGBoost + Song_2015
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8,
          tree_method="hist", random_state=42, verbosity=0)
    loo_t2 = get_loo_preds(xgb, X_t2)
    xgb.fit(X_t2, y)
    q_t2 = np.quantile(np.abs(y - loo_t2), 0.90)

    # Tier 3: PINN
    pinn = PhysicsInformedNN(lambda_ts16=0.2, epochs=600)
    loo_t3 = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_raw):
        p = PhysicsInformedNN(lambda_ts16=0.2, epochs=600)
        p.fit(X_raw[tr], y[tr])
        loo_t3[te[0]] = p.predict(X_raw[te])[0]
    pinn.fit(X_raw, y)
    q_t3 = np.quantile(np.abs(y - loo_t3), 0.90)

    # ── SHAP (full data)
    explainer_t1 = shap.TreeExplainer(gb)
    sv_t1 = explainer_t1.shap_values(pd.DataFrame(X_t1, columns=TOP6))

    X_t2_df = pd.DataFrame(X_t2, columns=TOP6 + ["KM_Song_2015_log"])
    explainer_t2 = shap.PermutationExplainer(xgb.predict, X_t2_df)
    sv_t2 = explainer_t2(X_t2_df).values

    return {
        "df": df, "y": y,
        "X_t1": X_t1, "X_t2": X_t2, "X_raw": X_raw,
        "gb": gb, "xgb": xgb, "pinn": pinn,
        "q_t1": q_t1, "q_t2": q_t2, "q_t3": q_t3,
        "loo_t1": loo_t1, "loo_t2": loo_t2, "loo_t3": loo_t3,
        "sv_t1": sv_t1, "sv_t2": sv_t2,
    }


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Seismic Lateral Displacement Predictor",
    page_icon="🌍",
    layout="wide"
)

st.sidebar.image(
    "https://www.floridapoly.edu/wp-content/uploads/2019/07/FPU_Primary_Logo_RGB.png",
    width=200
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🔮 Single Prediction",
    "📊 Model Comparison",
    "🔍 Explainability",
    "ℹ️ About"
])

# Load models
data = load_and_train()
df   = data["df"]
y    = data["y"]

# ══════════════════════════════════════════════════════════════════
# PAGE 1: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Single Prediction":
    st.title("🔮 Seismic Lateral Displacement Prediction")
    st.markdown(
        "Enter the 13 seismic intensity measures below. "
        "All three model tiers run automatically and predictions are shown side by side."
    )

    st.subheader("Input Features")
    cols = st.columns(3)
    user_inputs = {}
    for i, feat in enumerate(RAW_FEATURES):
        mn, mx, step = FEATURE_RANGES[feat]
        user_inputs[feat] = cols[i % 3].number_input(
            FEATURE_LABELS[feat],
            min_value=float(mn), max_value=float(mx),
            value=float(FEATURE_DEFAULTS[feat]),
            step=float(step), format="%.4f"
        )

    if st.button("▶ Predict Displacement", type="primary"):
        row = pd.Series(user_inputs)

        # Compute Song_2015 KM for Tier 2
        song_val = compute_song2015(**user_inputs)
        song_log = np.log1p(max(song_val, 0))

        x_t1  = np.array([[user_inputs[f] for f in TOP6]])
        x_t2  = np.array([[user_inputs[f] for f in TOP6] + [song_log]])
        x_raw = np.array([[user_inputs[f] for f in RAW_FEATURES]])

        pred_t1 = float(data["gb"].predict(x_t1)[0])
        pred_t2 = float(data["xgb"].predict(x_t2)[0])
        pred_t3 = float(data["pinn"].predict(x_raw)[0])

        q1, q2, q3 = data["q_t1"], data["q_t2"], data["q_t3"]

        st.markdown("---")
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        for col, label, pred, q, color, note in [
            (c1, "Tier 1: GB Top 6\n(Baseline ML)",
             pred_t1, q1, "#2196F3", "6 SHAP-selected features"),
            (c2, "Tier 2: XGB + Song_2015\n(Knowledge-Augmented)",
             pred_t2, q2, "#FF5722", "Song_2015 auto-computed"),
            (c3, "Tier 3: PINN TS16 λ=0.2\n(Physics-Informed NN)",
             pred_t3, q3, "#4CAF50", "Physics-constrained training"),
        ]:
            col.markdown(f"**{label}**")
            col.metric("Predicted Displacement",
                       f"{pred:.1f} cm",
                       delta=f"90% PI: [{max(0,pred-q):.1f}, {pred+q:.1f}] cm")
            col.caption(note)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        preds  = [pred_t1, pred_t2, pred_t3]
        qs     = [q1, q2, q3]
        labels = ["Tier 1\nGB Top 6", "Tier 2\nXGB+Song", "Tier 3\nPINN"]
        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        bars = ax.bar(labels, preds, color=colors, alpha=0.85,
                      edgecolor="white", width=0.5)
        ax.errorbar(labels, preds, yerr=qs, fmt="none",
                    color="black", capsize=8, linewidth=2)
        for bar, p in zip(bars, preds):
            ax.text(bar.get_x() + bar.get_width()/2, p + 2,
                    f"{p:.1f} cm", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Displacement (cm)", fontsize=11)
        ax.set_title("Three-Tier Prediction Comparison\n(error bars = 90% Jackknife+ PI)",
                     fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.info(
            f"**Song_2015 auto-computed value:** {song_val:.3f} cm "
            f"(log-transformed input to Tier 2: {song_log:.4f})"
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("LOO-CV results across all model configurations. ★ = best per tier.")

    results_data = {
        "Tier": ["Empirical", "Tier 1", "Tier 1", "Tier 1",
                 "Tier 2", "Tier 3", "Tier 3", "Tier 3"],
        "Model": [
            "HL_2010 (best empirical)",
            "GB — All 13 features", "GB — Top 6 (SHAP) ★", "RF — Top 6 (SHAP)",
            "XGB — Top 6 + Song_2015 ★",
            "NN — Baseline (13 raw)", "PINN — TS16 λ=0.2 ★", "PINN — Best 3 KM",
        ],
        "R²":   [-7.18, 0.714, 0.733, 0.730, 0.738, 0.712, 0.721, 0.720],
        "RMSE (cm)": [114.7, 21.45, 20.72, 20.83, 20.52, 21.51, 21.20, 21.23],
        "MAE (cm)":  [75.3,  13.39, 13.13, 13.40, 12.57, 14.73, 14.50, 14.62],
        "Coverage (%)": ["—", "—", "89.3", "—", "89.3", "—", "89.3", "—"],
        "PI Width (cm)": ["—", "—", "56.7", "—", "56.9", "—", "69.9", "—"],
    }
    df_res = pd.DataFrame(results_data)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # RMSE bar chart (excluding empirical)
    st.subheader("RMSE Comparison (ML Models Only)")
    df_ml = df_res[df_res["Tier"] != "Empirical"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = {
        "Tier 1": "#2196F3", "Tier 2": "#FF5722", "Tier 3": "#4CAF50"
    }
    colors = [bar_colors.get(t, "#999") for t in df_ml["Tier"]]
    bars = ax.bar(range(len(df_ml)), df_ml["RMSE (cm)"],
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(df_ml)))
    ax.set_xticklabels(df_ml["Model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (cm)")
    ax.set_title("LOO-CV RMSE Across All ML Configurations")
    ax.axhline(y=20.52, color="#FF5722", linestyle="--",
               linewidth=1.2, label="Best: XGB+Song_2015 (20.52 cm)")
    ax.legend(); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, df_ml["RMSE (cm)"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.2f}", ha="center", fontsize=8)
    st.pyplot(fig); plt.close()

    # LOO-CV Predicted vs Observed
    st.subheader("LOO-CV: Predicted vs Observed")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (label, yt, yp, color) in zip(axes, [
        ("Tier 1: GB Top 6",        y, data["loo_t1"], "#2196F3"),
        ("Tier 2: XGB + Song_2015", y, data["loo_t2"], "#FF5722"),
        ("Tier 3: PINN TS16 λ=0.2", y, data["loo_t3"], "#4CAF50"),
    ]):
        r2   = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.scatter(yt, yp, alpha=0.75, s=45, color=color,
                   edgecolors="white", linewidths=0.5)
        lim = max(yt.max(), yp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="1:1")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Observed (cm)"); ax.set_ylabel("Predicted (cm)")
        ax.set_title(f"{label}\nR²={r2:.3f}  RMSE={rmse:.1f} cm",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════
# PAGE 3: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.title("🔍 Model Explainability — SHAP Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Tier 1: GB Top 6", "Tier 2: XGB + Song_2015", "CAV Dependence"
    ])

    with tab1:
        st.subheader("SHAP Feature Importance — Tier 1: GB Top 6")
        sv_t1 = data["sv_t1"]
        mean_abs_t1 = np.abs(sv_t1).mean(axis=0)
        sorted_idx  = np.argsort(mean_abs_t1)
        sorted_f    = [TOP6[i] for i in sorted_idx]
        sorted_v    = mean_abs_t1[sorted_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Bar
        axes[0].barh(range(6), sorted_v, color="#2196F3",
                     alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(6)); axes[0].set_yticklabels(sorted_f)
        axes[0].set_xlabel("Mean |SHAP value| (cm)")
        axes[0].set_title("Feature Importance", fontweight="bold")
        for i, v in enumerate(sorted_v):
            axes[0].text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
        # Beeswarm
        X_t1_df = pd.DataFrame(data["X_t1"], columns=TOP6)
        feat_order = X_t1_df.abs().mean().sort_values(ascending=True).index.tolist()
        for i, feat in enumerate(feat_order):
            vals   = sv_t1[:, TOP6.index(feat)]
            feat_v = data["X_t1"][:, TOP6.index(feat)]
            norm   = (feat_v - feat_v.min()) / (feat_v.max() - feat_v.min() + 1e-9)
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            sc = axes[1].scatter(vals, i + jitter, c=norm, cmap="coolwarm",
                                 alpha=0.7, s=25, vmin=0, vmax=1)
        axes[1].set_yticks(range(6))
        axes[1].set_yticklabels(feat_order)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("SHAP value (cm)")
        axes[1].set_title("Beeswarm Plot", fontweight="bold")
        plt.colorbar(sc, ax=axes[1], label="Feature value\n(blue=low, red=high)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "**CAV (Cumulative Absolute Velocity)** is the dominant predictor "
            "across all three models, reflecting its role as the primary "
            "measure of total seismic energy input to the soil — the key "
            "driver of pore-pressure buildup and liquefaction-induced displacement."
        )

    with tab2:
        st.subheader("SHAP Feature Importance — Tier 2: XGB + Song_2015")
        sv_t2     = data["sv_t2"]
        T2_FEATS  = TOP6 + ["KM_Song_2015_log"]
        feat_labels = ["Song_2015 (auto)" if f == "KM_Song_2015_log"
                       else f for f in T2_FEATS]
        mean_abs_t2 = np.abs(sv_t2).mean(axis=0)
        sorted_idx  = np.argsort(mean_abs_t2)
        sorted_f    = [feat_labels[i] for i in sorted_idx]
        sorted_v    = mean_abs_t2[sorted_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(range(7), sorted_v, color="#FF5722",
                     alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(7)); axes[0].set_yticklabels(sorted_f)
        axes[0].set_xlabel("Mean |SHAP value| (cm)")
        axes[0].set_title("Feature Importance", fontweight="bold")
        for i, v in enumerate(sorted_v):
            axes[0].text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
        # Beeswarm
        for i, feat in enumerate(sorted_f):
            orig_idx = feat_labels.index(feat)
            vals     = sv_t2[:, orig_idx]
            feat_v   = data["X_t2"][:, orig_idx]
            norm     = (feat_v - feat_v.min()) / (feat_v.max() - feat_v.min() + 1e-9)
            jitter   = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            sc = axes[1].scatter(vals, i + jitter, c=norm, cmap="coolwarm",
                                 alpha=0.7, s=25, vmin=0, vmax=1)
        axes[1].set_yticks(range(7)); axes[1].set_yticklabels(sorted_f)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("SHAP value (cm)")
        axes[1].set_title("Beeswarm Plot", fontweight="bold")
        plt.colorbar(sc, ax=axes[1], label="Feature value\n(blue=low, red=high)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "**Song_2015** ranks last in SHAP importance (1.25 cm) yet improves "
            "RMSE by 0.58 cm — it contributes through nonlinear interaction "
            "effects in XGBoost's ensemble structure, encoding velocity- and "
            "spectral-period-dependent scaling not captured by the six raw features."
        )

    with tab3:
        st.subheader("CAV Dependence Plot — Tier 1 vs Tier 2")
        cav_idx  = TOP6.index("CAV")
        cav_vals = data["X_t1"][:, cav_idx]
        sv_cav_t1 = data["sv_t1"][:, cav_idx]
        sv_cav_t2 = data["sv_t2"][:, cav_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(cav_vals, sv_cav_t1, alpha=0.75, s=45, color="#2196F3",
                   label="Tier 1: GB Top 6", edgecolors="white", linewidths=0.4)
        ax.scatter(cav_vals, sv_cav_t2, alpha=0.75, s=45, color="#FF5722",
                   label="Tier 2: XGB+Song", edgecolors="white",
                   linewidths=0.4, marker="s")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("CAV (cm/s)", fontsize=11)
        ax.set_ylabel("SHAP value for CAV (cm)", fontsize=11)
        ax.set_title("CAV SHAP Dependence — Tier 1 vs Tier 2", fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(
            "High CAV values (red zone, >2000 cm/s) consistently push predicted "
            "displacement upward with SHAP contributions exceeding +30 cm, "
            "while low CAV suppresses predictions below the baseline expectation. "
            "The nonlinear threshold effect is consistent across both model families."
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")

    st.markdown("""
    ### Three-Tier Hybrid ML Framework for Seismic Lateral Displacement

    This application implements a three-tier hybrid machine learning framework
    developed to predict **seismic-induced lateral displacement (cm)** from
    13 standard ground motion intensity measures. All models are trained
    on n = 75 numerical simulation records and evaluated using
    **Leave-One-Out Cross-Validation (LOO-CV)**.
    """)

    st.markdown("---")
    st.markdown("### The Three Tiers")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **🔵 Tier 1 — Baseline ML**
    - Gradient Boosting Regressor
    - Top 6 features selected by SHAP
    - R² = 0.733 | RMSE = 20.72 cm
    - Most interpretable; no formula needed
    """)
    col2.markdown("""
    **🟠 Tier 2 — Knowledge-Augmented ML**
    - XGBoost + Song_2015 (auto-computed)
    - Best point-prediction accuracy
    - R² = 0.738 | RMSE = 20.52 cm
    - Song_2015 computed internally from inputs
    """)
    col3.markdown("""
    **🟢 Tier 3 — Physics-Informed NN**
    - Neural network (64→32→16)
    - TS16 physics loss constraint (λ=0.2)
    - R² = 0.721 | RMSE = 21.20 cm
    - No formula evaluation at inference time
    """)

    st.markdown("---")
    st.markdown("### Key Findings")
    st.markdown("""
    - All three tiers converge to R² ≈ 0.72–0.74, suggesting an
      **information ceiling** for this simulation dataset
    - **CAV** is the dominant predictor across all model families
      (consensus SHAP rank = 1)
    - SHAP-guided feature selection (13 → 6 features) improves
      generalization in small-sample settings (n = 75)
    - Empirical formulas fail stand-alone (best R² = −7.18) but
      provide complementary signal when integrated into ML
    - Jackknife+ prediction intervals achieve **89.3% empirical coverage**
      at the 90% nominal level across all three tiers
    """)

    st.markdown("---")
    st.markdown("### Input Features Used")
    feat_df = pd.DataFrame({
        "Feature": list(FEATURE_LABELS.keys()),
        "Description": list(FEATURE_LABELS.values()),
        "SHAP Rank (consensus)": [6, 7, 11, 9, 2, 12, 1, 4, 5, 10, 8, 13, 3]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    **Developed by:** Dr. Parisa Hajibabaee | Florida Polytechnic University

    **Citation:** *(manuscript in preparation)*

    **Note:** All models train on startup from the dataset file.
    LOO-CV is used for all reported metrics. Jackknife+ intervals
    are evaluated on held-out LOO predictions (not in-sample).
    """)


