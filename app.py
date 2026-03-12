# ── app.py — Seismic Lateral Displacement Prediction (Three-Tier Hybrid ML)
# ── Parisa Hajibabaee | Florida Polytechnic University

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FILE_PATH = "DATA-ML v03-Parisa.xlsx"
FC        = 0.007
Dr        = 500
g         = 9.81

RAW_FEATURES = [
    "M", "R", "Max_Acc", "Max_Vel", "Arias", "Char_Int",
    "CAV", "Housner", "Sa_avg", "SaT", "Ia_Du",
    "ru_max_ave", "Shear_strain"
]

TOP6 = ["CAV", "Arias", "Shear_strain", "Housner", "Sa_avg", "M"]

FEATURE_LABELS = {
    "M":            "Magnitude (M)",
    "R":            "Distance R (km)",
    "Max_Acc":      "Peak Ground Acceleration (g)",
    "Max_Vel":      "Peak Ground Velocity (cm/s)",
    "Arias":        "Arias Intensity (m/s)",
    "Char_Int":     "Characteristic Intensity",
    "CAV":          "Cumul. Abs. Velocity — CAV (cm/s)",
    "Housner":      "Housner Intensity (cm)",
    "Sa_avg":       "Average Spectral Accel. Sa_avg (g)",
    "SaT":          "Spectral Accel. at T — SaT (g)",
    "Ia_Du":        "Arias Intensity Duration Ia_Du",
    "ru_max_ave":   "Max Excess Pore Pressure Ratio ru",
    "Shear_strain": "Maximum Shear Strain (%)",
}

FEATURE_DEFAULTS = {
    "M": 6.5, "R": 10.0, "Max_Acc": 0.3, "Max_Vel": 30.0,
    "Arias": 1.5, "Char_Int": 0.5, "CAV": 800.0, "Housner": 80.0,
    "Sa_avg": 0.4, "SaT": 0.5, "Ia_Du": 0.8,
    "ru_max_ave": 0.7, "Shear_strain": 2.5,
}

FEATURE_RANGES = {
    "M":            (4.0,   9.0,    0.01),
    "R":            (0.1,   200.0,  0.1),
    "Max_Acc":      (0.01,  2.0,    0.001),
    "Max_Vel":      (0.1,   300.0,  0.1),
    "Arias":        (0.001, 20.0,   0.001),
    "Char_Int":     (0.001, 5.0,    0.001),
    "CAV":          (10.0,  5000.0, 1.0),
    "Housner":      (1.0,   500.0,  0.1),
    "Sa_avg":       (0.01,  3.0,    0.001),
    "SaT":          (0.01,  3.0,    0.001),
    "Ia_Du":        (0.001, 5.0,    0.001),
    "ru_max_ave":   (0.0,   1.0,    0.001),
    "Shear_strain": (0.001, 20.0,   0.001),
}

# ─────────────────────────────────────────────
# HARDCODED LOO RESULTS FROM NOTEBOOK
# q = 90th percentile of |LOO residuals| (Cell 9)
# ─────────────────────────────────────────────
Q_T1 = 28.66   # GB Top 6
Q_T2 = 28.71   # XGBoost + Song_2015
Q_T3 = 35.76   # PINN TS16 λ=0.2

LOO_METRICS = {
    "Tier 1: GB Top 6":        {"R2": 0.733, "RMSE": 20.72, "MAE": 13.13,
                                 "Bias": -0.15, "Coverage": 89.3, "PI": 56.7},
    "Tier 2: XGB + Song_2015": {"R2": 0.738, "RMSE": 20.52, "MAE": 12.57,
                                 "Bias":  0.14, "Coverage": 89.3, "PI": 56.9},
    "Tier 3: PINN TS16 λ=0.2": {"R2": 0.721, "RMSE": 21.20, "MAE": 14.50,
                                 "Bias": -1.66, "Coverage": 89.3, "PI": 69.9},
}

# ─────────────────────────────────────────────
# EMPIRICAL FORMULAS
# ─────────────────────────────────────────────
def compute_all_kms(row):
    M_v = row["M"];       R_v = row["R"]
    D   = row["Max_Acc"]; E   = row["Max_Vel"]
    F   = row["Arias"];   K   = row["SaT"]
    out = {}
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
        ratio = FC / D
        out["SR09"] = np.exp(-1.56 - 4.58*FC - 20.84*ratio**2
                             + 44.75*ratio**3 - 30.5*ratio**4
                             - 0.64*np.log(D) + 1.55*np.log(E)
                             + 0.405 + 0.524*ratio)
    except: out["SR09"] = 0.0
    # TS16
    try:
        ratio = FC / D
        out["TS16"] = np.exp(6.4 - 8.374*ratio - 0.419*ratio**2
                             + 6.366*ratio**3 - 7.031*ratio**4
                             + 0.767*np.log(D) + 0.67*np.log(F/100))
    except: out["TS16"] = 0.0
    # Lashgari_2021
    try:
        c1 = 6.31*Dr**0.06; c2 = 9.89*Dr**0.04; c3 = 1.27*Dr**0.05
        out["Lashgari_2021"] = np.exp(
            c1 - c2*np.exp(np.log(FC/D)/c3) + np.log(FC*g))
    except: out["Lashgari_2021"] = 0.0
    # Youd_2002
    try:
        out["Youd_2002"] = 10**(-16.213 + 1.532*M_v
            - 1.406*np.log10(R_v + 10**(0.89*M_v - 5.64))
            - 0.012*R_v + 0.338*np.log10(6*100/30.9)
            + 3.413*np.log10(100 - 0.7)
            - 0.795*np.log10(0.18 + 0.1)) * 100
    except: out["Youd_2002"] = 0.0
    return out

# ─────────────────────────────────────────────
# PINN (pure numpy)
# ─────────────────────────────────────────────
class PhysicsInformedNN:
    def __init__(self, hidden_layers=(64, 32, 16), lr=0.0005,
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
                     - 7.031*r**4
                     + 0.767*np.log(np.clip(D, 1e-9, None))
                     + 0.67*np.log(np.clip(F/100, 1e-9, None)))
        return np.log1p(np.clip(val, 0, None))

    def _init_weights(self, n_in, rng):
        sizes = [n_in] + list(self.hidden_layers) + [1]
        self.weights = []; self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.weights.append(
                rng.standard_normal((fan_in, sizes[i+1])) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros((1, sizes[i+1])))

    def _forward(self, X):
        # X shape: (batch, features)
        activations = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b          # (batch, out)
            a = np.maximum(0, z) if i < len(self.weights) - 1 else z
            activations.append(a)
        return a.reshape(-1), activations  # flatten output to 1D

    def _backward(self, activations, y_true):
        # y_true: 1D array of length batch
        n      = len(y_true)
        y_pred = activations[-1].reshape(-1)          # (batch,)
        # gradient of MSE w.r.t. output layer
        delta  = (2.0 / n) * (y_pred - y_true)        # (batch,)
        delta  = delta.reshape(-1, 1)                  # (batch, 1)

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        for i in range(len(self.weights) - 1, -1, -1):
            a_in  = activations[i]                     # (batch, in)
            a_out = activations[i + 1]                 # (batch, out)
            # ReLU derivative for hidden layers
            if i < len(self.weights) - 1:
                delta = delta * (a_out > 0)            # (batch, out)
            grads_w[i] = a_in.T @ delta                # (in, out)
            grads_b[i] = delta.sum(axis=0, keepdims=True)  # (1, out)
            delta = delta @ self.weights[i].T          # (batch, in)

        return grads_w, grads_b

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma                # (n, features)
        self._init_weights(Xs.shape[1], rng)
        n = len(y)

        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b_idx = idx[start:start + self.batch_size]
                Xb    = Xs[b_idx]                      # (batch, features)
                yb    = y[b_idx]                       # (batch,)

                # Forward + backward for data loss
                y_pred, acts = self._forward(Xb)
                gw, gb = self._backward(acts, yb)

                # Physics loss (TS16)
                if self.lambda_ts16 > 0:
                    km_target = self._km_ts16(X[b_idx])         # (batch,)
                    y_log     = np.log1p(np.clip(y_pred, 0, None))
                    # gradient of physics MSE w.r.t. y_pred
                    p_grad    = (2.0 * self.lambda_ts16 / len(b_idx)) * \
                                (y_log - km_target) / \
                                (np.clip(y_pred, 0, None) + 1.0)  # (batch,)
                    # backprop physics gradient as additional output delta
                    phys_delta = p_grad.reshape(-1, 1)            # (batch,1)
                    # reuse activations, only output layer delta differs
                    pgw = [None] * len(self.weights)
                    pgb = [None] * len(self.weights)
                    delta = phys_delta
                    for i in range(len(self.weights) - 1, -1, -1):
                        a_in  = acts[i]
                        a_out = acts[i + 1]
                        if i < len(self.weights) - 1:
                            delta = delta * (a_out > 0)
                        pgw[i] = a_in.T @ delta
                        pgb[i] = delta.sum(axis=0, keepdims=True)
                        delta  = delta @ self.weights[i].T
                    gw = [g + pg for g, pg in zip(gw, pgw)]
                    gb = [g + pg for g, pg in zip(gb, pgb)]

                # Update with gradient clipping
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * np.clip(gw[i], -1.0, 1.0)
                    self.biases[i]  -= self.lr * np.clip(gb[i], -1.0, 1.0)
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sigma
        pred, _ = self._forward(Xs)
        return np.clip(pred, 0, None)


# ─────────────────────────────────────────────
# DATA + MODEL TRAINING (cached — fast version)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on dataset (< 30 sec)...")
def load_and_train():
    # Load data
    df_raw  = pd.read_excel(FILE_PATH, header=3)
    col_map = {df_raw.columns[i]: name for i, name in enumerate(
        ["Rec_No"] + RAW_FEATURES + ["Displacement"])}
    df = (df_raw.rename(columns=col_map)
                .iloc[:, :15]
                .dropna(subset=["Displacement"])
                .reset_index(drop=True))

    # Compute KM features
    kms = df.apply(compute_all_kms, axis=1, result_type="expand")
    for km in kms.columns:
        df[f"KM_{km}"]     = kms[km].clip(lower=0)
        df[f"KM_{km}_log"] = np.log1p(df[f"KM_{km}"])

    X_raw = df[RAW_FEATURES].values
    X_t1  = df[TOP6].values
    X_t2  = df[TOP6 + ["KM_Song_2015_log"]].values
    y     = df["Displacement"].values

    # ── Tier 1: GB — fit on full data
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(X_t1, y)

    # LOO-CV predictions for scatter plot and residuals
    loo_t1 = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_t1):
        m = GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42)
        m.fit(X_t1[tr], y[tr])
        loo_t1[te[0]] = m.predict(X_t1[te])[0]

    # ── Tier 2: XGBoost + Song_2015 — fit on full data
    T2_FEATS  = TOP6 + ["KM_Song_2015_log"]
    X_t2_df   = pd.DataFrame(X_t2, columns=T2_FEATS)
    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, verbosity=0)
    xgb.fit(X_t2_df, y)

    # LOO-CV predictions
    loo_t2 = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X_t2):
        m = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=42, verbosity=0)
        m.fit(pd.DataFrame(X_t2[tr], columns=T2_FEATS), y[tr])
        loo_t2[te[0]] = m.predict(
            pd.DataFrame(X_t2[te], columns=T2_FEATS))[0]

    # ── Tier 3: PINN — full data fit only (LOO metrics hardcoded)
    pinn = PhysicsInformedNN(lambda_ts16=0.2, epochs=600)
    pinn.fit(X_raw, y)

    # ── SHAP for GB (TreeExplainer — fast)
    explainer_t1 = shap.TreeExplainer(gb)
    sv_t1 = explainer_t1.shap_values(
        pd.DataFrame(X_t1, columns=TOP6))

    # XGBoost feature importances (fast — no PermutationExplainer)
    xgb_importance = xgb.feature_importances_

    return {
        "df": df, "y": y,
        "X_t1": X_t1, "X_t2": X_t2, "X_raw": X_raw,
        "T2_FEATS": T2_FEATS,
        "gb": gb, "xgb": xgb, "pinn": pinn,
        "loo_t1": loo_t1, "loo_t2": loo_t2,
        "sv_t1": sv_t1,
        "xgb_importance": xgb_importance,
    }


# ─────────────────────────────────────────────
# PAGE CONFIG
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
    "📂 Batch Prediction",
    "📊 Model Comparison",
    "🔍 Explainability",
    "ℹ️ About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Framework Summary**")
st.sidebar.markdown("🔵 Tier 1: GB Top 6 — R²=0.733")
st.sidebar.markdown("🟠 Tier 2: XGB+Song — R²=0.738")
st.sidebar.markdown("🟢 Tier 3: PINN TS16 — R²=0.721")
st.sidebar.markdown("---")
st.sidebar.caption("All metrics from LOO-CV (n=75). "
                   "Jackknife+ coverage = 89.3% (all tiers).")

# Load models
data = load_and_train()
y    = data["y"]


# ══════════════════════════════════════════════════════════════════
# PAGE 1: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Single Prediction":
    st.title("🔮 Seismic Lateral Displacement Prediction")
    st.markdown(
        "Enter the 13 seismic intensity measures below. "
        "All three model tiers run automatically and predictions "
        "are shown side by side with 90% Jackknife+ prediction intervals."
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
        # Compute Song_2015 for Tier 2
        try:
            E = user_inputs["Max_Vel"]; K = user_inputs["SaT"]
            song_val = np.exp(
                -8.436 - 3.195*np.log(FC) - 0.346*np.log(FC)**2
                + 1.579*np.log(K) + 0.393*np.log(FC)*np.log(K)
                - 0.136*np.log(K)**2 + 1.542*np.log(E) + 0.112*0.1)
            song_val = max(song_val, 0)
        except:
            song_val = 0.0
        song_log = np.log1p(song_val)

        x_t1  = np.array([[user_inputs[f] for f in TOP6]])
        x_t2  = np.array([[user_inputs[f] for f in TOP6] + [song_log]])
        x_raw = np.array([[user_inputs[f] for f in RAW_FEATURES]])

        pred_t1 = float(data["gb"].predict(x_t1)[0])
        pred_t2 = float(data["xgb"].predict(
            pd.DataFrame(x_t2, columns=data["T2_FEATS"]))[0])
        pred_t3 = float(data["pinn"].predict(x_raw)[0])

        st.markdown("---")
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        for col, label, pred, q, color, note in [
            (c1, "🔵 Tier 1: GB Top 6",
             pred_t1, Q_T1, "#2196F3",
             "Baseline ML — 6 SHAP-selected features"),
            (c2, "🟠 Tier 2: XGB + Song_2015",
             pred_t2, Q_T2, "#FF5722",
             "Knowledge-Augmented — Song_2015 auto-computed"),
            (c3, "🟢 Tier 3: PINN TS16 λ=0.2",
             pred_t3, Q_T3, "#4CAF50",
             "Physics-Informed NN — TS16 constraint during training"),
        ]:
            col.markdown(f"**{label}**")
            col.metric(
                "Predicted Displacement",
                f"{pred:.1f} cm",
                delta=f"90% PI: [{max(0, pred-q):.1f}, {pred+q:.1f}] cm"
            )
            col.caption(note)

        # Bar chart with error bars
        fig, ax = plt.subplots(figsize=(8, 4))
        preds  = [pred_t1, pred_t2, pred_t3]
        qs     = [Q_T1, Q_T2, Q_T3]
        labels = ["Tier 1\nGB Top 6", "Tier 2\nXGB+Song", "Tier 3\nPINN"]
        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        bars = ax.bar(labels, preds, color=colors, alpha=0.85,
                      edgecolor="white", width=0.5)
        ax.errorbar(labels, preds, yerr=qs, fmt="none",
                    color="black", capsize=8, linewidth=2)
        for bar, p in zip(bars, preds):
            ax.text(bar.get_x() + bar.get_width()/2, p + 1,
                    f"{p:.1f} cm", ha="center",
                    fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Displacement (cm)", fontsize=11)
        ax.set_title(
            "Three-Tier Prediction Comparison\n"
            "(error bars = 90% Jackknife+ prediction interval)",
            fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            f"**Song_2015 auto-computed:** {song_val:.3f} cm  "
            f"→  log-transformed input to Tier 2: {song_log:.4f}"
        )

        # Show all 7 KM values for reference
        with st.expander("Show all 7 empirical formula outputs"):
            row = pd.Series(user_inputs)
            kms = compute_all_kms(row)
            km_df = pd.DataFrame({
                "Empirical Model": list(kms.keys()),
                "Predicted Displacement (cm)": [
                    f"{v:.2f}" for v in kms.values()],
                "Used in": [
                    "Tier 2 (best KM)", "—", "—", "—",
                    "Tier 3 physics loss", "—", "—"
                ]
            })
            st.dataframe(km_df, use_container_width=True, hide_index=True)
            st.caption(
                "Note: Empirical formulas perform poorly in isolation "
                "(best standalone R² = −7.18) but provide complementary "
                "signal when integrated into ML models.")

# ══════════════════════════════════════════════════════════════════
# PAGE 2: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV or Excel file containing multiple records. "
        "The file must include the 13 input feature columns listed below. "
        "All three model tiers will run on every row and results can be downloaded."
    )

    # ── Template download
    with st.expander("📋 Required column names (click to expand)"):
        st.markdown(
            "Your file must contain these exact column headers "
            "(order does not matter):"
        )
        st.code(", ".join(RAW_FEATURES))
        # Build a one-row template dataframe
        template_df = pd.DataFrame([FEATURE_DEFAULTS])
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download blank template (CSV)",
            data=csv_template,
            file_name="seismic_batch_template.csv",
            mime="text/csv"
        )

    # ── File upload
    uploaded = st.file_uploader(
        "Upload your file (.csv or .xlsx)",
        type=["csv", "xlsx"]
    )

    if uploaded is not None:
        # Load uploaded file
        try:
            if uploaded.name.endswith(".xlsx"):
                df_up = pd.read_excel(uploaded)
            else:
                df_up = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        # Check required columns
        missing_cols = [f for f in RAW_FEATURES if f not in df_up.columns]
        if missing_cols:
            st.error(
                f"The following required columns are missing from your file: "
                f"{', '.join(missing_cols)}"
            )
            st.stop()

        st.success(f"File loaded: {len(df_up)} records detected.")
        st.dataframe(df_up[RAW_FEATURES].head(5), use_container_width=True)

        if st.button("▶ Run Batch Prediction", type="primary"):
            with st.spinner("Running predictions on all records..."):

                results = []
                for i, row in df_up.iterrows():
                    # ── Tier 1
                    x_t1 = np.array([[row[f] for f in TOP6]])
                    pred_t1 = float(data["gb"].predict(x_t1)[0])

                    # ── Tier 2 — compute Song_2015
                    try:
                        E = row["Max_Vel"]; K = row["SaT"]
                        song_val = np.exp(
                            -8.436 - 3.195*np.log(FC) - 0.346*np.log(FC)**2
                            + 1.579*np.log(K) + 0.393*np.log(FC)*np.log(K)
                            - 0.136*np.log(K)**2 + 1.542*np.log(E) + 0.112*0.1)
                        song_val = max(song_val, 0)
                    except:
                        song_val = 0.0
                    song_log = np.log1p(song_val)
                    x_t2 = pd.DataFrame(
                        [[row[f] for f in TOP6] + [song_log]],
                        columns=data["T2_FEATS"]
                    )
                    pred_t2 = float(data["xgb"].predict(x_t2)[0])

                    # ── Tier 3
                    x_raw = np.array([[row[f] for f in RAW_FEATURES]])
                    pred_t3 = float(data["pinn"].predict(x_raw)[0])

                    results.append({
                        "Record": i + 1,
                        # Pass through any ID column if present
                        **({col: row[col]} if col in df_up.columns
                           else {} for col in ["Rec_No", "ID", "id", "Name"]
                           if col in df_up.columns),
                        # Tier 1
                        "T1_Pred (cm)":  round(pred_t1, 2),
                        "T1_PI_Lower":   round(max(0, pred_t1 - Q_T1), 2),
                        "T1_PI_Upper":   round(pred_t1 + Q_T1, 2),
                        # Tier 2
                        "T2_Pred (cm)":  round(pred_t2, 2),
                        "T2_PI_Lower":   round(max(0, pred_t2 - Q_T2), 2),
                        "T2_PI_Upper":   round(pred_t2 + Q_T2, 2),
                        # Tier 3
                        "T3_Pred (cm)":  round(pred_t3, 2),
                        "T3_PI_Lower":   round(max(0, pred_t3 - Q_T3), 2),
                        "T3_PI_Upper":   round(pred_t3 + Q_T3, 2),
                    })

            df_results = pd.DataFrame(results)

            st.markdown("---")
            st.subheader(f"Results — {len(df_results)} records")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # ── Summary statistics
            st.markdown("#### Prediction Summary")
            c1, c2, c3 = st.columns(3)
            for col, label, color in [
                (c1, "Tier 1: GB Top 6",        "#2196F3"),
                (c2, "Tier 2: XGB + Song_2015",  "#FF5722"),
                (c3, "Tier 3: PINN TS16 λ=0.2",  "#4CAF50"),
            ]:
                key = {"Tier 1: GB Top 6":        "T1_Pred (cm)",
                       "Tier 2: XGB + Song_2015":  "T2_Pred (cm)",
                       "Tier 3: PINN TS16 λ=0.2":  "T3_Pred (cm)"}[label]
                vals = df_results[key]
                col.markdown(f"**{label}**")
                col.metric("Mean prediction",  f"{vals.mean():.1f} cm")
                col.metric("Min / Max",
                           f"{vals.min():.1f} / {vals.max():.1f} cm")

            # ── Distribution plot
            fig, ax = plt.subplots(figsize=(10, 4))
            bins = 20
            ax.hist(df_results["T1_Pred (cm)"], bins=bins, alpha=0.6,
                    color="#2196F3", label="Tier 1: GB")
            ax.hist(df_results["T2_Pred (cm)"], bins=bins, alpha=0.6,
                    color="#FF5722", label="Tier 2: XGB+Song")
            ax.hist(df_results["T3_Pred (cm)"], bins=bins, alpha=0.6,
                    color="#4CAF50", label="Tier 3: PINN")
            ax.set_xlabel("Predicted Displacement (cm)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Distribution of Batch Predictions — All Three Tiers",
                         fontweight="bold")
            ax.legend(fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Download buttons
            st.markdown("#### Download Results")
            c1, c2 = st.columns(2)

            csv_out = df_results.to_csv(index=False)
            c1.download_button(
                label="⬇ Download as CSV",
                data=csv_out,
                file_name="seismic_predictions.csv",
                mime="text/csv"
            )

            # Excel download
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_results.to_excel(writer, index=False, sheet_name="Predictions")
                # Add a summary sheet
                summary = pd.DataFrame({
                    "Model": ["Tier 1: GB Top 6",
                              "Tier 2: XGB + Song_2015",
                              "Tier 3: PINN TS16 λ=0.2"],
                    "LOO-CV R²":   [0.733, 0.738, 0.721],
                    "LOO-CV RMSE": [20.72, 20.52, 21.20],
                    "PI Width (cm)":[56.7,  56.9,  69.9],
                    "Coverage (%)": [89.3,  89.3,  89.3],
                })
                summary.to_excel(writer, index=False, sheet_name="Model Summary")
            c2.download_button(
                label="⬇ Download as Excel",
                data=buffer.getvalue(),
                file_name="seismic_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.caption(
                "PI Lower/Upper = 90% Jackknife+ prediction interval bounds. "
                "T1/T2/T3 = Tier 1 / Tier 2 / Tier 3 respectively."
            )
# ══════════════════════════════════════════════════════════════════
# PAGE 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown(
        "Full LOO-CV results across all configurations. ★ = best per tier.")

    # Full results table
    results_data = {
        "Tier": [
            "Empirical",
            "Tier 1", "Tier 1", "Tier 1",
            "Tier 2",
            "Tier 3", "Tier 3", "Tier 3"
        ],
        "Model": [
            "HL_2010 (best empirical)",
            "GB — All 13 features",
            "GB — Top 6 (SHAP) ★",
            "RF — Top 6 (SHAP)",
            "XGB — Top 6 + Song_2015 ★",
            "NN — Baseline (13 raw)",
            "PINN — TS16 λ=0.2 ★",
            "PINN — Best 3 KM",
        ],
        "R²":       [-7.18, 0.714, 0.733, 0.730, 0.738, 0.712, 0.721, 0.720],
        "RMSE (cm)":[114.7, 21.45, 20.72, 20.83, 20.52, 21.51, 21.20, 21.23],
        "MAE (cm)": [75.3,  13.39, 13.13, 13.40, 12.57, 14.73, 14.50, 14.62],
        "Coverage (%)" : ["—", "—", "89.3", "—", "89.3", "—", "89.3", "—"],
        "PI Width (cm)": ["—", "—", "56.7", "—", "56.9", "—", "69.9", "—"],
    }
    df_res = pd.DataFrame(results_data)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.markdown("---")

    # RMSE bar chart ML models only
    st.subheader("RMSE Comparison — ML Models Only")
    df_ml = df_res[df_res["Tier"] != "Empirical"].copy()
    tier_colors = {
        "Tier 1": "#2196F3",
        "Tier 2": "#FF5722",
        "Tier 3": "#4CAF50"
    }
    colors = [tier_colors.get(t, "#999") for t in df_ml["Tier"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(df_ml)), df_ml["RMSE (cm)"],
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(df_ml)))
    ax.set_xticklabels(df_ml["Model"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (cm)", fontsize=11)
    ax.set_title("LOO-CV RMSE Across All ML Configurations", fontsize=12)
    ax.axhline(y=20.52, color="#FF5722", linestyle="--", linewidth=1.2,
               label="Best: XGB+Song_2015 (20.52 cm)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, df_ml["RMSE (cm)"]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.2f}", ha="center", fontsize=8)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in tier_colors.items()]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="#FF5722", linestyle="--",
                   label="Best: 20.52 cm")
    ], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # LOO-CV scatter plots — Tier 1 and Tier 2
    st.subheader("LOO-CV: Predicted vs Observed")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, yt, yp, color) in zip(axes, [
        ("Tier 1: GB Top 6",        y, data["loo_t1"], "#2196F3"),
        ("Tier 2: XGB + Song_2015", y, data["loo_t2"], "#FF5722"),
    ]):
        r2   = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        ax.scatter(yt, yp, alpha=0.75, s=45, color=color,
                   edgecolors="white", linewidths=0.5)
        lim = max(yt.max(), yp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="1:1 line")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Observed Displacement (cm)", fontsize=10)
        ax.set_ylabel("Predicted Displacement (cm)", fontsize=10)
        ax.set_title(
            f"{label}\nR²={r2:.3f}  RMSE={rmse:.2f} cm  MAE={mae:.2f} cm",
            fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # PINN stored metrics
    st.subheader("Tier 3: PINN TS16 λ=0.2 — LOO-CV Metrics (from offline run)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",       "0.721")
    c2.metric("RMSE",     "21.20 cm")
    c3.metric("MAE",      "14.50 cm")
    c4.metric("Coverage", "89.3%")
    st.caption(
        "PINN LOO-CV was computed offline (75 folds × 600 epochs). "
        "The full-data PINN is trained at startup for live predictions.")

    st.markdown("---")

    # Key findings callout
    st.subheader("Key Findings")
    st.success(
        "All three tiers converge to R² ≈ 0.72–0.74, suggesting an **information "
        "ceiling** for this simulation dataset. The 82% RMSE reduction vs. the "
        "best empirical formula (114.7 → 20.52 cm) is achieved using the same "
        "13 input features available to those formulas.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.title("🔍 Model Explainability — SHAP Analysis")

    tab1, tab2, tab3 = st.tabs([
        "🔵 Tier 1: GB Top 6",
        "🟠 Tier 2: XGB + Song_2015",
        "📈 CAV Dependence"
    ])

    # ── Tab 1: GB SHAP
    with tab1:
        st.subheader("SHAP Feature Importance — Tier 1: GB Top 6")
        sv_t1       = data["sv_t1"]
        mean_abs_t1 = np.abs(sv_t1).mean(axis=0)
        sorted_idx  = np.argsort(mean_abs_t1)
        sorted_f    = [TOP6[i] for i in sorted_idx]
        sorted_v    = mean_abs_t1[sorted_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        axes[0].barh(range(6), sorted_v, color="#2196F3",
                     alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(6))
        axes[0].set_yticklabels(sorted_f, fontsize=10)
        axes[0].set_xlabel("Mean |SHAP value| (cm)", fontsize=10)
        axes[0].set_title("Feature Importance (Mean |SHAP|)",
                          fontweight="bold")
        for i, v in enumerate(sorted_v):
            axes[0].text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=9)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Beeswarm
        for i, feat in enumerate(sorted_f):
            idx    = TOP6.index(feat)
            vals   = sv_t1[:, idx]
            feat_v = data["X_t1"][:, idx]
            norm   = ((feat_v - feat_v.min()) /
                      (feat_v.max() - feat_v.min() + 1e-9))
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            sc = axes[1].scatter(vals, i + jitter, c=norm,
                                 cmap="coolwarm", alpha=0.7,
                                 s=25, vmin=0, vmax=1)
        axes[1].set_yticks(range(6))
        axes[1].set_yticklabels(sorted_f, fontsize=10)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("SHAP value (cm)", fontsize=10)
        axes[1].set_title("Beeswarm Plot", fontweight="bold")
        plt.colorbar(sc, ax=axes[1],
                     label="Feature value (blue=low, red=high)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "**CAV** dominates all three models (consensus rank = 1), "
            "reflecting its role as the primary measure of total seismic "
            "energy input to the soil — the key driver of pore-pressure "
            "buildup and liquefaction-induced displacement. "
            "High CAV (red) consistently produces large positive SHAP "
            "values; low CAV (blue) suppresses predictions."
        )

        # Consensus ranking table
        st.subheader("Consensus SHAP Rankings (All 3 Tier 1 Models)")
        rank_df = pd.DataFrame({
            "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "Feature": [
                "CAV", "Arias", "Shear_strain", "Housner", "Sa_avg",
                "M", "R", "Ia_Du", "Max_Vel", "SaT",
                "Max_Acc", "Char_Int", "ru_max_ave"
            ],
            "GB Rank": [1, 3, 2, 4, 6, 5, 7, 11, 8, 9, 10, 13, 12],
            "RF Rank": [1, 2, 3, 4, 5, 6, 10, 7, 8, 9, 13, 11, 12],
            "XGB Rank":[1, 2, 3, 4, 5, 8, 6, 7, 10, 9, 12, 11, 13],
            "Avg Rank":[1.0,2.3,2.7,4.0,5.3,6.3,7.7,8.3,8.7,9.0,
                        11.7,11.7,12.3],
        })
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

    # ── Tab 2: XGBoost importance
    with tab2:
        st.subheader("Feature Importance — Tier 2: XGB + Song_2015")
        T2_FEATS    = data["T2_FEATS"]
        feat_labels = ["Song_2015 (auto)" if f == "KM_Song_2015_log"
                       else f for f in T2_FEATS]
        importance  = data["xgb_importance"]
        sorted_idx  = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(range(7), importance[sorted_idx],
                       color="#FF5722", alpha=0.85, edgecolor="white")
        ax.set_yticks(range(7))
        ax.set_yticklabels([feat_labels[i] for i in sorted_idx], fontsize=10)
        ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=10)
        ax.set_title("Tier 2: XGB + Song_2015 — Feature Importance",
                     fontweight="bold")
        for i, v in enumerate(importance[sorted_idx]):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "**Song_2015** ranks last in importance yet improves RMSE by "
            "0.58 cm over the Top 6 baseline. It encodes velocity- and "
            "spectral-period-dependent scaling (Max_Vel, SaT) not directly "
            "captured by the six raw features, contributing through nonlinear "
            "interaction effects in XGBoost's ensemble. Importantly, it is "
            "computed automatically — users enter only the 13 standard inputs."
        )

        st.subheader("Knowledge Model Ablation Summary")
        ablation_df = pd.DataFrame({
            "Empirical Model": [
                "Song_2015", "Jafarian_2019", "HL_2010",
                "SR09", "TS16", "Lashgari_2021", "Youd_2002", "+All 7 KM"
            ],
            "ΔRMSE: GB (cm)": [
                "−0.19 ✓", "+0.03", "+0.03",
                "+0.18", "+0.17", "−0.01 ✓", "+0.71", "+0.65"
            ],
            "ΔRMSE: RF (cm)": [
                "+0.37", "+0.07", "+0.07",
                "+0.57", "+0.06", "+0.29", "+0.26", "+0.92"
            ],
            "ΔRMSE: XGB (cm)": [
                "−0.58 ✓", "−0.49 ✓", "−0.49 ✓",
                "−0.49 ✓", "−0.37 ✓", "−0.25 ✓", "−0.03 ✓", "+0.57"
            ],
            "Key Inputs": [
                "Max_Vel, SaT", "Arias", "Arias",
                "Max_Acc, Max_Vel", "Max_Acc, Arias",
                "Max_Acc", "M, R", "All (collinear)"
            ],
        })
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
        st.caption(
            "✓ = improvement vs Top 6 baseline. "
            "XGBoost benefits from every individual KM; "
            "RF benefits from none. Adding all 7 KMs hurts every model.")

    # ── Tab 3: CAV dependence
    with tab3:
        st.subheader("CAV SHAP Dependence — Tier 1 vs Tier 2")
        st.markdown(
            "CAV (Cumulative Absolute Velocity) is the single most "
            "important feature across all models. This plot shows how "
            "its SHAP contribution changes with CAV magnitude."
        )

        cav_idx   = TOP6.index("CAV")
        cav_vals  = data["X_t1"][:, cav_idx]
        sv_cav_t1 = data["sv_t1"][:, cav_idx]

        # For Tier 2 we use XGBoost feature importances proxy
        # (no full SHAP matrix available without PermutationExplainer)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(cav_vals, sv_cav_t1, alpha=0.8, s=55,
                   color="#2196F3", label="Tier 1: GB Top 6",
                   edgecolors="white", linewidths=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("CAV (cm/s)", fontsize=11)
        ax.set_ylabel("SHAP value for CAV (cm)", fontsize=11)
        ax.set_title("CAV SHAP Dependence — Tier 1: GB Top 6",
                     fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            "High CAV values (>2000 cm/s) consistently produce SHAP "
            "contributions exceeding +30 cm, while low CAV (<500 cm/s) "
            "suppresses predictions by up to −15 cm. The nonlinear threshold "
            "effect reflects the physical mechanism: CAV above a critical level "
            "triggers sustained pore-pressure buildup driving large displacements."
        )

        # Residual comparison
        st.subheader("Residual Analysis — All Three Tiers")
        pinn_bias = -1.66
        st.markdown(f"""
        | Metric | Tier 1: GB | Tier 2: XGB+Song | Tier 3: PINN |
        |--------|-----------|-----------------|--------------|
        | Bias (cm) | −0.15 | +0.14 | −1.66 |
        | Max overpredict (cm) | 52.1 | 42.2 | 76.5 |
        | Max underpredict (cm) | 78.2 | 89.3 | 50.2 |
        | % within ±RMSE | 84.0% | 86.7% | 80.0% |

        Tier 2 achieves the lowest maximum overprediction and highest
        within-RMSE rate. The PINN shows larger overprediction in the
        high-displacement tail due to stochastic training variance.
        """)


# ══════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")

    st.markdown("""
    ### Three-Tier Hybrid ML Framework for Seismic Lateral Displacement

    This application implements a three-tier hybrid machine learning
    framework to predict **seismic-induced lateral displacement (cm)**
    from 13 standard ground motion intensity measures.
    All models are trained on **n = 75 numerical simulation records**
    and evaluated using **Leave-One-Out Cross-Validation (LOO-CV)**.
    Prediction uncertainty is quantified using **Jackknife+ intervals**
    at 90% nominal coverage.
    """)

    st.markdown("---")
    st.subheader("The Three Tiers")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **🔵 Tier 1 — Baseline ML**
    Gradient Boosting Regressor trained on the 6 most important features
    identified by SHAP consensus ranking across 3 tree models.

    - R² = 0.733 | RMSE = 20.72 cm
    - Most interpretable approach
    - No empirical formula required
    - Jackknife+ PI width = 56.7 cm
    """)
    c2.markdown("""
    **🟠 Tier 2 — Knowledge-Augmented ML**
    XGBoost trained on Top 6 features plus the Song (2015) empirical
    formula output, auto-computed from Max_Vel and SaT at inference time.

    - R² = 0.738 | RMSE = 20.52 cm
    - Best point-prediction accuracy
    - Song_2015 computed internally
    - Jackknife+ PI width = 56.9 cm
    """)
    c3.markdown("""
    **🟢 Tier 3 — Physics-Informed NN**
    Neural network (64→32→16) trained with TS16 empirical formula as a
    soft physics constraint in the loss function (λ=0.2).

    - R² = 0.721 | RMSE = 21.20 cm
    - No formula evaluation at inference
    - Physics-consistent representations
    - Jackknife+ PI width = 69.9 cm
    """)

    st.markdown("---")
    st.subheader("Key Scientific Findings")
    st.markdown("""
    1. **Information ceiling at R² ≈ 0.73** — all three hybrid approaches
       converge to the same performance band regardless of how empirical
       knowledge is introduced, suggesting irreducible aleatory uncertainty
       in the simulation dataset.

    2. **CAV is the dominant predictor** (consensus SHAP rank = 1 across
       all three model families), reflecting its role as the primary measure
       of total seismic energy input to the soil.

    3. **SHAP-guided dimensionality reduction helps** — reducing 13 to 6
       features improves LOO-CV performance in this small-sample setting
       (n = 75), consistent with the variance-bias tradeoff.

    4. **Empirical formulas fail standalone** (best R² = −7.18) but provide
       complementary signal when integrated into ML — Song_2015 improves
       XGBoost RMSE by 0.58 cm despite ranking last in SHAP importance.

    5. **XGBoost uniquely benefits** from all 7 KMs individually due to
       column subsampling; RF benefits from none; adding all 7 together
       hurts every model (multicollinearity).

    6. **Jackknife+ coverage = 89.3%** across all tiers at the 90% nominal
       level, meeting the theoretical minimum guarantee of 88.0% (= 1−α−1/n).
    """)

    st.markdown("---")
    st.subheader("Input Features")
    feat_df = pd.DataFrame({
        "Feature": list(FEATURE_LABELS.keys()),
        "Description": list(FEATURE_LABELS.values()),
        "SHAP Consensus Rank": [
            6, 7, 11, 9, 2, 12, 1, 4, 5, 10, 8, 13, 3],
        "In Top 6": [
            "✓", "", "", "", "✓", "", "✓", "✓", "✓", "", "", "", "✓"],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    **Developed by:** Dr. Parisa Hajibabaee
    Florida Polytechnic University — Department of Data Science & Business Analytics

    **Citation:** *(manuscript in preparation)*

    **Technical notes:**
    - Tree models (GB, XGBoost) are fitted on full data at startup (~10 sec each)
    - PINN is fitted on full data at startup (~15 sec); LOO metrics are from offline notebook run
    - All reported LOO-CV metrics are from the research notebook, not recomputed at runtime
    - Jackknife+ q values are hardcoded from the offline LOO run (Cell 9 of the notebook)
    """)
