# Den-Fl-2026
Displacement
# 🌍 Seismic Displacement Predictor

A hybrid machine learning app for predicting seismic-induced lateral displacement,
combining ground motion intensity measures with knowledge-based empirical model outputs.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📋 Features

| Page | Description |
|------|-------------|
| 🔮 Single Prediction | Enter one record → get displacement + 90% Jackknife+ interval |
| 📂 Batch Prediction  | Upload CSV/Excel → download results for all records |
| 🧠 Explainability    | SHAP feature importance, beeswarm, and dependence plots |
| ℹ️  About            | Model performance, methodology, references |

---

## 🏆 Model Performance (LOO-CV)

| Model | R² | RMSE (cm) |
|-------|----|-----------|
| **Gradient Boosting (Hybrid)** | **0.659** | **23.4** |
| Random Forest (Hybrid) | 0.642 | 24.0 |
| XGBoost (Hybrid) | 0.607 | 25.1 |
| HL_2010 (best empirical) | −7.18 | 114.7 |

> The hybrid ML model reduces RMSE by **~79%** vs the best empirical model.

---

## 🚀 How to Deploy

### Step 1 — Generate model files (run in your Jupyter notebook)

After running your full modeling pipeline (Cells 1–18), run:

```python
exec(open("save_model.py").read())
```

This generates three files:
- `gb_model.pkl`
- `feature_info.pkl`  
- `training_data.pkl`

### Step 2 — Set up GitHub repo

```
seismic-displacement-predictor/
├── app.py
├── requirements.txt
├── save_model.py
├── gb_model.pkl          ← generated from notebook
├── feature_info.pkl      ← generated from notebook
├── training_data.pkl     ← generated from notebook
└── README.md
```

### Step 3 — Deploy to Streamlit Cloud

1. Push all files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** — live in ~2 minutes ✅

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📥 Input Variables

| Variable | Description | Unit |
|----------|-------------|------|
| M | Moment magnitude | — |
| R | Source-to-site distance | km |
| Max_Vel | Peak ground velocity | cm/s |
| Arias | Arias Intensity | cm/s |
| CAV | Cumulative Absolute Velocity | cm/s |
| Housner | Housner Intensity | cm |
| Sa_avg | Average spectral acceleration | g |
| SaT | Spectral accel. at 1.5T | g |
| Ia_Du | Intensity/Duration ratio | — |
| Song_2015 | Song & Rodriguez-Marek (2015) prediction | cm |
| Jafarian_2019 | Jafarian et al. (2019) prediction | cm |
| HL_2010 | Haskell & Lacasse (2010) prediction | cm |
| SR09 | SR09 model prediction | cm |
| TS16 | TS16 model prediction | cm |
| Lashgari_2021 | Lashgari et al. (2021) prediction | cm |
| Youd_2002 | Youd (2002) prediction | cm |

---

## 📚 References

- Barber, R.F., et al. (2021). *Predictive Inference with the Jackknife+*. Annals of Statistics.
- Jafarian, Y., et al. (2019). Empirical seismic displacement model.
- Haskell & Lacasse (2010). HL displacement model.
- Youd, T.L., et al. (2002). Revised multilinear regression equations.
- Song & Rodriguez-Marek (2015). Probabilistic seismic displacement model.
