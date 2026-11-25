# 🏆 US Open 2026 ML Predictor

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172B4D?style=flat)

Predicts the probability that one tennis player beats another, then simulates the full US Open draw thousands of times to estimate each player's title chances.

![Head-to-head tab](docs/head_to_head.png)

## 1. Approach

### Match model
Each match is one row comparing the two players. Features are the **difference** between Player 1 and Player 2, using only information available **before** the match:
* **Ranking:** log ATP rank and ranking points
* **Recent form (last 10 matches):** 1st serve in %, 1st-serve points won, break points saved, aces per service game, win rate
* **Fatigue:** minutes already played in the current tournament
* **Head-to-head:** previous meetings and H2H win rate
* **Physical:** height

### Tournament simulator (Monte Carlo)
Winning a Slam means winning 5 matches in a row against opponents that depend on earlier results, so there's no single prediction to make. The app plays out the whole bracket up to 5,000 times, deciding each match randomly using the model's probability, so upsets happen at a realistic rate. A player's title chance is the share of simulations they win. Simulations are vectorized: 1,000 tournaments run in well under a second.

## 2. Results

Trained on ATP matches 2023–2025, tested on the following ~1,000 matches (time-based split, both players ranked in the top 100):

| Model | Accuracy | Log loss ↓ | ROC-AUC |
|---|---|---|---|
| Coin flip | 47.8% | 0.693 | 0.500 |
| Rank only (baseline) | **67.3%** | 0.607 | 0.733 |
| Logistic Regression | 67.1% | **0.606** | 0.731 |
| XGBoost (default) | 62.0% | 0.711 | 0.678 |
| XGBoost (tuned) | 66.2% | **0.606** | 0.732 |

**Key finding:** ranking alone already captures most of the predictable signal. Tuning improved XGBoost a lot (0.711 → 0.606), but it only tied logistic regression and the rank-only baseline. Form, fatigue and H2H add little, because ranking itself summarizes recent results. Both models are well calibrated, so their probabilities are safe to use in the simulation, and the app lets you choose either.

## 3. Key Design Decisions
* **Clean before reshaping:** filtering after splitting matches into winner/loser rows had created an artificial 67/33 class imbalance.
* **One row per match, random P1/P2:** balanced classes (49/51) with no class weighting, which would distort probabilities.
* **No data leakage:** every feature uses only earlier matches (`shift(1)`), and the train/test split and cross-validation are time-ordered (`TimeSeriesSplit`).
* **Evaluated on probabilities:** log loss, Brier score and a calibration curve, not just accuracy, because the simulation uses the probabilities directly.
* **Shared feature code:** `features.py` is imported by both the notebook and the app, with a test asserting it reproduces the training data exactly (no train/serve skew).

## 4. Application Features
* **👥 Player Selection:** choose up to 32 players; seeds come from `Players.csv`.
* **📊 Tournament Prediction:** seeded bracket or random draw, title and final probabilities for every player, choice of model.
* **⚔️ Head to Head:** win probability for any two players, with their current form, fatigue history and past meetings.

## 5. Repository Structure
```
├── app.py               # Streamlit app (simulation and UI)
├── features.py          # Shared cleaning and feature engineering
├── US_Open_26.ipynb     # EDA, model training and evaluation
├── requirements.txt     # Python dependencies
├── models/
│   └── match_model.pkl  # Both trained models + feature list + test metrics
└── data/
    ├── 2023.csv … 2026.csv  # ATP match results by year
    └── Players.csv          # Seeded players for the tournament
```

## 6. Installation & Usage
```bash
git clone https://github.com/Lukevu-hub/us-open-ml-predictor.git
cd us-open-ml-predictor
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```
To retrain the models, run `US_Open_26.ipynb` from top to bottom.

## 7. Limitations & Next Steps
* Surface isn't modeled. The US Open is on hard courts, so surface-specific Elo ratings are a natural next feature.
* About 4 seasons of data; players with few matches have noisy form and H2H values.
* Injuries, withdrawals and time off between tournaments aren't modeled.

**Author:** Luke Vu