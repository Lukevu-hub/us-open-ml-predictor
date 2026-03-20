# 🏆 US Open 2026 AI Predictor

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172B4D?style=flat&logo=xgboost&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Predictive%20Analytics-green)

An end-to-end Machine Learning web application designed to predict the outcomes of the US Open tennis tournament. The system combines **XGBoost** for match-level probability predictions and **Monte Carlo Simulations** for tournament-level bracket forecasting.

## 1. Core Methodology

This project treats sports prediction not as deterministic choices, but as probability distributions.

### Match Prediction (XGBoost)
The system uses an XGBoost Classifier trained on historical ATP match data (2023-2026). The model maps a feature matrix $X \in \mathbb{R}^m$ to a probability output $P(\text{Player A wins})$.
Core features include:
* **Physical Attributes & Ranking:** Height, ATP Rank, Rank Points.
* **Tournament Fatigue:** Penalizing players with high accumulated minutes in the current tournament.
* **Recent Form:** A rolling average of the last 10 matches (1st Serve Win %, Break Points Save %, Ace per Game).
* **H2H Matrix:** Historical Head-to-Head win rates.

### Tournament Simulation (Monte Carlo)
Predicting a tournament winner requires accounting for upsets (variance).
* The system simulates the entire knockout bracket thousands of times.
* For each match, the model generates a win probability (e.g., 70%). The simulation then "rolls a weighted dice" to determine who advances.
* This Stochastic Modeling approach, based on the Law of Large Numbers, outputs the actual probability distribution of a player lifting the trophy.

## 2. Repository Structure 

```
├── app.py                 # Main Streamlit web application
├── US_Open_26.ipynb       # Jupyter Notebook for EDA, Data Cleaning & Model Training
├── requirements.txt       # Python dependencies
└── data/                  # Dataset directory
    ├── 2023.csv           # Historical ATP matches 2023
    ├── 2024.csv           # Historical ATP matches 2024    
    ├── 2025.csv           # Historical ATP matches 2025
    ├── 2026.csv           # Historical ATP matches 2026
    ├── Players.csv        # Seeded players list for the current tournament
    └── formatted-data.csv # Processed dataset (Long format)
```

## 3. Installation & Usage 

### Clone the repository

Bash

```
git clone https://github.com/Lukevu-hub/us-open-ml-predictor.git
cd us-open-ml-predictor
```

### Set up Virtual Environment 

Bash

```
python -m venv venv
source venv/bin/activate  
```

### Install Dependencies

Bash

```
pip install -r requirements.txt
```

### Run the Application

Bash

```
streamlit run app.py
```

## 4.  Application Features 
1.  **👥 Player Selection:** Initialize the tournament bracket by selecting up to 32 players from the database.

2.  **📊 Tournament Prediction:** Run batch-vectorized Monte Carlo simulations (up to 10,000 iterations) to forecast the champion.

3.  **⚔️ Head to Head Prediction:** Fast 1v1 matchup analysis.

##### Author: Luke VU    