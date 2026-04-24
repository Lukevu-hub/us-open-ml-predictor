# 🏆 US Open 2026 ML Predictor

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172B4D?style=flat&logo=xgboost&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Predictive%20Analytics-green)

## 1. The Idea & Methodology

In a high-stakes environment like the US Open, deterministic models often fail because they don't account for variance and physical strain. This project solves that by splitting the logic into two core engines:

### The Engine (XGBoost)
We trained an **XGBoost Classifier** on historical ATP match data (2023-2026) to predict the win probability of individual matchups. The model analyzes:
* **Player status:** Height, current ATP rank, and total ranking points.
* **Current Performance:** Performance metrics from the last 10 matches, including 1st Serve Win %, Break Points Saved, and Aces per game.
* **Fatigue Factor:** A custom feature tracking accumulated on-court minutes during the tournament. This penalizes players who had grueling previous rounds, reflecting real-world physical limits.
* **H2H Dynamics:** Historical head-to-head win rates between opponents.

### The Tournament Simulator (Monte Carlo)
Predicting a champion is more complex than just picking the best player. We use **Monte Carlo Simulations** to run the entire knockout bracket thousands of times. In each simulated match, the system "rolls a weighted dice" based on the XGBoost win probability to account for tournament variance and potential upsets.


## 2. Application Features

* **👥 Player Pool Selection:** Initialize the tournament bracket by selecting and seeding up to 32 players from the live ATP database.
* **📊 Tournament Prediction:** Run batch-vectorized Monte Carlo simulations (up to 10,000 iterations if run locally) to generate a probability-based leaderboard forecasting the champion.
* **⚔️ Head-to-Head Analysis:** Fast 1v1 matchup prediction. Compare any two players via Radar Charts (Technical Form) and Bar Charts (Fatigue levels) to see why the AI favors one over the other.



## 3. Tech Stack 

* **Language:** Python 3.13
* **Core ML:** XGBoost, Scikit-learn, Pandas, NumPy
* **Visualization:** Plotly, Seaborn
* **Interface:** Streamlit


## 4. Repository Structure 

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
## 5. Installation & Usage 

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

### Author: Luke VU    