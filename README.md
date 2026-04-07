# Championship Football Match Outcome Predictor

A machine learning project that predicts the outcome of English Championship football matches (Home Win / Draw / Away Win) using historical match data and engineered features.

---

## Project Overview

This project follows a professional ML pipeline to predict football match outcomes for the English Championship — England's second division. The goal is to predict whether a match will result in a Home Win, Away Win, or Draw before kickoff using only pre-match information.

---

## Dataset

- **Source:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
- **League:** English Championship (Division 1 / Second Tier)
- **Seasons:** 2016/17 → 2023/24 (Training) | 2024/25 (Testing)
- **Total Matches:** 4,415 matches across 8 seasons

---

## Project Pipeline

### 1. Data Collection
Downloaded historical Championship match data from football-data.co.uk covering 8 seasons. Each season contains match results, in-game statistics, and betting odds.

### 2. Data Cleaning
- Combined all 8 season CSV files into one dataset
- Removed irrelevant columns (individual bookmaker odds, Asian handicap etc.)
- Fixed date formatting and sorted matches chronologically
- Dropped 1 incomplete match record
- Removed odds columns due to 37.5% missing data in older seasons

### 3. Feature Engineering
Built pre-match features using only information available before kickoff:

**Form Features (Last 5 Matches)**
- Points earned by home and away team
- Goals scored and conceded
- Goal difference

**Rolling Match Stats (Last 5 Matches)**
- Average shots and shots on target
- Average corners
- Average fouls and yellow cards
- Average half time goals

**Elo Rating**
- Dynamic team strength score that updates after every match
- Captures overall team quality and strength of opposition faced
- Elo difference between home and away team

### 4. Feature Selection
Used Recursive Feature Elimination with Cross Validation (RFECV) — Backward Wrapper method — with Random Forest as the base estimator and 10-fold cross validation to identify the optimal feature set.

Result: All 20 engineered features were selected as meaningful predictors.

### 5. Model Training
Evaluated two machine learning models:

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | 45.30% |
| XGBoost (8 features) | 44.39% |
| XGBoost (20 features) | 44.28% |
| XGBoost + Elo Rating | 44.03% |

### 6. Evaluation
- Used 10-fold Stratified Cross Validation for reliable accuracy estimation
- Evaluated using Accuracy, Precision, Recall and F1 Score per class

---

## Key Findings

- Championship football is extremely difficult to predict due to its highly competitive nature — any team can beat any team on a given day
- Pre-match prediction ceiling for 3-outcome football prediction is approximately 55-60% even with perfect data
- High accuracy papers (65%+) typically use in-match statistics (half time goals, live shots) which are unavailable before kickoff — this constitutes data leakage for true pre-match prediction
- Our model achieves ~44% accuracy which is 11 percentage points above random guessing (33%) and slightly above always predicting Home Win (43%)
- Draws remain the hardest outcome to predict across all models

---

## Tech Stack

- **Platform:** Google Colab
- **Language:** Python 3
- **Libraries:**
  - Pandas — data manipulation
  - NumPy — numerical computation
  - Scikit-learn — feature selection and evaluation
  - XGBoost — gradient boosting model
  - Seaborn / Matplotlib — data visualization
  - TQDM — progress tracking

---

## Research References

1. Supervised machine learning classification using Random Forest, XGBoost and C5.0 with backward wrapper feature selection — EPL match prediction
2. Knowledge Discovery in Databases framework using ANN and Logistic Regression for EPL prediction — Igiri Chinwe Peace and Enoch Okechukwu Nwachukwu

---

## Project Structure
Championship-Football-Prediction/
├── Championship.ipynb    ← Main notebook with full pipeline
├── 16 17.csv            ← Season data files
├── 17 18.csv
├── 18 19.csv
├── 19 20.csv
├── 20 21.csv
├── 21 22.csv
├── 22 23.csv
├── 23 24.csv
└── README.md

---

## Future Improvements

- Add home and away specific form features separately
- Incorporate league table position at time of each match
- Explore binary prediction (Win vs Loss) to improve accuracy
- Add player availability and injury data
- Try ensemble methods combining multiple models
- Experiment with Neural Networks with larger datasets

---

## Author

**Mahesh** — Aspiring ML/LLM Engineer based in Chennai, India

Building a portfolio of AI and ML projects while transitioning into the LLM/GenAI space.

---

## License

This project is for educational and portfolio purposes.
