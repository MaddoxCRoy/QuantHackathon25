# QUANTHACKATHON25

This project is designed to analyze **market regimes, predict market states (Bull, Bear, Static), and implement a Kelly Criterion-based trading strategy**. The project is divided into **three questions**, each addressing a different aspect of market analysis.

---

## 🚀 Features
- **Market Regime Visualization** (Question 1)
- **Machine Learning for Market Prediction** (Question 2) 
- **Kelly Criterion-Based Trading Strategy** (Question 3) 
- **Portfolio Performance Analysis** (Win Rate, Sharpe Ratio, Drawdowns)

---

## 🛠️ Installation & Setup

 - **Clone the Repository**
 - ```sh
 - git clone https://github.com/your-username/QuantHackathon25.git
 - cd QuantHackathon25

---

## Project Structure

# Data Files (`/Data`)
- `market_predictions.csv` - Machine Learning-based market state predictions.
- `MergedData_with_predictions.csv` - Merged dataset with prrediction estimates.
- `MergedData.csv` - Cleaned historical market data.
- `Prob_Data.csv` - Probability-related features for market classification.
- `Quantathon_Data_2025.csv` - Original dataset.
- `SP500.csv` - Historical S&P 500 price data.
- `SP2019-2022.csv` - S&P 500 data filtered for the trading period (2019-2022).
- `trading_results_2019_2022.csv` - Backtest results from 2019-2022 trading strategy.

# Question 1: Market Regime Analysis
- `regimes_visualized.py` - Visualizes different market regimes using historical data.

# Question 2: Machine Learning for Market Prediction
- Part 1:
  Accuracy of Calculated Probabilities using 2% Threshold:

  Classification
  False Static      448
  False Positive    112
  True Positive      56
  True Static        41
  False Negative     23
  True Negative       1
  Name: count, dtype: int64

  Overall classification accuracy: 14.39%

- `Data_Parser.py` - Parses and cleans raw market data.
- `MarketPredictionPlot.py` - Visualizes machine learning-based market predictions.
- `RandomForest.py` - Trains a Random Forest model to classify market states.

# Question 3: Trading Strategy Using Kelly Criterion
- `KellyCriterionBackTest.py` - Backtests Kelly Criterion-based trading strategy.
- `our_strategy.py` - Executes trading decisions based on predictions.

# Additional Files
- `README.md` - Project documentation.

## Contributers
 -Jack Huynh
 -Maddox Roy
 -Charles Tirendi
 -Ryan Jackman
 -Pranshu Shrivastava
