import pandas as pd
import numpy as np

# Load the market predictions
df = pd.read_csv("Data/market_predictions.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Define the trading period (2019-2022)
start_date = "2019-01-01"
end_date = "2022-12-31"

# Define initial trading parameters
initial_capital = 10000  # Starting amount
capital = initial_capital
risk_per_trade = 0.02  # Risk 2% of capital per trade
win_rate = 0.55  # Assumed probability of winning
reward_risk_ratio = 2  # Expected reward-to-risk ratio

# Calculate Kelly Criterion fraction
p = win_rate
q = 1 - win_rate
kelly_fraction = (p - q) / reward_risk_ratio  # Kelly Criterion formula

# Filter the dataset for trading period
trading_data = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

# Trading logic based on predicted market states
positions = []  # Store trades

for i in range(1, len(trading_data)):
    predicted_market = trading_data.iloc[i]["Predicted_Market"]
    prev_price = trading_data.iloc[i-1]["S&P500"]
    current_price = trading_data.iloc[i]["S&P500"]
    
    # Calculate position size using Kelly Criterion
    position_size = capital * kelly_fraction
    
    # Trading rules based on market state
    if predicted_market == "Bull":
        # Buy (Go Long)
        profit_loss = (current_price - prev_price) / prev_price
        capital += position_size * profit_loss
        positions.append(("Buy", trading_data.iloc[i]["Date"], current_price, capital))

    elif predicted_market == "Bear":
        # Sell (Go Short)
        profit_loss = (prev_price - current_price) / prev_price  # Reverse profit calculation
        capital += position_size * profit_loss
        positions.append(("Sell", trading_data.iloc[i]["Date"], current_price, capital))

    elif predicted_market == "Static":
        # No trade
        positions.append(("Hold", trading_data.iloc[i]["Date"], current_price, capital))

# Convert results into a DataFrame
trades_df = pd.DataFrame(positions, columns=["Action", "Date", "Price", "Capital"])

# Save trade results to CSV
trades_df.to_csv("Data/trading_results_2019_2022.csv", index=False)

# Display final capital
print(f"Final Capital (2019-2022): ${capital:.2f}")
print("Trading results saved to 'Data/trading_results_2019_2022.csv'")

import pandas as pd
import numpy as np

# Load trading results
trades_df = pd.read_csv("Data/trading_results_2019_2022.csv")
trades_df["Date"] = pd.to_datetime(trades_df["Date"])

# Calculate daily returns
trades_df["Returns"] = trades_df["Capital"].pct_change()

# Portfolio Statistics
total_return = trades_df["Capital"].iloc[-1] / trades_df["Capital"].iloc[0] - 1
annualized_return = (1 + total_return) ** (1 / ((trades_df["Date"].iloc[-1] - trades_df["Date"].iloc[0]).days / 365)) - 1
annualized_volatility = trades_df["Returns"].std() * np.sqrt(252)

# Sharpe Ratio (assumes risk-free rate is 0%)
sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

# Maximum Drawdown (Worst peak-to-trough loss)
cumulative_capital = trades_df["Capital"].cummax()
drawdowns = (trades_df["Capital"] - cumulative_capital) / cumulative_capital
max_drawdown = drawdowns.min()

# Print Portfolio Statistics
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

