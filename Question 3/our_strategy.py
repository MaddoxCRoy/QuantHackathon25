import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# 1) Load Base Market Data and Market Predictions

df_base = pd.read_csv("MergedData.csv", parse_dates=["Date"])
df_base = df_base.sort_values("Date")

df_pred = pd.read_csv("market_predictions.csv", parse_dates=["Date"])
df_pred = df_pred.sort_values("Date")


# 2) Merge the Predictions into the Base Data

df_merged = pd.merge(df_base, 
                     df_pred[["Date", "Predicted_Market_Encoded"]], 
                     on="Date", 
                     how="left")
df_merged.to_csv("MergedData_with_predictions.csv", index=False)


# 3) Filter for Entire Period: 2008-01-01 to 2024-12-31

start_date = pd.Timestamp("2008-01-01")
end_date   = pd.Timestamp("2024-12-31")
df_full = df_merged[(df_merged["Date"] >= start_date) & (df_merged["Date"] <= end_date)].copy()


# 4) Calculate Basic Features and New Signals

df_full["Daily_Returns"] = df_full["S&P500"].pct_change()

TARGET_VOL = 0.15
df_full["Realized_Volatility"] = df_full["Daily_Returns"].rolling(window=20).std() * (252**0.5)
df_full["Scaling_Factor"] = TARGET_VOL / df_full["Realized_Volatility"]
df_full["Scaling_Factor"] = df_full["Scaling_Factor"].fillna(1).clip(upper=1)

df_full["Bond_Yield_Trend"] = df_full["Bond_Rate"].pct_change(periods=20)

df_full["S&P500_LogReturn"] = np.log(df_full["S&P500"] / df_full["S&P500"].shift(1))
df_full["Cum_LogReturn_20"] = df_full["S&P500_LogReturn"].rolling(window=20).sum()

def shannon_entropy(x, bins=20):
    hist, _ = np.histogram(x, bins=bins)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log(p))
df_full["Entropy"] = df_full["Daily_Returns"].rolling(window=20).apply(lambda x: shannon_entropy(x, bins=20), raw=True) - 1

df_full.dropna(inplace=True)


# 5) Create Logistic Regression Target and Train the Model

# Define target signal based on next day S&P500 return:
# if next return > 0.1% -> buy (1), if < -0.1% -> sell (-1), else hold (0)
df_full["Next_Return"] = df_full["S&P500"].pct_change().shift(-1)
def label_signal(x):
    if x > 0.001:
        return 1
    elif x < -0.001:
        return -1
    else:
        return 0
df_full["LR_Signal"] = df_full["Next_Return"].apply(label_signal)

# Use data before 2019 for training
train_df = df_full[df_full["Date"] < pd.Timestamp("2019-01-01")]
# Features for logistic regression
features = ["Daily_Returns", "Realized_Volatility", "Bond_Yield_Trend", "Cum_LogReturn_20", "Entropy"]
X_train = train_df[features]
y_train = train_df["LR_Signal"]

lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
lr_model.fit(X_train, y_train)


# 6) Define Base Allocation Rules (Regime-Based)

def allocation_for_regime(market_regime):
    # market_regime: 0 = Bear, 1 = Bull, 2 = Static.
    if market_regime == 1:       # Bull Market
        return 0.90, 0.10      # 85% stocks, 15% bonds
    elif market_regime == 0:     # Bear Market
        return 0.20, 0.80       # 20% stocks, 80% bonds
    elif market_regime == 2:     # Static Market
        return 0.60, 0.40      # 60% stocks, 40% bonds
    else:
        return 0.60, 0.40


# 7) Initialize Portfolio, Trailing Stop Tracking, and Trade Logging

initial_cash = 100000
positions = {
    'stock1_shares': 0,
    'cash': initial_cash
}
max_price = None  # For trailing stop loss

portfolio_value = []
dates = []

# For trade logging from both regime trades and logistic regression signals:
buy_dates = []
buy_prices = []
sell_dates = []
sell_prices = []
lr_buy_dates = []
lr_buy_prices = []
lr_sell_dates = []
lr_sell_prices = []

sim_start = pd.Timestamp('2019-01-01')
sim_end = pd.Timestamp('2022-12-31')
sim_df = df_full[(df_full["Date"] >= sim_start) & (df_full["Date"] <= sim_end)].copy()

# Rebalance every 20 trading days
for idx, row in sim_df.iterrows():
    current_date = row["Date"]
    sp_price = row["S&P500"]
    bond_rate = row["Bond_Rate"]
    predicted_regime = row["Predicted_Market_Encoded"]
    scaling_factor = row["Scaling_Factor"]
    bond_yield_trend = row["Bond_Yield_Trend"] if not pd.isna(row["Bond_Yield_Trend"]) else 0
    cum_log20 = row["Cum_LogReturn_20"]
    entropy_val = row["Entropy"]

    # Update portfolio value
    total_value = positions['cash'] + positions['stock1_shares'] * sp_price

    # Trailing Stop Loss
    if positions['stock1_shares'] > 0:
        if max_price is None:
            max_price = sp_price
        else:
            max_price = max(max_price, sp_price)
        trailing_stop = max_price * 0.90  # 10% below maximum
        if sp_price < trailing_stop:
            proceeds = positions['stock1_shares'] * sp_price
            positions['cash'] += proceeds
            positions['stock1_shares'] = 0
            sell_dates.append(current_date)
            sell_prices.append(sp_price)
            max_price = None

    # LOGISTIC REGRESSION SIGNAL TRADING

    X_current = row[features].to_frame().T
    lr_signal = lr_model.predict(X_current)[0]
    # Calculate current stock allocation fraction
    total_value = positions['cash'] + positions['stock1_shares'] * sp_price
    current_stock_value = positions['stock1_shares'] * sp_price
    current_stock_frac = current_stock_value / total_value if total_value > 0 else 0

    if lr_signal == 1:  # Buy signal: increase stock by 5% of portfolio value if below 95% in stocks.
        if current_stock_frac < 0.95:
            additional_dollar = 0.05 * total_value
            max_possible_increase = (0.95 - current_stock_frac) * total_value
            additional_dollar = min(additional_dollar, max_possible_increase)
            shares_to_buy_lr = additional_dollar / sp_price
            positions['stock1_shares'] += shares_to_buy_lr
            positions['cash'] -= additional_dollar
            lr_buy_dates.append(current_date)
            lr_buy_prices.append(sp_price)
    elif lr_signal == -1:  # Sell signal: decrease stock by 5% of portfolio value if holding stocks.
        if current_stock_frac > 0:
            sell_dollar = 0.05 * total_value
            max_possible_sell = current_stock_value
            sell_dollar = min(sell_dollar, max_possible_sell)
            shares_to_sell_lr = sell_dollar / sp_price
            positions['stock1_shares'] -= shares_to_sell_lr
            positions['cash'] += sell_dollar
            lr_sell_dates.append(current_date)
            lr_sell_prices.append(sp_price)



    # Rebalance monthly (every 20 trading days) using regime and other signals.

    if idx % 20 == 0:
        base_stock_alloc, base_bond_alloc = allocation_for_regime(predicted_regime)
        # Adjust for Bond Yield Trend.
        if bond_yield_trend > 0.05:
            base_stock_alloc = max(base_stock_alloc - 0.10, 0)
        elif bond_yield_trend < -0.05:
            base_stock_alloc = min(base_stock_alloc + 0.05, 1)
        # Adjust for Explosiveness.
        if cum_log20 > 0.05:
            base_stock_alloc *= 0.80
            base_bond_alloc *= 1.2
        # Adjust for Entropy.
        if entropy_val < 1.0:
            base_stock_alloc += 0.10
            base_bond_alloc -= 0.10
        adjusted_stock_alloc = min(max(base_stock_alloc, 0), 0.95)
        effective_stock_alloc = adjusted_stock_alloc * scaling_factor
        effective_bond_alloc = 1 - effective_stock_alloc

        total_value = positions['cash'] + positions['stock1_shares'] * sp_price
        desired_stock_value = total_value * effective_stock_alloc
        current_stock_value = positions['stock1_shares'] * sp_price

        if current_stock_value < desired_stock_value:
            diff = desired_stock_value - current_stock_value
            shares_to_buy = diff / sp_price
            if positions['stock1_shares'] == 0:
                max_price = sp_price
            positions['stock1_shares'] += shares_to_buy
            positions['cash'] -= diff
            buy_dates.append(current_date)
            buy_prices.append(sp_price)
        elif current_stock_value > desired_stock_value:
            diff = current_stock_value - desired_stock_value
            shares_to_sell = diff / sp_price
            positions['stock1_shares'] -= shares_to_sell
            positions['cash'] += diff
            sell_dates.append(current_date)
            sell_prices.append(sp_price)
            if positions['stock1_shares'] < 1e-6:
                positions['stock1_shares'] = 0
                max_price = None


    # Bond Portion: Accrue Daily Interest (252 trading days per year)

    daily_bond_rate = (bond_rate / 100) / 252.0
    positions['cash'] *= (1 + daily_bond_rate)

    total_value = positions['cash'] + positions['stock1_shares'] * sp_price
    portfolio_value.append(total_value)
    dates.append(current_date)


# 9) Calculate Portfolio Statistics

sp500_prices = sim_df["S&P500"]
sp500_daily_returns = sp500_prices.pct_change() * 100  # as percentages

# Remove the first NaN value
sp500_daily_returns = sp500_daily_returns.dropna()
total_return_sp500 = (np.prod(1 + sp500_daily_returns.values/100) - 1) * 100

# Annualized mean return and standard deviation assuming 252 trading days
sp500_mean_daily_return = np.mean(sp500_daily_returns)
sp500_std_daily_return = np.std(sp500_daily_returns)

sp500_mean_return_annual = sp500_mean_daily_return * 252
sp500_std_return_annual = sp500_std_daily_return * np.sqrt(252)

total_return_sp500 = (np.prod(1 + sp500_daily_returns.values/100) - 1) * 100

# Use the same risk-free rate as before (mean of Bond_Rate in df_base)
risk_free_rate = np.mean(df_base["Bond_Rate"])

# Calculate Sharpe Ratio for the S&P500 (annualized)
sharpe_ratio_sp500 = (sp500_mean_return_annual - risk_free_rate) / sp500_std_return_annual \
    if sp500_std_return_annual != 0 else np.nan

# Assuming 'df' is your DataFrame with the S&P500 data for the desired period
sp500_prices = sim_df["S&P500"].values

# Compute the running maximum of the S&P500 prices
running_max = np.maximum.accumulate(sp500_prices)

# Calculate drawdowns at each time step (as a fraction)
drawdowns = (sp500_prices - running_max) / running_max

# The maximum drawdown is the most negative drawdown value
max_drawdown = drawdowns.min()

print(f"S&P500 Maximum Drawdown: {max_drawdown*100:.2f}%")

print(f"S&P500 Total Return: {total_return_sp500:.2f}%")
print(f"S&P500 Annualized Mean Return: {sp500_mean_return_annual:.2f}%")
print(f"S&P500 Annualized Standard Deviation: {sp500_std_return_annual:.2f}%")
print(f"S&P500 Sharpe Ratio: {sharpe_ratio_sp500:.2f}")
print(f"S&P500 Buy and Hold Return: {(total_return_sp500 ):.2f}%")

portfolio_returns = [ (current - previous) / previous * 100 
                      for previous, current in zip(portfolio_value[:-1], portfolio_value[1:]) ]
total_return = (np.prod(1 + np.array(portfolio_returns)/100) - 1) * 100
portfolio_mean_returns = np.mean(portfolio_returns) * 252
portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
risk_free_rate = np.mean(df_base['Bond_Rate'])
sharpe_ratio = (portfolio_mean_returns - risk_free_rate) / portfolio_std_percentage if portfolio_std_percentage != 0 else np.nan

print(f"Strategy Starting Value: ${portfolio_value[0]:.2f}")
print(f"Final Strategy Value: ${portfolio_value[-1]:.2f}")
print(f"Strategy Total Return: {total_return:.2f}%")
print(f"Strategy Annualized Standard Deviation: {portfolio_std_percentage:.2f}%")
print(f"Strategy Mean Annual Return: {portfolio_mean_returns:.2f}%")
print(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")

covariance = np.cov(portfolio_returns, sp500_daily_returns)[0, 1]
variance_benchmark = np.var(sp500_daily_returns)
beta = covariance / variance_benchmark

# Calculate Maximum Drawdown:
running_max = np.maximum.accumulate(portfolio_value)
drawdowns = (portfolio_value - running_max) / running_max
max_drawdown = drawdowns.min()  # This will be a negative value

# Calculate Value at Risk (VaR) at 95% confidence level:
# We compute the 5th percentile of daily returns.
VaR_95 = np.percentile(portfolio_returns, 5)  # in decimals, e.g., -0.02 means -2%

# Convert beta, max_drawdown, and VaR into more interpretable forms:
print(f"Beta of the strategy: {beta:.2f}")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
print(f"95% Daily VaR: {VaR_95:.2f}%")

import numpy as np

# Ensure portfolio_returns is a NumPy array
portfolio_returns_array = np.array(portfolio_returns)

# Ensure that sp500_daily_returns (from sim_df) is aligned with your simulation period
# (Assuming sim_df and portfolio_returns cover the same dates)
sp500_daily_returns_array = sp500_daily_returns.values

# Compute the correlation coefficient matrix
corr_matrix = np.corrcoef(portfolio_returns_array, sp500_daily_returns_array)
correlation = corr_matrix[0, 1]

print(f"Correlation between strategy returns and S&P500 returns: {correlation:.2f}")



# 10) Plot the Equity Curve and Trade Signals

plt.figure(figsize=(12,6))
plt.plot(dates, portfolio_value, label="Equity Curve", color='green')
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Equity Curve")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(sim_df["Date"], sim_df["S&P500"], label='S&P500', color='blue')
plt.scatter(buy_dates, buy_prices, marker='o', color='green', s=50, label="Buy (Rebalance)")
plt.scatter(sell_dates, sell_prices, marker='o', color='red', s=50, label="Sell (Rebalance)")
plt.scatter(lr_buy_dates, lr_buy_prices, marker='^', color='lime', s=70, label="Buy (LR)")
plt.scatter(lr_sell_dates, lr_sell_prices, marker='v', color='magenta', s=70, label="Sell (LR)")
plt.xlabel("Date")
plt.ylabel("S&P500 Price")
plt.title("Trade Signals on S&P500")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(sim_df["Date"], sim_df["S&P500"], label='S&P500', color='blue')
plt.xlabel("Date")
plt.ylabel("S&P500 Price")
plt.title("S&P500 Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
