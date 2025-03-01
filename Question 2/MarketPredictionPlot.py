import pandas as pd
import matplotlib.pyplot as plt

# Load the market predictions CSV
df = pd.read_csv("Data/market_predictions.csv")

# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Plot S&P 500 Prices with Predicted Bull/Bear Market Coloring
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["S&P500"], label="S&P 500", color="black")

# Highlight Predicted Bear Markets
bear_dates = df[df["Predicted_Market"] == 0]["Date"]
bear_prices = df[df["Predicted_Market"] == 0]["S&P500"]
plt.scatter(bear_dates, bear_prices, color="red", label="Predicted Bear Market", s=10)

# Highlight Predicted Bull Markets
bull_dates = df[df["Predicted_Market"] == 1]["Date"]
bull_prices = df[df["Predicted_Market"] == 1]["S&P500"]
plt.scatter(bull_dates, bull_prices, color="green", label="Predicted Bull Market", s=10)

# Formatting the Plot
plt.xlabel("Date")
plt.ylabel("S&P 500 Price")
plt.title("S&P 500 with Predicted Bull/Bear Market Indicators")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.show()
