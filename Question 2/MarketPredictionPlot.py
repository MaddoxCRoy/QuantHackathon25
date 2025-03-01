import pandas as pd
import matplotlib.pyplot as plt

# Load the market predictions CSV
df = pd.read_csv("Data/market_predictions.csv")

# Convert Date to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Define colors for correct market classifications
market_colors = {0: "red", 1: "green", 2: "blue"}
market_labels = {0: "Predicted Bear Market", 1: "Predicted Bull Market", 2: "Predicted Static Market"}

# Identify incorrect predictions
df["Incorrect_Prediction"] = df["Market_Type_Encoded"] != df["Predicted_Market_Encoded"]

# Create plot
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["S&P500"], label="S&P 500", color="black", linewidth=1)

# Scatter plot for correctly predicted market types
for market_type, color in market_colors.items():
    market_data = df[(df["Predicted_Market_Encoded"] == market_type) & (~df["Incorrect_Prediction"])]
    plt.scatter(market_data["Date"], market_data["S&P500"], color=color, label=market_labels[market_type], s=10)

# Scatter plot for incorrect predictions (highlighted in purple)
incorrect_data = df[df["Incorrect_Prediction"]]
plt.scatter(incorrect_data["Date"], incorrect_data["S&P500"], color="purple", label="Incorrect Prediction", s=20, marker="x")

# Formatting the Plot
plt.xlabel("Date")
plt.ylabel("S&P 500 Price")
plt.title("S&P 500 with Predicted Market Types & Incorrect Predictions Highlighted")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.show()
