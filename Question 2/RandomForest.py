import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the preprocessed datasets
sp500_data = pd.read_csv("Data/SP500.csv")
prob_data = pd.read_csv("Data/Prob_Data.csv")

# Convert Date columns to datetime
sp500_data["Date"] = pd.to_datetime(sp500_data["Date"])
prob_data["Date"] = pd.to_datetime(prob_data["Date"])

# Merge data based on the closest previous probability date
df = pd.merge_asof(sp500_data.sort_values("Date"), prob_data.sort_values("Date"), on="Date", direction="backward")

# Sort by date after merging
df = df.sort_values("Date")

# Calculate daily returns
df["Returns"] = df["S&P500"].pct_change()

# Calculate volatility (rolling standard deviation)
df["Volatility"] = df["Returns"].rolling(30).std()

# Drop NaN values (caused by rolling calculations)
df.dropna(inplace=True)

# Identify peaks and drawdowns for market classification
df["Peak"] = df["S&P500"].cummax()
df["Trough"] = df["S&P500"].cummin()
df["Drawdown"] = (df["S&P500"] - df["Peak"]) / df["Peak"]
df["Growth"] = (df["S&P500"] - df["Trough"]) / df["Trough"]

# Label market types
conditions = [
    df["Drawdown"] <= -0.20,  # Bear Market: 20% decline
    df["Growth"] >= 0.20       # Bull Market: 20% increase
]
choices = ["Bear", "Bull"]
df["Market_Type"] = np.select(conditions, choices, default="Static")  # Static if neither

# Drop unnecessary columns
df.drop(columns=["Peak", "Trough", "Drawdown", "Growth"], inplace=True)

# Encode Market_Type as numerical values for the model
df["Market_Type_Encoded"] = df["Market_Type"].map({"Bear": 0, "Bull": 1, "Static": 2})

# Define features and target
X = df[["Returns", "Volatility", "PrDec", "PrInc", "Bond_Rate"]]
y = df["Market_Type_Encoded"]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Predict market type for all data
df["Predicted_Market_Encoded"] = rf_model.predict(X)
df["Predicted_Market"] = df["Predicted_Market_Encoded"].map({0: "Bear", 1: "Bull", 2: "Static"})

# Save predictions to CSV
df.to_csv("Data/market_predictions.csv", index=False)

print("Market predictions saved to 'market_predictions.csv'")