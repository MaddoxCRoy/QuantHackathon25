import pandas as pd

# Load CSV file with correct handling of headers
file_path = "Data/Quantathon_Data_2025csv.csv"
df = pd.read_csv(file_path, skiprows=1)  # Skip the first row if needed

# Rename columns for clarity
df.columns = ["Date_S&P500", "S&P500", "Bond_Rate", "Empty_Col", "Date_Prob", "PrDec", "PrInc"]
df.drop(columns=["Empty_Col"], inplace=True)  # Drop empty column

# Remove any non-numeric values from S&P 500 column (in case of extra text)
df = df[df["S&P500"].str.contains("[0-9]", na=False, regex=True)]

# Convert S&P500 and Bond Rate to numeric (remove commas first)
df["S&P500"] = df["S&P500"].str.replace(",", "").astype(float)
df["Bond_Rate"] = df["Bond_Rate"].astype(float)

# Convert dates explicitly
df["Date_S&P500"] = pd.to_datetime(df["Date_S&P500"], format="%d-%b-%y", errors="coerce")
df["Date_Prob"] = pd.to_datetime(df["Date_Prob"], format="%m/%d/%Y", errors="coerce")

# Separate into two DataFrames
sp500_data = df[["Date_S&P500", "S&P500", "Bond_Rate"]].dropna()
prob_data = df[["Date_Prob", "PrDec", "PrInc"]].dropna()

sp500_data.columns = ["Date", "S&P500", "Bond_Rate"]
prob_data.columns = ["Date", "PrDec", "PrInc"]


sp500_data.to_csv("SP500.csv", index=False)
prob_data.to_csv("Prob_Data.csv", index=False)

