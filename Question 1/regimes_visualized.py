import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Step 1: Load the Data
df = pd.read_csv('Data/market_predictions.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')  # Ensure data is sorted by date

# Step 2: Calculate Drawdown
df['rolling_max'] = df['S&P500'].cummax()  # Rolling maximum of S&P 500
df['drawdown'] = (df['S&P500'] - df['rolling_max']) / df['rolling_max']  # Drawdown calculation

# Step 3: Classify Market States
def classify_market(drawdown):
    if drawdown <= -0.20:  # Bear market: 20% or more decline from peak
        return 'Bear'
    elif drawdown > -0.05:  # Bull market: Less than 5% decline from peak
        return 'Bull'
    else:  # Static market: Between 5% and 20% decline
        return 'Static'

df['market_state'] = df['drawdown'].apply(classify_market)

# Step 4: Visualize the Results
# Set plot style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Define color mapping for market states
color_map = {'Bear': '#FF6B6B', 'Static': '#F7F7F7', 'Bull': '#4ECDC4'}  # Modern colors

# Plot the S&P 500 price
ax.plot(df['Date'], df['S&P500'], label='S&P 500', color='#2C3E50', linewidth=2.5)

# Color the background based on market states
for i in range(len(df) - 1):
    ax.axvspan(df['Date'].iloc[i], df['Date'].iloc[i+1], 
               facecolor=color_map[df['market_state'].iloc[i]], alpha=0.3)

# Create custom legend handles
legend_handles = [
    mpatches.Patch(color='#4ECDC4', alpha=0.3, label='Bull Market'),
    mpatches.Patch(color='#FF6B6B', alpha=0.3, label='Bear Market'),
    mpatches.Patch(color='#F7F7F7', alpha=0.3, label='Static Market')
]

# Add the custom legend
ax.legend(handles=legend_handles, loc='upper left', fontsize=12)

# Add a frame around the plot
for spine in ax.spines.values():
    spine.set_edgecolor('#2C3E50')  # Dark gray frame
    spine.set_linewidth(2)  # Thicker frame

# Labels and title
ax.set_title("S&P 500 Market States Over Time", fontsize=18, pad=20, color='#2C3E50')
ax.set_xlabel("Date", fontsize=14, color='#2C3E50')
ax.set_ylabel("S&P 500 Price", fontsize=14, color='#2C3E50')

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=12, colors='#2C3E50')

# Remove top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.tight_layout()
plt.show()
