import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv', parse_dates=['Date Recorded'])

# 1. Number of transactions by town
top_20_towns = df['Town'].value_counts().index[:20]
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Town', order=top_20_towns)
plt.title('Total Real Estate Sales Transactions by Top 20 Towns (2001-2020)')
plt.xlabel('Number of Transactions')
plt.ylabel('Town Name')
plt.tight_layout()
plt.savefig('Total_Transactions_by_Town.jpg', bbox_inches='tight')
plt.show()

# 2. Average Real Estate Sales Price Trend (2001-2020) for selected towns
towns_to_plot = ['Bridgeport','Stamford', 'Waterbury', 'Norwalk', 'New Haven', 'Danbury', 'West Hartford', 'Hartford', 'Milford']
plt.figure(figsize=(12, 7))
for town in towns_to_plot:
    town_data = df[df['Town'] == town]
    yearly_avg_price = town_data.groupby(town_data['List Year'])['Sale Amount'].mean()
    plt.plot(yearly_avg_price.index, yearly_avg_price.values/1000000.0, marker='o', linestyle='-', label=town)

plt.title('Average Real Estate Sales Price Trend (2001-2020)')
plt.xlabel('Year')
plt.ylabel('Average Sale Amount (Million USD)')
plt.grid(True)
plt.xticks(range(2001, 2021))  # Use only existing years as ticks
plt.legend(loc='upper left')  # set the position of the legend
plt.tight_layout()
plt.savefig('Average_Sales_Price_Trend.jpg', bbox_inches='tight')
plt.show()

# 3. Stem Plot
fig, ax = plt.subplots(figsize=(12, 7))

# Create a list to hold all the labels for the x-axis
all_labels = []

# Define a set of marker shapes
marker_shapes = ['^', 's', 'p', 'd', 'o', '*', 'p', 'H', '8', 'X']

# For each town in the list, calculate the frequency for each Residential Type and plot a stem plot
for i, town in enumerate(towns_to_plot):
    town_data = df[df['Town'] == town]
    residential_type_counts = town_data['Residential Type'].value_counts(dropna=True)
    
    # Get the labels and frequencies for residential types
    labels = residential_type_counts.index
    freqs = residential_type_counts.values
    
    # For each residential type, plot a line (stem) using a unique marker shape
    for j, (label, freq) in enumerate(zip(labels, freqs)):
        marker_style = marker_shapes[j % len(marker_shapes)]
        markerline, stemlines, baseline = ax.stem([i + j*0.1], [freq], linefmt='-C{}'.format(j), markerfmt='C{}{}'.format(j, marker_style))
        all_labels.append(label)

# Remove duplicates from the labels and set them as x-tick labels
unique_labels = list(np.unique(all_labels))
ax.set_xticks(np.arange(len(towns_to_plot)) + (len(unique_labels) - 1)*0.1/2)  # Adjust x-tick positions to center them
ax.set_xticklabels(towns_to_plot)
ax.set_title("Distribution of Residential Types")
ax.set_xlabel("Town")
ax.set_ylabel("Frequency")
ax.legend(unique_labels)

plt.tight_layout()
plt.savefig('Residential_Types_Distribution.jpg', bbox_inches='tight')
plt.show()