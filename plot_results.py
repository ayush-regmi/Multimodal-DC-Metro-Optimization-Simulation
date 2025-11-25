import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CSV data
# Ensure 'results.csv' is in the same directory as this script
df = pd.read_csv('results.csv')

# 2. Pivot the data for the Heatmap
# X-axis: Buses, Y-axis: Trains, Value: AvgServiceTime
heatmap_data = df.pivot(index='Trains', columns='Buses', values='AvgServiceTime')

# 3. Create the Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis_r', annot=False) 
# cmap='viridis_r' reverses the color map so darker/cooler colors (better) represent lower times.

plt.title('Optimization Landscape: Average Service Time (Minutes)')
plt.xlabel('Number of Buses')
plt.ylabel('Number of Trains')

# 4. Save and Show
plt.savefig('optimization_heatmap.png')
plt.show()


# Scatter plot of all simulations
plt.scatter(df['AvgServiceTime'], df['CompletedJobs'], 
            c=df['Trains'], cmap='viridis', alpha=0.7, label='Simulations')

plt.colorbar(label='Number of Trains')
plt.title('Trade-off Analysis: Service Time vs. Throughput')
plt.xlabel('Average Service Time (Lower is Better)')
plt.ylabel('Total Completed Jobs (Higher is Better)')
plt.grid(True, linestyle='--', alpha=0.5)

# Highlight the "Sweet Spot" (Top Left Quadrant)
# Ideally, you want points in the top-left of this specific graph setup
plt.savefig('pareto_frontier.png')
plt.show()

