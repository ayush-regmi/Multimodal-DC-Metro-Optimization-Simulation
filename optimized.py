import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv('results.csv')

print("=" * 80)
print("OPTIMAL TRAIN AND BUS CONFIGURATION ANALYSIS")
print("=" * 80)
print(f"\nTotal simulations analyzed: {len(df)}")
print(f"Train range: {df['Trains'].min()} - {df['Trains'].max()}")
print(f"Bus range: {df['Buses'].min()} - {df['Buses'].max()}")
print(f"\nMetrics available:")
print(f"  - Average Service Time: {df['AvgServiceTime'].min():.2f} - {df['AvgServiceTime'].max():.2f} minutes")
print(f"  - Completed Jobs: {df['CompletedJobs'].min()} - {df['CompletedJobs'].max()}")
print(f"  - Longest Service Time: {df['LongestServiceTime'].min():.2f} - {df['LongestServiceTime'].max():.2f} minutes")

# Calculate efficiency metrics early
df['TotalVehicles'] = df['Trains'] + df['Buses']
df['JobsPerVehicle'] = df['CompletedJobs'] / df['TotalVehicles']
df['ServiceTimePerVehicle'] = df['AvgServiceTime'] / df['TotalVehicles']

# 1. OPTIMIZATION BY MINIMIZING SERVICE TIME
print("\n" + "=" * 80)
print("1. OPTIMAL CONFIGURATION: MINIMIZE AVERAGE SERVICE TIME")
print("=" * 80)
min_service_time_idx = df['AvgServiceTime'].idxmin()
optimal_min_time = df.loc[min_service_time_idx]
print(f"\nBest Configuration:")
print(f"  Trains: {int(optimal_min_time['Trains'])}")
print(f"  Buses: {int(optimal_min_time['Buses'])}")
print(f"  Average Service Time: {optimal_min_time['AvgServiceTime']:.2f} minutes")
print(f"  Completed Jobs: {int(optimal_min_time['CompletedJobs'])}")
print(f"  Total Vehicles: {int(optimal_min_time['Trains'] + optimal_min_time['Buses'])}")

# 2. OPTIMIZATION BY MAXIMIZING THROUGHPUT
print("\n" + "=" * 80)
print("2. OPTIMAL CONFIGURATION: MAXIMIZE COMPLETED JOBS (THROUGHPUT)")
print("=" * 80)
max_jobs_idx = df['CompletedJobs'].idxmax()
optimal_max_jobs = df.loc[max_jobs_idx]
print(f"\nBest Configuration:")
print(f"  Trains: {int(optimal_max_jobs['Trains'])}")
print(f"  Buses: {int(optimal_max_jobs['Buses'])}")
print(f"  Average Service Time: {optimal_max_jobs['AvgServiceTime']:.2f} minutes")
print(f"  Completed Jobs: {int(optimal_max_jobs['CompletedJobs'])}")
print(f"  Total Vehicles: {int(optimal_max_jobs['Trains'] + optimal_max_jobs['Buses'])}")

# 3. EFFICIENCY METRIC: Jobs per Vehicle
print("\n" + "=" * 80)
print("3. OPTIMAL CONFIGURATION: MAXIMIZE EFFICIENCY (JOBS PER VEHICLE)")
print("=" * 80)
max_efficiency_idx = df['JobsPerVehicle'].idxmax()
optimal_efficiency = df.loc[max_efficiency_idx]
print(f"\nBest Configuration:")
print(f"  Trains: {int(optimal_efficiency['Trains'])}")
print(f"  Buses: {int(optimal_efficiency['Buses'])}")
print(f"  Average Service Time: {optimal_efficiency['AvgServiceTime']:.2f} minutes")
print(f"  Completed Jobs: {int(optimal_efficiency['CompletedJobs'])}")
print(f"  Total Vehicles: {int(optimal_efficiency['TotalVehicles'])}")
print(f"  Jobs per Vehicle: {optimal_efficiency['JobsPerVehicle']:.2f}")

# 4. COST-BENEFIT ANALYSIS: Service Time per Vehicle
print("\n" + "=" * 80)
print("4. OPTIMAL CONFIGURATION: MINIMIZE SERVICE TIME PER VEHICLE")
print("=" * 80)
min_st_per_vehicle_idx = df['ServiceTimePerVehicle'].idxmin()
optimal_st_per_vehicle = df.loc[min_st_per_vehicle_idx]
print(f"\nBest Configuration:")
print(f"  Trains: {int(optimal_st_per_vehicle['Trains'])}")
print(f"  Buses: {int(optimal_st_per_vehicle['Buses'])}")
print(f"  Average Service Time: {optimal_st_per_vehicle['AvgServiceTime']:.2f} minutes")
print(f"  Completed Jobs: {int(optimal_st_per_vehicle['CompletedJobs'])}")
print(f"  Service Time per Vehicle: {optimal_st_per_vehicle['ServiceTimePerVehicle']:.4f}")

# 5. PARETO OPTIMAL SOLUTIONS (Multi-objective optimization)
print("\n" + "=" * 80)
print("5. PARETO-OPTIMAL SOLUTIONS (Best Trade-offs)")
print("=" * 80)
print("Finding configurations that are not dominated by others...")

# Normalize objectives (lower is better for service time, higher is better for jobs)
# Manual normalization: (x - min) / (max - min)
service_min, service_max = df['AvgServiceTime'].min(), df['AvgServiceTime'].max()
jobs_min, jobs_max = df['CompletedJobs'].min(), df['CompletedJobs'].max()
df['NormalizedServiceTime'] = (df['AvgServiceTime'] - service_min) / (service_max - service_min)
df['NormalizedJobs'] = -(df['CompletedJobs'] - jobs_min) / (jobs_max - jobs_min)  # Negate because higher is better

# Find Pareto front
def is_pareto_optimal(idx, df):
    """Check if a point is Pareto optimal"""
    point_service = df.loc[idx, 'NormalizedServiceTime']
    point_jobs = df.loc[idx, 'NormalizedJobs']
    
    # Check if any other point dominates this one
    for other_idx in df.index:
        if other_idx == idx:
            continue
        other_service = df.loc[other_idx, 'NormalizedServiceTime']
        other_jobs = df.loc[other_idx, 'NormalizedJobs']
        
        # Dominated if other has lower service time AND higher jobs (lower normalized jobs)
        if (other_service <= point_service and other_jobs <= point_jobs and 
            (other_service < point_service or other_jobs < point_jobs)):
            return False
    return True

pareto_indices = [idx for idx in df.index if is_pareto_optimal(idx, df)]
pareto_df = df.loc[pareto_indices].sort_values('AvgServiceTime')

print(f"\nFound {len(pareto_indices)} Pareto-optimal configurations:")
print("\nTop 10 Pareto-optimal solutions (sorted by service time):")
for i, (idx, row) in enumerate(pareto_df.head(10).iterrows(), 1):
    print(f"\n{i}. Trains: {int(row['Trains'])}, Buses: {int(row['Buses'])}")
    print(f"   Service Time: {row['AvgServiceTime']:.2f} min | Jobs: {int(row['CompletedJobs'])} | "
          f"Efficiency: {row['JobsPerVehicle']:.2f} jobs/vehicle")

# 6. RECOMMENDED CONFIGURATION (Balanced approach)
print("\n" + "=" * 80)
print("6. RECOMMENDED CONFIGURATION (Balanced Approach)")
print("=" * 80)

# Create a composite score: normalize and combine service time and throughput
# Lower service time is better, higher jobs is better
df['NormalizedServiceTime'] = (df['AvgServiceTime'] - df['AvgServiceTime'].min()) / (df['AvgServiceTime'].max() - df['AvgServiceTime'].min())
df['NormalizedJobs'] = (df['CompletedJobs'] - df['CompletedJobs'].min()) / (df['CompletedJobs'].max() - df['CompletedJobs'].min())

# Composite score: weight service time more (0.6) and jobs less (0.4)
# Lower composite score is better
df['CompositeScore'] = 0.6 * df['NormalizedServiceTime'] - 0.4 * df['NormalizedJobs']
best_composite_idx = df['CompositeScore'].idxmin()
optimal_composite = df.loc[best_composite_idx]

print(f"\nRecommended Configuration (balanced service time and throughput):")
print(f"  Trains: {int(optimal_composite['Trains'])}")
print(f"  Buses: {int(optimal_composite['Buses'])}")
print(f"  Average Service Time: {optimal_composite['AvgServiceTime']:.2f} minutes")
print(f"  Completed Jobs: {int(optimal_composite['CompletedJobs'])}")
print(f"  Total Vehicles: {int(optimal_composite['TotalVehicles'])}")
print(f"  Efficiency: {optimal_composite['JobsPerVehicle']:.2f} jobs/vehicle")

# 7. SENSITIVITY ANALYSIS
print("\n" + "=" * 80)
print("7. SENSITIVITY ANALYSIS")
print("=" * 80)

# Analyze how performance changes with number of trains
train_analysis = df.groupby('Trains').agg({
    'AvgServiceTime': 'mean',
    'CompletedJobs': 'mean',
    'JobsPerVehicle': 'mean'
}).reset_index()

print("\nAverage performance by number of trains:")
print(f"{'Trains':<8} {'Avg Service Time':<18} {'Avg Completed Jobs':<20} {'Avg Efficiency':<15}")
print("-" * 70)
for _, row in train_analysis.iterrows():
    print(f"{int(row['Trains']):<8} {row['AvgServiceTime']:<18.2f} {row['CompletedJobs']:<20.0f} {row['JobsPerVehicle']:<15.2f}")

# Analyze how performance changes with number of buses (for optimal train count)
optimal_train_count = int(optimal_composite['Trains'])
bus_analysis = df[df['Trains'] == optimal_train_count].sort_values('Buses')
print(f"\nPerformance by number of buses (for {optimal_train_count} trains):")
print(f"{'Buses':<8} {'Service Time':<15} {'Completed Jobs':<18} {'Efficiency':<15}")
print("-" * 60)
for _, row in bus_analysis.head(10).iterrows():
    print(f"{int(row['Buses']):<8} {row['AvgServiceTime']:<15.2f} {int(row['CompletedJobs']):<18} {row['JobsPerVehicle']:<15.2f}")
if len(bus_analysis) > 10:
    print("...")
    for _, row in bus_analysis.tail(5).iterrows():
        print(f"{int(row['Buses']):<8} {row['AvgServiceTime']:<15.2f} {int(row['CompletedJobs']):<18} {row['JobsPerVehicle']:<15.2f}")

# 8. VISUALIZATION
print("\n" + "=" * 80)
print("8. GENERATING OPTIMIZATION VISUALIZATIONS")
print("=" * 80)

# Create a comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Heatmap of service time
ax1 = plt.subplot(2, 3, 1)
heatmap_data = df.pivot(index='Trains', columns='Buses', values='AvgServiceTime')
sns.heatmap(heatmap_data, cmap='viridis_r', annot=False, cbar_kws={'label': 'Service Time (min)'})
plt.title('Average Service Time Heatmap', fontsize=12, fontweight='bold')
plt.xlabel('Number of Buses')
plt.ylabel('Number of Trains')
# Mark optimal point
optimal_train = int(optimal_composite['Trains'])
optimal_bus = int(optimal_composite['Buses'])
plt.plot(optimal_bus - 0.5, optimal_train - 0.5, 'r*', markersize=20, label='Recommended')
plt.legend()

# 2. Heatmap of completed jobs
ax2 = plt.subplot(2, 3, 2)
heatmap_jobs = df.pivot(index='Trains', columns='Buses', values='CompletedJobs')
sns.heatmap(heatmap_jobs, cmap='viridis', annot=False, cbar_kws={'label': 'Completed Jobs'})
plt.title('Completed Jobs Heatmap', fontsize=12, fontweight='bold')
plt.xlabel('Number of Buses')
plt.ylabel('Number of Trains')
plt.plot(optimal_bus - 0.5, optimal_train - 0.5, 'r*', markersize=20, label='Recommended')
plt.legend()

# 3. Heatmap of efficiency
ax3 = plt.subplot(2, 3, 3)
heatmap_eff = df.pivot(index='Trains', columns='Buses', values='JobsPerVehicle')
sns.heatmap(heatmap_eff, cmap='plasma', annot=False, cbar_kws={'label': 'Jobs per Vehicle'})
plt.title('Efficiency Heatmap (Jobs per Vehicle)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Buses')
plt.ylabel('Number of Trains')
plt.plot(optimal_bus - 0.5, optimal_train - 0.5, 'r*', markersize=20, label='Recommended')
plt.legend()

# 4. Pareto frontier
ax4 = plt.subplot(2, 3, 4)
plt.scatter(df['AvgServiceTime'], df['CompletedJobs'], 
            c=df['TotalVehicles'], cmap='coolwarm', alpha=0.6, s=30, label='All Configurations')
plt.scatter(pareto_df['AvgServiceTime'], pareto_df['CompletedJobs'],
            c='red', s=100, marker='*', edgecolors='black', linewidths=1, 
            label='Pareto Optimal', zorder=5)
plt.scatter(optimal_composite['AvgServiceTime'], optimal_composite['CompletedJobs'],
            c='yellow', s=200, marker='D', edgecolors='black', linewidths=2,
            label='Recommended', zorder=6)
plt.colorbar(label='Total Vehicles')
plt.xlabel('Average Service Time (minutes)', fontweight='bold')
plt.ylabel('Completed Jobs', fontweight='bold')
plt.title('Pareto Frontier: Service Time vs Throughput', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 5. Service time vs number of trains
ax5 = plt.subplot(2, 3, 5)
train_means = df.groupby('Trains')['AvgServiceTime'].mean()
train_stds = df.groupby('Trains')['AvgServiceTime'].std()
plt.errorbar(train_means.index, train_means.values, yerr=train_stds.values, 
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
plt.axvline(optimal_train, color='r', linestyle='--', linewidth=2, label=f'Optimal: {optimal_train} trains')
plt.xlabel('Number of Trains', fontweight='bold')
plt.ylabel('Average Service Time (minutes)', fontweight='bold')
plt.title('Service Time vs Number of Trains', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 6. Efficiency vs total vehicles
ax6 = plt.subplot(2, 3, 6)
plt.scatter(df['TotalVehicles'], df['JobsPerVehicle'], 
            c=df['AvgServiceTime'], cmap='viridis_r', alpha=0.6, s=50)
plt.scatter(optimal_composite['TotalVehicles'], optimal_composite['JobsPerVehicle'],
            c='red', s=300, marker='*', edgecolors='black', linewidths=2,
            label='Recommended', zorder=5)
plt.colorbar(label='Service Time (min)')
plt.xlabel('Total Vehicles (Trains + Buses)', fontweight='bold')
plt.ylabel('Efficiency (Jobs per Vehicle)', fontweight='bold')
plt.title('Efficiency vs Fleet Size', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'optimization_analysis.png'")

# 9. SUMMARY TABLE
print("\n" + "=" * 80)
print("9. SUMMARY: ALL OPTIMAL CONFIGURATIONS")
print("=" * 80)

summary_data = {
    'Criterion': [
        'Minimize Service Time',
        'Maximize Throughput',
        'Maximize Efficiency',
        'Minimize Service Time/Vehicle',
        'Balanced (Recommended)'
    ],
    'Trains': [
        int(optimal_min_time['Trains']),
        int(optimal_max_jobs['Trains']),
        int(optimal_efficiency['Trains']),
        int(optimal_st_per_vehicle['Trains']),
        int(optimal_composite['Trains'])
    ],
    'Buses': [
        int(optimal_min_time['Buses']),
        int(optimal_max_jobs['Buses']),
        int(optimal_efficiency['Buses']),
        int(optimal_st_per_vehicle['Buses']),
        int(optimal_composite['Buses'])
    ],
    'Service Time (min)': [
        optimal_min_time['AvgServiceTime'],
        optimal_max_jobs['AvgServiceTime'],
        optimal_efficiency['AvgServiceTime'],
        optimal_st_per_vehicle['AvgServiceTime'],
        optimal_composite['AvgServiceTime']
    ],
    'Completed Jobs': [
        int(optimal_min_time['CompletedJobs']),
        int(optimal_max_jobs['CompletedJobs']),
        int(optimal_efficiency['CompletedJobs']),
        int(optimal_st_per_vehicle['CompletedJobs']),
        int(optimal_composite['CompletedJobs'])
    ],
    'Efficiency': [
        optimal_min_time['JobsPerVehicle'],
        optimal_max_jobs['JobsPerVehicle'],
        optimal_efficiency['JobsPerVehicle'],
        optimal_st_per_vehicle['JobsPerVehicle'],
        optimal_composite['JobsPerVehicle']
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('optimal_configurations.csv', index=False)
print("\n\nSummary saved to 'optimal_configurations.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nKey Insights:")
print(f"1. Minimum service time: {optimal_min_time['AvgServiceTime']:.2f} min with {int(optimal_min_time['Trains'])} trains, {int(optimal_min_time['Buses'])} buses")
print(f"2. Maximum throughput: {int(optimal_max_jobs['CompletedJobs'])} jobs with {int(optimal_max_jobs['Trains'])} trains, {int(optimal_max_jobs['Buses'])} buses")
print(f"3. Best efficiency: {optimal_efficiency['JobsPerVehicle']:.2f} jobs/vehicle with {int(optimal_efficiency['Trains'])} trains, {int(optimal_efficiency['Buses'])} buses")
print(f"4. Recommended configuration: {int(optimal_composite['Trains'])} trains, {int(optimal_composite['Buses'])} buses")
print(f"   (Balances service time and throughput)")
print("\n" + "=" * 80)

