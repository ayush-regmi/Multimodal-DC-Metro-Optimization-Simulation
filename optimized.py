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

# 6. MINIMUM OPTIMAL CONFIGURATION (Finding the knee point / diminishing returns)
print("\n" + "=" * 80)
print("6. MINIMUM OPTIMAL CONFIGURATION (Point of Diminishing Returns)")
print("=" * 80)

# Find best performance values
best_service_time = df['AvgServiceTime'].min()
best_completed_jobs = df['CompletedJobs'].max()
best_service_time_config = df.loc[df['AvgServiceTime'].idxmin()]
best_jobs_config = df.loc[df['CompletedJobs'].idxmax()]

print(f"\nBest possible performance:")
print(f"  Minimum Service Time: {best_service_time:.2f} min (Trains: {int(best_service_time_config['Trains'])}, Buses: {int(best_service_time_config['Buses'])})")
print(f"  Maximum Completed Jobs: {int(best_completed_jobs)} (Trains: {int(best_jobs_config['Trains'])}, Buses: {int(best_jobs_config['Buses'])})")

# Define acceptable thresholds (within 5% and 10% of best)
thresholds = [0.05, 0.10, 0.15]  # 5%, 10%, 15% thresholds

for threshold in thresholds:
    print(f"\n--- Finding minimum config within {threshold*100:.0f}% of best performance ---")
    
    # For service time: find minimum vehicles where service time is within threshold
    acceptable_service_time = best_service_time * (1 + threshold)
    service_time_candidates = df[df['AvgServiceTime'] <= acceptable_service_time].copy()
    
    if len(service_time_candidates) > 0:
        # Sort by total vehicles and find minimum
        service_time_candidates = service_time_candidates.sort_values('TotalVehicles')
        min_service_config = service_time_candidates.iloc[0]
        
        print(f"\nMinimum config for service time ≤ {acceptable_service_time:.2f} min ({threshold*100:.0f}% threshold):")
        print(f"  Trains: {int(min_service_config['Trains'])}, Buses: {int(min_service_config['Buses'])}")
        print(f"  Total Vehicles: {int(min_service_config['TotalVehicles'])}")
        print(f"  Service Time: {min_service_config['AvgServiceTime']:.2f} min")
        print(f"  Completed Jobs: {int(min_service_config['CompletedJobs'])}")
        print(f"  Savings vs best: {int(best_service_time_config['TotalVehicles'] - min_service_config['TotalVehicles'])} fewer vehicles")
    
    # For completed jobs: find minimum vehicles where jobs are within threshold
    acceptable_jobs = best_completed_jobs * (1 - threshold)
    jobs_candidates = df[df['CompletedJobs'] >= acceptable_jobs].copy()
    
    if len(jobs_candidates) > 0:
        # Sort by total vehicles and find minimum
        jobs_candidates = jobs_candidates.sort_values('TotalVehicles')
        min_jobs_config = jobs_candidates.iloc[0]
        
        print(f"\nMinimum config for completed jobs ≥ {int(acceptable_jobs)} ({threshold*100:.0f}% threshold):")
        print(f"  Trains: {int(min_jobs_config['Trains'])}, Buses: {int(min_jobs_config['Buses'])}")
        print(f"  Total Vehicles: {int(min_jobs_config['TotalVehicles'])}")
        print(f"  Service Time: {min_jobs_config['AvgServiceTime']:.2f} min")
        print(f"  Completed Jobs: {int(min_jobs_config['CompletedJobs'])}")
        print(f"  Savings vs best: {int(best_jobs_config['TotalVehicles'] - min_jobs_config['TotalVehicles'])} fewer vehicles")

# Marginal improvement analysis - find knee point
print("\n" + "=" * 80)
print("7. MARGINAL IMPROVEMENT ANALYSIS (Finding the Knee Point)")
print("=" * 80)

# Group by total vehicles and calculate average performance
vehicle_performance = df.groupby('TotalVehicles').agg({
    'AvgServiceTime': ['mean', 'min'],
    'CompletedJobs': ['mean', 'max'],
    'Trains': 'first',
    'Buses': 'first'
}).reset_index()
vehicle_performance.columns = ['TotalVehicles', 'AvgServiceTime_Mean', 'AvgServiceTime_Min', 
                                'CompletedJobs_Mean', 'CompletedJobs_Max', 'Trains', 'Buses']

# Calculate marginal improvements
vehicle_performance = vehicle_performance.sort_values('TotalVehicles')
vehicle_performance['ServiceTime_Improvement'] = vehicle_performance['AvgServiceTime_Min'].diff().abs() * -1  # Negative diff (lower is better)
vehicle_performance['Jobs_Improvement'] = vehicle_performance['CompletedJobs_Max'].diff()
vehicle_performance['MarginalServiceTimeGain'] = vehicle_performance['ServiceTime_Improvement'] / vehicle_performance['TotalVehicles'].diff()
vehicle_performance['MarginalJobsGain'] = vehicle_performance['Jobs_Improvement'] / vehicle_performance['TotalVehicles'].diff()

# Find knee point: where marginal improvement drops significantly
# Knee point is where adding one more vehicle improves service time by less than X minutes
# or improves jobs by less than Y jobs
knee_threshold_service = vehicle_performance['ServiceTime_Improvement'].quantile(0.25)  # Bottom 25% of improvements
knee_threshold_jobs = vehicle_performance['Jobs_Improvement'].quantile(0.25)  # Bottom 25% of improvements

knee_point_service = vehicle_performance[
    (vehicle_performance['ServiceTime_Improvement'] < knee_threshold_service) &
    (vehicle_performance['TotalVehicles'] > vehicle_performance['TotalVehicles'].min())
].iloc[0] if len(vehicle_performance[
    (vehicle_performance['ServiceTime_Improvement'] < knee_threshold_service) &
    (vehicle_performance['TotalVehicles'] > vehicle_performance['TotalVehicles'].min())
]) > 0 else None

knee_point_jobs = vehicle_performance[
    (vehicle_performance['Jobs_Improvement'] < knee_threshold_jobs) &
    (vehicle_performance['TotalVehicles'] > vehicle_performance['TotalVehicles'].min())
].iloc[0] if len(vehicle_performance[
    (vehicle_performance['Jobs_Improvement'] < knee_threshold_jobs) &
    (vehicle_performance['TotalVehicles'] > vehicle_performance['TotalVehicles'].min())
]) > 0 else None

print("\nKnee Point Analysis (where marginal improvements become small):")
if knee_point_service is not None:
    print(f"\nService Time Knee Point:")
    print(f"  Total Vehicles: {int(knee_point_service['TotalVehicles'])}")
    print(f"  Avg Service Time: {knee_point_service['AvgServiceTime_Min']:.2f} min")
    print(f"  Marginal improvement: {knee_point_service['ServiceTime_Improvement']:.4f} min per vehicle")
    
    # Find actual config closest to this
    knee_service_config = df[df['TotalVehicles'] == int(knee_point_service['TotalVehicles'])].sort_values('AvgServiceTime').iloc[0]
    print(f"  Recommended: {int(knee_service_config['Trains'])} trains, {int(knee_service_config['Buses'])} buses")

if knee_point_jobs is not None:
    print(f"\nCompleted Jobs Knee Point:")
    print(f"  Total Vehicles: {int(knee_point_jobs['TotalVehicles'])}")
    print(f"  Max Completed Jobs: {int(knee_point_jobs['CompletedJobs_Max'])}")
    print(f"  Marginal improvement: {knee_point_jobs['Jobs_Improvement']:.2f} jobs per vehicle")
    
    # Find actual config closest to this
    knee_jobs_config = df[df['TotalVehicles'] == int(knee_point_jobs['TotalVehicles'])].sort_values('CompletedJobs', ascending=False).iloc[0]
    print(f"  Recommended: {int(knee_jobs_config['Trains'])} trains, {int(knee_jobs_config['Buses'])} buses")

# Find balanced knee point (considering both metrics)
print("\n" + "=" * 80)
print("8. BALANCED MINIMUM OPTIMAL (Recommended)")
print("=" * 80)

# Initialize variables for visualization
min_balanced = None
balanced_candidates = pd.DataFrame()

# Find configurations that are within 10% of best on BOTH metrics
balanced_threshold = 0.10
acceptable_service = best_service_time * (1 + balanced_threshold)
acceptable_jobs = best_completed_jobs * (1 - balanced_threshold)

balanced_candidates = df[
    (df['AvgServiceTime'] <= acceptable_service) &
    (df['CompletedJobs'] >= acceptable_jobs)
].copy()

if len(balanced_candidates) > 0:
    # Sort by total vehicles to find minimum
    balanced_candidates = balanced_candidates.sort_values('TotalVehicles')
    min_balanced = balanced_candidates.iloc[0]
    
    print(f"\nMinimum balanced configuration (within 10% of best on both metrics):")
    print(f"  Trains: {int(min_balanced['Trains'])}, Buses: {int(min_balanced['Buses'])}")
    print(f"  Total Vehicles: {int(min_balanced['TotalVehicles'])}")
    print(f"  Service Time: {min_balanced['AvgServiceTime']:.2f} min (best: {best_service_time:.2f}, {((min_balanced['AvgServiceTime']/best_service_time - 1)*100):.1f}% worse)")
    print(f"  Completed Jobs: {int(min_balanced['CompletedJobs'])} (best: {int(best_completed_jobs)}, {((1 - min_balanced['CompletedJobs']/best_completed_jobs)*100):.1f}% less)")
    print(f"  Efficiency: {min_balanced['JobsPerVehicle']:.2f} jobs/vehicle")
    
    # Compare to maximum configs
    print(f"\n  Savings vs maximum service time config:")
    print(f"    Vehicles: {int(best_service_time_config['TotalVehicles'] - min_balanced['TotalVehicles'])} fewer")
    print(f"    Service time difference: {min_balanced['AvgServiceTime'] - best_service_time:.2f} min")
    
    print(f"\n  Savings vs maximum jobs config:")
    print(f"    Vehicles: {int(best_jobs_config['TotalVehicles'] - min_balanced['TotalVehicles'])} fewer")
    print(f"    Jobs difference: {int(best_completed_jobs - min_balanced['CompletedJobs'])} fewer")
else:
    print("\nNo configuration found within 10% threshold for both metrics.")
    print("Trying with 15% threshold...")
    balanced_threshold = 0.15
    acceptable_service = best_service_time * (1 + balanced_threshold)
    acceptable_jobs = best_completed_jobs * (1 - balanced_threshold)
    
    balanced_candidates = df[
        (df['AvgServiceTime'] <= acceptable_service) &
        (df['CompletedJobs'] >= acceptable_jobs)
    ].copy()
    
    if len(balanced_candidates) > 0:
        balanced_candidates = balanced_candidates.sort_values('TotalVehicles')
        min_balanced = balanced_candidates.iloc[0]
        
        print(f"\nMinimum balanced configuration (within 15% of best on both metrics):")
        print(f"  Trains: {int(min_balanced['Trains'])}, Buses: {int(min_balanced['Buses'])}")
        print(f"  Total Vehicles: {int(min_balanced['TotalVehicles'])}")
        print(f"  Service Time: {min_balanced['AvgServiceTime']:.2f} min")
        print(f"  Completed Jobs: {int(min_balanced['CompletedJobs'])}")
    else:
        # Fallback: use the best balanced config we can find
        min_balanced = None
        balanced_candidates = pd.DataFrame()

# 9. RECOMMENDED CONFIGURATION (Balanced approach)
print("\n" + "=" * 80)
print("9. RECOMMENDED CONFIGURATION (Balanced Approach)")
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

# 10. SENSITIVITY ANALYSIS
print("\n" + "=" * 80)
print("10. SENSITIVITY ANALYSIS")
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

# 11. VISUALIZATION
print("\n" + "=" * 80)
print("11. GENERATING OPTIMIZATION VISUALIZATIONS")
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

# Add visualization for marginal improvements
# Create additional figure for knee point analysis
fig2 = plt.figure(figsize=(16, 10))

# 1. Service Time vs Total Vehicles with knee point
ax1 = plt.subplot(2, 3, 1)
vehicle_perf_sorted = vehicle_performance.sort_values('TotalVehicles')
plt.plot(vehicle_perf_sorted['TotalVehicles'], vehicle_perf_sorted['AvgServiceTime_Min'], 
         'o-', linewidth=2, markersize=6, label='Minimum Service Time')
if knee_point_service is not None:
    plt.axvline(knee_point_service['TotalVehicles'], color='r', linestyle='--', 
                linewidth=2, label=f"Knee Point: {int(knee_point_service['TotalVehicles'])} vehicles")
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Minimum Service Time (minutes)', fontweight='bold')
plt.title('Service Time vs Fleet Size (Knee Point Analysis)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Completed Jobs vs Total Vehicles with knee point
ax2 = plt.subplot(2, 3, 2)
plt.plot(vehicle_perf_sorted['TotalVehicles'], vehicle_perf_sorted['CompletedJobs_Max'], 
         's-', linewidth=2, markersize=6, label='Maximum Completed Jobs', color='green')
if knee_point_jobs is not None:
    plt.axvline(knee_point_jobs['TotalVehicles'], color='r', linestyle='--', 
                linewidth=2, label=f"Knee Point: {int(knee_point_jobs['TotalVehicles'])} vehicles")
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Maximum Completed Jobs', fontweight='bold')
plt.title('Throughput vs Fleet Size (Knee Point Analysis)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Marginal Improvement in Service Time
ax3 = plt.subplot(2, 3, 3)
plt.plot(vehicle_perf_sorted['TotalVehicles'][1:], vehicle_perf_sorted['ServiceTime_Improvement'][1:], 
         'o-', linewidth=2, markersize=6, color='orange')
if knee_point_service is not None:
    plt.axvline(knee_point_service['TotalVehicles'], color='r', linestyle='--', linewidth=2)
plt.axhline(knee_threshold_service, color='gray', linestyle=':', linewidth=1, label='Threshold')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Service Time Improvement (min)', fontweight='bold')
plt.title('Marginal Service Time Improvement', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 4. Marginal Improvement in Jobs
ax4 = plt.subplot(2, 3, 4)
plt.plot(vehicle_perf_sorted['TotalVehicles'][1:], vehicle_perf_sorted['Jobs_Improvement'][1:], 
         's-', linewidth=2, markersize=6, color='purple')
if knee_point_jobs is not None:
    plt.axvline(knee_point_jobs['TotalVehicles'], color='r', linestyle='--', linewidth=2)
plt.axhline(knee_threshold_jobs, color='gray', linestyle=':', linewidth=1, label='Threshold')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Jobs Improvement', fontweight='bold')
plt.title('Marginal Jobs Improvement', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 5. Acceptable performance regions
ax5 = plt.subplot(2, 3, 5)
plt.scatter(df['TotalVehicles'], df['AvgServiceTime'], 
            c=df['CompletedJobs'], cmap='viridis', alpha=0.4, s=30, label='All Configurations')
# Highlight acceptable region (within 10% of best)
acceptable_service_10 = best_service_time * 1.10
acceptable_jobs_10 = best_completed_jobs * 0.90
acceptable_region = df[
    (df['AvgServiceTime'] <= acceptable_service_10) &
    (df['CompletedJobs'] >= acceptable_jobs_10)
]
if len(acceptable_region) > 0:
    plt.scatter(acceptable_region['TotalVehicles'], acceptable_region['AvgServiceTime'],
                c='red', s=100, marker='*', edgecolors='black', linewidths=1,
                label='Within 10% of Best', zorder=5)
    if min_balanced is not None and len(balanced_candidates) > 0:
        plt.scatter(min_balanced['TotalVehicles'], min_balanced['AvgServiceTime'],
                    c='yellow', s=300, marker='D', edgecolors='black', linewidths=2,
                    label='Minimum Optimal', zorder=6)
plt.axhline(acceptable_service_10, color='r', linestyle='--', alpha=0.5, label='10% Threshold')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Average Service Time (minutes)', fontweight='bold')
plt.title('Acceptable Performance Region', fontsize=12, fontweight='bold')
plt.colorbar(label='Completed Jobs')
plt.grid(True, alpha=0.3)
plt.legend()

# 6. Efficiency vs Total Vehicles with knee point
ax6 = plt.subplot(2, 3, 6)
vehicle_perf_sorted['Efficiency'] = vehicle_perf_sorted['CompletedJobs_Max'] / vehicle_perf_sorted['TotalVehicles']
plt.plot(vehicle_perf_sorted['TotalVehicles'], vehicle_perf_sorted['Efficiency'], 
         '^-', linewidth=2, markersize=6, color='teal', label='Efficiency')
if knee_point_jobs is not None:
    plt.axvline(knee_point_jobs['TotalVehicles'], color='r', linestyle='--', 
                linewidth=2, label=f"Knee Point")
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Efficiency (Jobs per Vehicle)', fontweight='bold')
plt.title('Efficiency vs Fleet Size', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('knee_point_analysis.png', dpi=300, bbox_inches='tight')
print("\nKnee point analysis visualization saved as 'knee_point_analysis.png'")

# 12. SUMMARY TABLE
print("\n" + "=" * 80)
print("12. SUMMARY: ALL OPTIMAL CONFIGURATIONS")
print("=" * 80)

# Build summary data
summary_criteria = ['Minimize Service Time', 'Maximize Throughput', 'Maximize Efficiency', 
                    'Minimize Service Time/Vehicle', 'Balanced (Recommended)']
summary_trains = [int(optimal_min_time['Trains']), int(optimal_max_jobs['Trains']),
                  int(optimal_efficiency['Trains']), int(optimal_st_per_vehicle['Trains']),
                  int(optimal_composite['Trains'])]
summary_buses = [int(optimal_min_time['Buses']), int(optimal_max_jobs['Buses']),
                 int(optimal_efficiency['Buses']), int(optimal_st_per_vehicle['Buses']),
                 int(optimal_composite['Buses'])]
summary_service = [optimal_min_time['AvgServiceTime'], optimal_max_jobs['AvgServiceTime'],
                   optimal_efficiency['AvgServiceTime'], optimal_st_per_vehicle['AvgServiceTime'],
                   optimal_composite['AvgServiceTime']]
summary_jobs = [int(optimal_min_time['CompletedJobs']), int(optimal_max_jobs['CompletedJobs']),
                int(optimal_efficiency['CompletedJobs']), int(optimal_st_per_vehicle['CompletedJobs']),
                int(optimal_composite['CompletedJobs'])]
summary_eff = [optimal_min_time['JobsPerVehicle'], optimal_max_jobs['JobsPerVehicle'],
               optimal_efficiency['JobsPerVehicle'], optimal_st_per_vehicle['JobsPerVehicle'],
               optimal_composite['JobsPerVehicle']]

# Add minimum optimal if found
if min_balanced is not None:
    summary_criteria.append('Minimum Optimal (10% threshold)')
    summary_trains.append(int(min_balanced['Trains']))
    summary_buses.append(int(min_balanced['Buses']))
    summary_service.append(min_balanced['AvgServiceTime'])
    summary_jobs.append(int(min_balanced['CompletedJobs']))
    summary_eff.append(min_balanced['JobsPerVehicle'])

summary_data = {
    'Criterion': summary_criteria,
    'Trains': summary_trains,
    'Buses': summary_buses,
    'Service Time (min)': summary_service,
    'Completed Jobs': summary_jobs,
    'Efficiency': summary_eff
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
if min_balanced is not None:
    print(f"\n5. ⭐ MINIMUM OPTIMAL CONFIGURATION: {int(min_balanced['Trains'])} trains, {int(min_balanced['Buses'])} buses")
    print(f"   - Service Time: {min_balanced['AvgServiceTime']:.2f} min (within 10% of best)")
    print(f"   - Completed Jobs: {int(min_balanced['CompletedJobs'])} (within 10% of best)")
    print(f"   - Total Vehicles: {int(min_balanced['TotalVehicles'])} (saves {int(best_service_time_config['TotalVehicles'] + best_jobs_config['TotalVehicles'] - 2*min_balanced['TotalVehicles'])} vehicles vs max configs)")
    print(f"   - This is the MINIMUM number of vehicles needed for near-optimal performance!")
print("\n" + "=" * 80)

