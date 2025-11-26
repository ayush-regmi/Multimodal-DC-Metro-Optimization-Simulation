import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the results
csv_path = 'csv/results.csv' if os.path.exists('csv/results.csv') else 'results.csv'
df = pd.read_csv(csv_path)

# Calculate total workers from stations.csv for minimum jobs constraint
stations_path = 'csv/stations.csv' if os.path.exists('csv/stations.csv') else 'stations.csv'
total_workers = 0
if os.path.exists(stations_path):
    try:
        stations_df = pd.read_csv(stations_path)
        total_workers = stations_df['numWorkers'].sum()
    except Exception as e:
        print(f"Warning: Could not read stations.csv: {e}")
        # Fallback: estimate from data if stations.csv not available
        # Use a reasonable estimate based on typical DC metro area
        total_workers = 900000  # Approximate from your stations.csv
else:
    # Fallback estimate
    total_workers = 900000

# Define minimum jobs requirement
# Realistic transit systems serve 70-80% of workers daily
# Minimum acceptable: 60% (system is failing if below this)
# Good performance: 75% (typical transit system)
MIN_JOBS_PERCENTAGE = 0.70  # 70% of workers must be served
min_jobs_required = int(total_workers * MIN_JOBS_PERCENTAGE)

print("=" * 80)
print("COMPREHENSIVE OPTIMAL TRAIN AND BUS CONFIGURATION ANALYSIS")
print("=" * 80)
print(f"\nTotal simulations analyzed: {len(df)}")
print(f"Train range: {df['Trains'].min()} - {df['Trains'].max()}")
print(f"Bus range: {df['Buses'].min()} - {df['Buses'].max()}")
print(f"\nMetrics available:")
print(f"  - Average Service Time: {df['AvgServiceTime'].min():.2f} - {df['AvgServiceTime'].max():.2f} minutes")
print(f"  - Completed Jobs: {df['CompletedJobs'].min()} - {df['CompletedJobs'].max()}")
print(f"  - Longest Service Time: {df['LongestServiceTime'].min():.2f} - {df['LongestServiceTime'].max():.2f} minutes")
print(f"\n📊 SERVICE LEVEL REQUIREMENT:")
print(f"  Total Workers in System: {total_workers:,}")
print(f"  Minimum Jobs Required: {min_jobs_required:,} ({MIN_JOBS_PERCENTAGE*100:.0f}% of workers)")
print(f"  Configurations meeting minimum: {(df['CompletedJobs'] >= min_jobs_required).sum()} out of {len(df)}")

# Calculate efficiency metrics
df['TotalVehicles'] = df['Trains'] + df['Buses']
df['JobsPerVehicle'] = df['CompletedJobs'] / df['TotalVehicles']
# Calculate buses per train (more meaningful for mass transit)
df['BusesPerTrain'] = df['Buses'] / (df['Trains'] + 1e-6)  # Avoid division by zero
df['TrainBusRatio'] = df['Trains'] / (df['Buses'] + 1)  # Keep for reference

# Normalize all metrics to 0-1 scale
def normalize_series(series, reverse=False):
    """Normalize a series to 0-1 scale. If reverse=True, higher values become lower (for minimization)."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    normalized = (series - min_val) / (max_val - min_val)
    if reverse:
        normalized = 1 - normalized
    return normalized

df['NormServiceTime'] = normalize_series(df['AvgServiceTime'], reverse=True)  # Lower is better
df['NormCompletedJobs'] = normalize_series(df['CompletedJobs'], reverse=False)  # Higher is better
df['NormEfficiency'] = normalize_series(df['JobsPerVehicle'], reverse=False)  # Higher is better
df['NormTotalVehicles'] = normalize_series(df['TotalVehicles'], reverse=True)  # Lower is better (cost)

# Calculate rate of change for service time and completed jobs
print("\n" + "=" * 80)
print("RATE OF CHANGE ANALYSIS (Finding Diminishing Returns)")
print("=" * 80)

# Sort by total vehicles for rate of change calculation
df_sorted = df.sort_values('TotalVehicles').reset_index(drop=True)

# Calculate rate of change (derivative approximation)
df_sorted['ServiceTime_RateOfChange'] = df_sorted['AvgServiceTime'].diff() * -1  # Negative because lower is better
df_sorted['Jobs_RateOfChange'] = df_sorted['CompletedJobs'].diff()
df_sorted['Vehicles_Increment'] = df_sorted['TotalVehicles'].diff()

# Normalize rate of change per vehicle added
df_sorted['ServiceTime_RatePerVehicle'] = df_sorted['ServiceTime_RateOfChange'] / df_sorted['Vehicles_Increment']
df_sorted['Jobs_RatePerVehicle'] = df_sorted['Jobs_RateOfChange'] / df_sorted['Vehicles_Increment']

# Replace NaN and inf values
df_sorted['ServiceTime_RatePerVehicle'] = df_sorted['ServiceTime_RatePerVehicle'].replace([np.inf, -np.inf], 0).fillna(0)
df_sorted['Jobs_RatePerVehicle'] = df_sorted['Jobs_RatePerVehicle'].replace([np.inf, -np.inf], 0).fillna(0)

# Find knee point: where rate of change drops significantly
# Only consider positive improvements (service time decreasing, jobs increasing)
df_sorted_improving = df_sorted[
    (df_sorted['ServiceTime_RatePerVehicle'] > 0) &  # Service time improving (decreasing)
    (df_sorted['Jobs_RatePerVehicle'] > 0)  # Jobs improving (increasing)
].copy()

if len(df_sorted_improving) > 0:
    # Use percentile-based threshold for improving configurations
    service_time_rate_threshold = df_sorted_improving['ServiceTime_RatePerVehicle'].quantile(0.25)  # Bottom 25%
    jobs_rate_threshold = df_sorted_improving['Jobs_RatePerVehicle'].quantile(0.25)  # Bottom 25%
    
    print(f"\nRate of Change Thresholds (for improving configurations):")
    print(f"  Service Time improvement threshold: {service_time_rate_threshold:.4f} min/vehicle")
    print(f"  Jobs improvement threshold: {jobs_rate_threshold:.2f} jobs/vehicle")
    
    # Find configurations where rate of change is below threshold (diminishing returns)
    knee_point_candidates = df_sorted_improving[
        (df_sorted_improving['ServiceTime_RatePerVehicle'] < service_time_rate_threshold) &
        (df_sorted_improving['Jobs_RatePerVehicle'] < jobs_rate_threshold) &
        (df_sorted_improving['TotalVehicles'] > df_sorted_improving['TotalVehicles'].min())
    ].copy()
else:
    # Fallback: use all data
    service_time_rate_threshold = df_sorted['ServiceTime_RatePerVehicle'].quantile(0.25)
    jobs_rate_threshold = df_sorted['Jobs_RatePerVehicle'].quantile(0.25)
    print(f"\nRate of Change Thresholds:")
    print(f"  Service Time improvement threshold: {service_time_rate_threshold:.4f} min/vehicle")
    print(f"  Jobs improvement threshold: {jobs_rate_threshold:.2f} jobs/vehicle")
    knee_point_candidates = df_sorted[
        (df_sorted['ServiceTime_RatePerVehicle'] > 0) &
        (df_sorted['Jobs_RatePerVehicle'] > 0) &
        (df_sorted['TotalVehicles'] > df_sorted['TotalVehicles'].min())
    ].copy()

if len(knee_point_candidates) > 0:
    # Take the first (minimum vehicles) knee point
    knee_point = knee_point_candidates.iloc[0]
    print(f"\n⭐ KNEE POINT FOUND (Point of Diminishing Returns):")
    print(f"  Total Vehicles: {int(knee_point['TotalVehicles'])}")
    print(f"  Trains: {int(knee_point['Trains'])}, Buses: {int(knee_point['Buses'])}")
    print(f"  Service Time: {knee_point['AvgServiceTime']:.2f} min")
    print(f"  Completed Jobs: {int(knee_point['CompletedJobs'])}")
    print(f"  Service Time Rate: {knee_point['ServiceTime_RatePerVehicle']:.4f} min/vehicle")
    print(f"  Jobs Rate: {knee_point['Jobs_RatePerVehicle']:.2f} jobs/vehicle")
else:
    print("\n⚠ No clear knee point found. Using alternative method...")
    # Alternative: find where improvement rate drops below median
    median_service_rate = df_sorted['ServiceTime_RatePerVehicle'].median()
    median_jobs_rate = df_sorted['Jobs_RatePerVehicle'].median()
    knee_point_candidates = df_sorted[
        (df_sorted['ServiceTime_RatePerVehicle'] < median_service_rate) &
        (df_sorted['Jobs_RatePerVehicle'] < median_jobs_rate) &
        (df_sorted['TotalVehicles'] > df_sorted['TotalVehicles'].min())
    ].copy()
    if len(knee_point_candidates) > 0:
        knee_point = knee_point_candidates.iloc[0]
    else:
        # Fallback: use median total vehicles
        knee_point = df_sorted.iloc[len(df_sorted) // 2]

# Train/Bus Ratio Analysis
print("\n" + "=" * 80)
print("TRAIN/BUS RATIO ANALYSIS")
print("=" * 80)

# Define reasonable ratio constraints
# Mass Transit Reality: In a hub-and-spoke system, you need MANY buses to feed trains
# Realistic: 5 to 50 buses per train (not trains per bus!)
# A single train carries ~1,000 people, a bus carries ~50 people
# You need 10-20 buses minimum to feed one train effectively
min_buses_per_train = 5   # Minimum 5 buses per train
max_buses_per_train = 50  # Maximum 50 buses per train (very high capacity)

df['ReasonableRatio'] = (df['BusesPerTrain'] >= min_buses_per_train) & (df['BusesPerTrain'] <= max_buses_per_train)

print(f"\nReasonable Buses per Train Range: {min_buses_per_train} - {max_buses_per_train}")
print(f"  (This means for every train, there should be {min_buses_per_train}-{max_buses_per_train} buses feeding it)")
print(f"Configurations with reasonable ratio: {df['ReasonableRatio'].sum()} out of {len(df)}")

# Filter out unreasonable ratios
df_reasonable = df[df['ReasonableRatio']].copy()

if len(df_reasonable) == 0:
    print("⚠ Warning: No configurations found with reasonable train/bus ratio. Using all configurations.")
    df_reasonable = df.copy()

# Apply minimum jobs constraint - filter out configurations that don't meet service level
print("\n" + "=" * 80)
print("MINIMUM JOBS CONSTRAINT (Service Level Validation)")
print("=" * 80)

df_meets_minimum = df_reasonable[df_reasonable['CompletedJobs'] >= min_jobs_required].copy()
configs_below_minimum = len(df_reasonable) - len(df_meets_minimum)

print(f"\nFiltering configurations that don't meet minimum service level:")
print(f"  Configurations with reasonable ratio: {len(df_reasonable)}")
print(f"  Configurations meeting minimum jobs ({min_jobs_required:,}): {len(df_meets_minimum)}")
print(f"  Configurations rejected (below minimum): {configs_below_minimum}")

if len(df_meets_minimum) == 0:
    print("\n⚠ WARNING: No configurations meet the minimum jobs requirement!")
    print(f"  This suggests the simulation may need:")
    print(f"    - More vehicles (trains/buses)")
    print(f"    - Longer simulation time")
    print(f"    - Different simulation parameters")
    print(f"  Proceeding with all configurations for analysis...")
    df_meets_minimum = df_reasonable.copy()
else:
    print(f"\n✓ {len(df_meets_minimum)} configurations meet the minimum service level requirement.")
    print(f"  Optimization will only consider these valid configurations.")

# Use configurations that meet minimum for optimization
df_reasonable = df_meets_minimum.copy()

# Composite Score Calculation
print("\n" + "=" * 80)
print("COMPOSITE SCORE CALCULATION")
print("=" * 80)

# Weight factors (sum should be 1.0)
# Mass Transit Priority: Throughput (Completed Jobs) is KING
# A system that only moves 5,000 people when demand is 50,000 is a failure
# regardless of how "efficient" those few vehicles are
WEIGHT_COMPLETED_JOBS = 0.60  # Throughput is the primary goal (60%)
WEIGHT_SERVICE_TIME = 0.20    # Service time matters but secondary (20%)
WEIGHT_EFFICIENCY = 0.10      # Efficiency per vehicle is nice but not critical (10%)
WEIGHT_COST = 0.10            # Cost (fewer vehicles) is a constraint, not the goal (10%)

# Calculate composite score (higher is better)
df_reasonable['CompositeScore'] = (
    WEIGHT_SERVICE_TIME * df_reasonable['NormServiceTime'] +
    WEIGHT_COMPLETED_JOBS * df_reasonable['NormCompletedJobs'] +
    WEIGHT_EFFICIENCY * df_reasonable['NormEfficiency'] +
    WEIGHT_COST * df_reasonable['NormTotalVehicles']
)

# Find optimal configuration with reasonable ratio
optimal_idx = df_reasonable['CompositeScore'].idxmax()
optimal_config = df_reasonable.loc[optimal_idx]

print(f"\nOptimal Configuration (with reasonable train/bus ratio):")
print(f"  Trains: {int(optimal_config['Trains'])}, Buses: {int(optimal_config['Buses'])}")
print(f"  Buses per Train: {optimal_config['BusesPerTrain']:.1f}")
print(f"  Service Time: {optimal_config['AvgServiceTime']:.2f} min")
print(f"  Completed Jobs: {int(optimal_config['CompletedJobs'])}")
print(f"  Total Vehicles: {int(optimal_config['TotalVehicles'])}")
print(f"  Efficiency: {optimal_config['JobsPerVehicle']:.2f} jobs/vehicle")
print(f"  Composite Score: {optimal_config['CompositeScore']:.4f}")

# Find optimal configuration considering rate of change (knee point)
print("\n" + "=" * 80)
print("OPTIMAL CONFIGURATION (Rate of Change Method)")
print("=" * 80)

# Improved knee point detection: look for negligible improvement (< 1% gain)
# Calculate percentage improvement per vehicle increment for reasonable configurations
df_reasonable_sorted = df_reasonable.sort_values('TotalVehicles').reset_index(drop=True)
df_reasonable_sorted['ServiceTime_PctImprovement'] = (
    df_reasonable_sorted['AvgServiceTime'].pct_change() * -1 * 100  # Negative pct change (lower is better)
)
df_reasonable_sorted['Jobs_PctImprovement'] = (
    df_reasonable_sorted['CompletedJobs'].pct_change() * 100
)
df_reasonable_sorted['Vehicles_Increment'] = df_reasonable_sorted['TotalVehicles'].diff()

# Find where improvement becomes negligible (< 1% per vehicle increment)
# Only consider configurations with at least moderate fleet size (avoid tiny fleets)
min_fleet_size = df_reasonable_sorted['TotalVehicles'].quantile(0.3)  # At least 30th percentile

negligible_improvement = df_reasonable_sorted[
    (df_reasonable_sorted['ServiceTime_PctImprovement'].fillna(100) < 1.0) &
    (df_reasonable_sorted['Jobs_PctImprovement'].fillna(100) < 1.0) &
    (df_reasonable_sorted['TotalVehicles'] >= min_fleet_size) &
    (df_reasonable_sorted['Vehicles_Increment'] > 0)
].copy()

if len(negligible_improvement) > 0:
    # Use the configuration with best composite score among negligible improvement points
    knee_optimal = negligible_improvement.loc[negligible_improvement['CompositeScore'].idxmax()]
    print(f"\n⭐ RECOMMENDED OPTIMAL CONFIGURATION (Negligible Improvement Point):")
    print(f"  Trains: {int(knee_optimal['Trains'])}, Buses: {int(knee_optimal['Buses'])}")
    print(f"  Buses per Train: {knee_optimal['BusesPerTrain']:.1f}")
    print(f"  Service Time: {knee_optimal['AvgServiceTime']:.2f} min")
    print(f"  Completed Jobs: {int(knee_optimal['CompletedJobs'])}")
    print(f"  Total Vehicles: {int(knee_optimal['TotalVehicles'])}")
    print(f"  Efficiency: {knee_optimal['JobsPerVehicle']:.2f} jobs/vehicle")
    print(f"  Composite Score: {knee_optimal['CompositeScore']:.4f}")
    
    # Compare knee point with composite score optimal
    # Use the one with better composite score (prioritize throughput)
    if knee_optimal['CompositeScore'] >= optimal_config['CompositeScore'] * 0.95:
        # Knee point is within 5% of optimal composite score - use it
        final_recommendation = knee_optimal
        print(f"\n  ✓ Knee point selected (composite score within 5% of optimal)")
    else:
        # Composite score method is significantly better - use it instead
        final_recommendation = optimal_config
        print(f"\n  ✓ Composite score method selected (significantly better than knee point)")
        print(f"    Knee point composite score: {knee_optimal['CompositeScore']:.4f}")
        print(f"    Optimal composite score: {optimal_config['CompositeScore']:.4f}")
else:
    print(f"\n⚠ No negligible improvement point found. Using optimal configuration from composite score method.")
    final_recommendation = optimal_config

# Comparison with best possible
print("\n" + "=" * 80)
print("COMPARISON WITH BEST POSSIBLE")
print("=" * 80)

best_service_time = df['AvgServiceTime'].min()
best_completed_jobs = df['CompletedJobs'].max()
best_service_config = df.loc[df['AvgServiceTime'].idxmin()]
best_jobs_config = df.loc[df['CompletedJobs'].idxmax()]

print(f"\nBest Possible Performance:")
print(f"  Minimum Service Time: {best_service_time:.2f} min")
print(f"    Config: {int(best_service_config['Trains'])} trains, {int(best_service_config['Buses'])} buses")
print(f"  Maximum Completed Jobs: {int(best_completed_jobs)}")
print(f"    Config: {int(best_jobs_config['Trains'])} trains, {int(best_jobs_config['Buses'])} buses")

print(f"\nRecommended Configuration Performance:")
print(f"  Service Time: {final_recommendation['AvgServiceTime']:.2f} min")
print(f"    Difference: {final_recommendation['AvgServiceTime'] - best_service_time:.2f} min ({((final_recommendation['AvgServiceTime']/best_service_time - 1)*100):.1f}% worse)")
print(f"  Completed Jobs: {int(final_recommendation['CompletedJobs']):,}")
print(f"    Difference: {int(final_recommendation['CompletedJobs'] - best_completed_jobs):,} jobs ({((1 - final_recommendation['CompletedJobs']/best_completed_jobs)*100):.1f}% less)")
print(f"  Service Level: {(final_recommendation['CompletedJobs']/total_workers*100):.1f}% of workers served")
print(f"    Meets minimum ({MIN_JOBS_PERCENTAGE*100:.0f}%): {'✓ YES' if final_recommendation['CompletedJobs'] >= min_jobs_required else '✗ NO'}")
print(f"  Total Vehicles: {int(final_recommendation['TotalVehicles'])}")
print(f"    Savings vs best service: {int(best_service_config['TotalVehicles'] - final_recommendation['TotalVehicles'])} vehicles")
print(f"    Savings vs best jobs: {int(best_jobs_config['TotalVehicles'] - final_recommendation['TotalVehicles'])} vehicles")

# Visualization
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))

# 1. Rate of Change Analysis
ax1 = plt.subplot(3, 3, 1)
df_sorted_plot = df_sorted[(df_sorted['Vehicles_Increment'] > 0) & (df_sorted['ServiceTime_RatePerVehicle'] > 0)]
if len(df_sorted_plot) > 0:
    plt.plot(df_sorted_plot['TotalVehicles'], df_sorted_plot['ServiceTime_RatePerVehicle'], 
             'o-', linewidth=2, markersize=4, label='Service Time Rate', color='blue', alpha=0.7)
    plt.axhline(service_time_rate_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
    if 'knee_point' in locals() and len(knee_point_candidates) > 0:
        plt.axvline(knee_point['TotalVehicles'], color='green', linestyle='--', linewidth=2, label='Knee Point')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Service Time Improvement Rate (min/vehicle)', fontweight='bold')
plt.title('Rate of Change: Service Time', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)

# 2. Jobs Rate of Change
ax2 = plt.subplot(3, 3, 2)
df_sorted_plot_jobs = df_sorted[(df_sorted['Vehicles_Increment'] > 0) & (df_sorted['Jobs_RatePerVehicle'] > 0)]
if len(df_sorted_plot_jobs) > 0:
    plt.plot(df_sorted_plot_jobs['TotalVehicles'], df_sorted_plot_jobs['Jobs_RatePerVehicle'], 
             's-', linewidth=2, markersize=4, label='Jobs Rate', color='green', alpha=0.7)
    plt.axhline(jobs_rate_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
    if 'knee_point' in locals() and len(knee_point_candidates) > 0:
        plt.axvline(knee_point['TotalVehicles'], color='green', linestyle='--', linewidth=2, label='Knee Point')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Jobs Improvement Rate (jobs/vehicle)', fontweight='bold')
plt.title('Rate of Change: Completed Jobs', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)

# 3. Service Time vs Total Vehicles
ax3 = plt.subplot(3, 3, 3)
plt.scatter(df['TotalVehicles'], df['AvgServiceTime'], 
            c=df['BusesPerTrain'], cmap='coolwarm', alpha=0.5, s=20, label='All Configs')
plt.scatter(df_reasonable['TotalVehicles'], df_reasonable['AvgServiceTime'],
            c='green', s=30, marker='*', edgecolors='black', linewidths=0.5,
            label='Reasonable Ratio', zorder=5)
plt.scatter(final_recommendation['TotalVehicles'], final_recommendation['AvgServiceTime'],
            c='yellow', s=400, marker='D', edgecolors='black', linewidths=2,
            label='Recommended', zorder=6)
plt.colorbar(label='Train/Bus Ratio')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Average Service Time (min)', fontweight='bold')
plt.title('Service Time vs Fleet Size', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)

# 4. Completed Jobs vs Total Vehicles
ax4 = plt.subplot(3, 3, 4)
plt.scatter(df['TotalVehicles'], df['CompletedJobs'], 
            c=df['BusesPerTrain'], cmap='coolwarm', alpha=0.5, s=20)
plt.scatter(df_reasonable['TotalVehicles'], df_reasonable['CompletedJobs'],
            c='green', s=30, marker='*', edgecolors='black', linewidths=0.5, zorder=5)
plt.scatter(final_recommendation['TotalVehicles'], final_recommendation['CompletedJobs'],
            c='yellow', s=400, marker='D', edgecolors='black', linewidths=2, zorder=6)
plt.colorbar(label='Train/Bus Ratio')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Completed Jobs', fontweight='bold')
plt.title('Throughput vs Fleet Size', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 5. Buses per Train Heatmap
ax5 = plt.subplot(3, 3, 5)
heatmap_ratio = df.pivot_table(index='Trains', columns='Buses', values='BusesPerTrain', aggfunc='mean')
sns.heatmap(heatmap_ratio, cmap='RdYlGn', annot=False, cbar_kws={'label': 'Buses per Train'}, 
            vmin=min_buses_per_train, vmax=max_buses_per_train)
plt.axhline(int(final_recommendation['Trains']) - 0.5, color='yellow', linewidth=3, label='Recommended')
plt.axvline(int(final_recommendation['Buses']) - 0.5, color='yellow', linewidth=3)
plt.title('Buses per Train Heatmap', fontsize=11, fontweight='bold')
plt.xlabel('Number of Buses', fontweight='bold')
plt.ylabel('Number of Trains', fontweight='bold')

# 6. Composite Score Heatmap
ax6 = plt.subplot(3, 3, 6)
heatmap_score = df_reasonable.pivot_table(index='Trains', columns='Buses', values='CompositeScore', aggfunc='mean')
sns.heatmap(heatmap_score, cmap='viridis', annot=False, cbar_kws={'label': 'Composite Score'})
plt.axhline(int(final_recommendation['Trains']) - 0.5, color='yellow', linewidth=3)
plt.axvline(int(final_recommendation['Buses']) - 0.5, color='yellow', linewidth=3)
plt.title('Composite Score Heatmap (Reasonable Ratios)', fontsize=11, fontweight='bold')
plt.xlabel('Number of Buses', fontweight='bold')
plt.ylabel('Number of Trains', fontweight='bold')

# 7. Efficiency vs Total Vehicles
ax7 = plt.subplot(3, 3, 7)
plt.scatter(df['TotalVehicles'], df['JobsPerVehicle'], 
            c=df['AvgServiceTime'], cmap='viridis_r', alpha=0.5, s=20)
plt.scatter(df_reasonable['TotalVehicles'], df_reasonable['JobsPerVehicle'],
            c='green', s=30, marker='*', edgecolors='black', linewidths=0.5, zorder=5)
plt.scatter(final_recommendation['TotalVehicles'], final_recommendation['JobsPerVehicle'],
            c='yellow', s=400, marker='D', edgecolors='black', linewidths=2, zorder=6)
plt.colorbar(label='Service Time (min)')
plt.xlabel('Total Vehicles', fontweight='bold')
plt.ylabel('Efficiency (Jobs per Vehicle)', fontweight='bold')
plt.title('Efficiency vs Fleet Size', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 8. Pareto Frontier
ax8 = plt.subplot(3, 3, 8)
plt.scatter(df['AvgServiceTime'], df['CompletedJobs'], 
            c=df['TotalVehicles'], cmap='coolwarm', alpha=0.4, s=20, label='All Configs')
plt.scatter(df_reasonable['AvgServiceTime'], df_reasonable['CompletedJobs'],
            c='green', s=40, marker='*', edgecolors='black', linewidths=0.5,
            label='Reasonable Ratio', zorder=5)
plt.scatter(final_recommendation['AvgServiceTime'], final_recommendation['CompletedJobs'],
            c='yellow', s=500, marker='D', edgecolors='black', linewidths=2,
            label='Recommended', zorder=6)
plt.colorbar(label='Total Vehicles')
plt.xlabel('Average Service Time (min)', fontweight='bold')
plt.ylabel('Completed Jobs', fontweight='bold')
plt.title('Pareto Frontier', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)

# 9. Summary Statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
service_level_pct = (final_recommendation['CompletedJobs']/total_workers*100) if total_workers > 0 else 0
meets_minimum = "✓ YES" if final_recommendation['CompletedJobs'] >= min_jobs_required else "✗ NO"

summary_text = f"""
⭐ RECOMMENDED CONFIGURATION

Trains: {int(final_recommendation['Trains'])}
Buses: {int(final_recommendation['Buses'])}
Buses/Train: {final_recommendation['BusesPerTrain']:.1f}

Service Time: {final_recommendation['AvgServiceTime']:.2f} min
Completed Jobs: {int(final_recommendation['CompletedJobs']):,}
Service Level: {service_level_pct:.1f}% of workers
Meets Minimum: {meets_minimum}
Total Vehicles: {int(final_recommendation['TotalVehicles'])}
Efficiency: {final_recommendation['JobsPerVehicle']:.2f} jobs/vehicle

Composite Score: {final_recommendation['CompositeScore']:.4f}

Performance vs Best:
• Service Time: {((final_recommendation['AvgServiceTime']/best_service_time - 1)*100):+.1f}%
• Completed Jobs: {((1 - final_recommendation['CompletedJobs']/best_completed_jobs)*100):+.1f}%
• Vehicle Savings: {int(best_service_config['TotalVehicles'] - final_recommendation['TotalVehicles'])} vs best service
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'optimization_analysis.png'")

# Save summary
summary_data = {
    'Metric': ['Recommended Trains', 'Recommended Buses', 'Buses per Train', 
               'Service Time (min)', 'Completed Jobs', 'Total Vehicles', 
               'Efficiency (jobs/vehicle)', 'Composite Score'],
    'Value': [int(final_recommendation['Trains']), int(final_recommendation['Buses']),
              f"{final_recommendation['BusesPerTrain']:.1f}",
              f"{final_recommendation['AvgServiceTime']:.2f}",
              int(final_recommendation['CompletedJobs']),
              int(final_recommendation['TotalVehicles']),
              f"{final_recommendation['JobsPerVehicle']:.2f}",
              f"{final_recommendation['CompositeScore']:.4f}"]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('optimal_configuration.csv', index=False)
print("Summary saved to 'optimal_configuration.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\n⭐ FINAL RECOMMENDATION:")
print(f"   {int(final_recommendation['Trains'])} Trains, {int(final_recommendation['Buses'])} Buses")
service_level = (final_recommendation['CompletedJobs']/total_workers*100) if total_workers > 0 else 0
print(f"   Service Level: {service_level:.1f}% of workers served ({int(final_recommendation['CompletedJobs']):,} jobs)")
if final_recommendation['CompletedJobs'] >= min_jobs_required:
    print(f"   ✓ Meets minimum service level requirement ({MIN_JOBS_PERCENTAGE*100:.0f}%)")
else:
    print(f"   ⚠ Does NOT meet minimum service level requirement ({MIN_JOBS_PERCENTAGE*100:.0f}%)")
print(f"   This configuration balances performance, efficiency, and cost while maintaining")
print(f"   a reasonable train/bus ratio and avoiding diminishing returns.")
print("=" * 80)
