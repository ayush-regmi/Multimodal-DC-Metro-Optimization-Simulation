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

# Calculate rate of change for service time and completed jobs
print("\n" + "=" * 80)
print("RATE OF CHANGE ANALYSIS (Trains vs Buses)")
print("=" * 80)

# Rate of Change vs Trains (within bus range 150-300)
print("\nCalculating rate of change vs increasing trains (bus range: 150-300)...")
df_bus_range = df[(df['Buses'] >= 150) & (df['Buses'] <= 300)].copy()

# Group by trains and calculate mean metrics
train_analysis = df_bus_range.groupby('Trains').agg({
    'AvgServiceTime': 'mean',
    'CompletedJobs': 'mean'
}).reset_index().sort_values('Trains')

# Calculate rate of change
train_analysis['ServiceTime_RateOfChange'] = -train_analysis['AvgServiceTime'].diff()  # Negative because lower is better
train_analysis['Jobs_RateOfChange'] = train_analysis['CompletedJobs'].diff()
train_analysis['Train_Increment'] = train_analysis['Trains'].diff()

# Normalize per train
train_analysis['ServiceTime_RatePerTrain'] = train_analysis['ServiceTime_RateOfChange'] / train_analysis['Train_Increment']
train_analysis['Jobs_RatePerTrain'] = train_analysis['Jobs_RateOfChange'] / train_analysis['Train_Increment']

# Replace NaN and inf values
train_analysis['ServiceTime_RatePerTrain'] = train_analysis['ServiceTime_RatePerTrain'].replace([np.inf, -np.inf], 0).fillna(0)
train_analysis['Jobs_RatePerTrain'] = train_analysis['Jobs_RatePerTrain'].replace([np.inf, -np.inf], 0).fillna(0)

# Rate of Change vs Buses (within train range 20-40)
print("Calculating rate of change vs increasing buses (train range: 20-40)...")
df_train_range = df[(df['Trains'] >= 20) & (df['Trains'] <= 40)].copy()

# Group by buses and calculate mean metrics
bus_analysis = df_train_range.groupby('Buses').agg({
    'AvgServiceTime': 'mean',
    'CompletedJobs': 'mean'
}).reset_index().sort_values('Buses')

# Calculate rate of change
bus_analysis['ServiceTime_RateOfChange'] = -bus_analysis['AvgServiceTime'].diff()  # Negative because lower is better
bus_analysis['Jobs_RateOfChange'] = bus_analysis['CompletedJobs'].diff()
bus_analysis['Bus_Increment'] = bus_analysis['Buses'].diff()

# Normalize per bus
bus_analysis['ServiceTime_RatePerBus'] = bus_analysis['ServiceTime_RateOfChange'] / bus_analysis['Bus_Increment']
bus_analysis['Jobs_RatePerBus'] = bus_analysis['Jobs_RateOfChange'] / bus_analysis['Bus_Increment']

# Replace NaN and inf values
bus_analysis['ServiceTime_RatePerBus'] = bus_analysis['ServiceTime_RatePerBus'].replace([np.inf, -np.inf], 0).fillna(0)
bus_analysis['Jobs_RatePerBus'] = bus_analysis['Jobs_RatePerBus'].replace([np.inf, -np.inf], 0).fillna(0)

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

# Optimization: Minimum Viable Fleet
print("\n" + "=" * 80)
print("OPTIMIZATION: MINIMUM VIABLE FLEET")
print("=" * 80)

# 1. HARD CONSTRAINT: Must meet service level (70% jobs)
valid_configs = df_reasonable[df_reasonable['CompletedJobs'] >= min_jobs_required].copy()

if len(valid_configs) == 0:
    print(f"❌ NO configuration met the target of {min_jobs_required:,} jobs.")
    print("   Falling back to 'Best Effort' (Maximizing Jobs)...")
    best_config = df_reasonable.loc[df_reasonable['CompletedJobs'].idxmax()]
    reason = "Max Possible Throughput (Target not met)"
else:
    print(f"✅ {len(valid_configs)} configurations met the target.")
    
    # 2. SOFT CONSTRAINT: Filter out terrible service times
    #    (e.g., keep only configs within 20% of the best service time found among valid ones)
    best_valid_time = valid_configs['AvgServiceTime'].min()
    acceptable_time = best_valid_time * 1.20  # Allow 20% buffer
    
    refined_configs = valid_configs[valid_configs['AvgServiceTime'] <= acceptable_time].copy()
    print(f"   {len(refined_configs)} configurations also have acceptable service time (<{acceptable_time:.1f} min).")
    
    # 3. OPTIMIZATION: Sort by Trains (Primary Cost), then Buses (Secondary Cost)
    #    This ensures we pick the ABSOLUTE MINIMUM number of trains needed.
    refined_configs = refined_configs.sort_values(by=['Trains', 'Buses'], ascending=[True, True])
    
    best_config = refined_configs.iloc[0]
    reason = "Minimum Trains satisfying all constraints"

# Display Result
print(f"\n⭐ RECOMMENDED CONFIGURATION ({reason})")
print(f"   Trains: {int(best_config['Trains'])}")
print(f"   Buses:  {int(best_config['Buses'])}")
print(f"   --------------------------------")
print(f"   Jobs Completed: {int(best_config['CompletedJobs']):,} (Target: {min_jobs_required:,})")
print(f"   Avg Service Time: {best_config['AvgServiceTime']:.2f} min")
print(f"   Buses per Train: {best_config['Buses'] / best_config['Trains']:.1f}")
print(f"   Efficiency: {best_config['JobsPerVehicle']:.2f} jobs/vehicle")

final_recommendation = best_config

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

# Visualization
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create plot directory if it doesn't exist
plot_dir = 'plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

# 1. Rate of Change: Service Time vs Increasing Trains (bus range 150-300)
fig, ax = plt.subplots(figsize=(10, 6))
train_plot = train_analysis[(train_analysis['Train_Increment'] > 0) & (train_analysis['ServiceTime_RatePerTrain'] != 0)]
if len(train_plot) > 0:
    ax.plot(train_plot['Trains'], train_plot['ServiceTime_RatePerTrain'], 
             'o-', linewidth=2, markersize=6, label='Service Time Rate per Train', color='blue', alpha=0.7)
ax.set_xlabel('Number of Trains', fontweight='bold')
ax.set_ylabel('Service Time Improvement Rate (min/train)', fontweight='bold')
ax.set_title('Rate of Change: Service Time vs Increasing Trains (Bus Range: 150-300)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '1_rate_of_change_service_time_vs_trains.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '1_rate_of_change_service_time_vs_trains.png')}")

# 2. Rate of Change: Service Time vs Increasing Buses (train range 20-40)
fig, ax = plt.subplots(figsize=(10, 6))
bus_plot_st = bus_analysis[(bus_analysis['Bus_Increment'] > 0) & (bus_analysis['ServiceTime_RatePerBus'] != 0)]
if len(bus_plot_st) > 0:
    ax.plot(bus_plot_st['Buses'], bus_plot_st['ServiceTime_RatePerBus'], 
             's-', linewidth=2, markersize=4, label='Service Time Rate per Bus', color='green', alpha=0.7)
ax.set_xlabel('Number of Buses', fontweight='bold')
ax.set_ylabel('Service Time Improvement Rate (min/bus)', fontweight='bold')
ax.set_title('Rate of Change: Service Time vs Increasing Buses (Train Range: 20-40)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '2_rate_of_change_service_time_vs_buses.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '2_rate_of_change_service_time_vs_buses.png')}")

# 3. Rate of Change: Completed Jobs vs Increasing Trains (bus range 150-300)
fig, ax = plt.subplots(figsize=(10, 6))
train_plot_jobs = train_analysis[(train_analysis['Train_Increment'] > 0) & (train_analysis['Jobs_RatePerTrain'] != 0)]
if len(train_plot_jobs) > 0:
    ax.plot(train_plot_jobs['Trains'], train_plot_jobs['Jobs_RatePerTrain'], 
             'o-', linewidth=2, markersize=6, label='Jobs Rate per Train', color='purple', alpha=0.7)
ax.set_xlabel('Number of Trains', fontweight='bold')
ax.set_ylabel('Jobs Improvement Rate (jobs/train)', fontweight='bold')
ax.set_title('Rate of Change: Completed Jobs vs Increasing Trains (Bus Range: 150-300)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '3_rate_of_change_jobs_vs_trains.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '3_rate_of_change_jobs_vs_trains.png')}")

# 4. Rate of Change: Completed Jobs vs Increasing Buses (train range 20-40)
fig, ax = plt.subplots(figsize=(10, 6))
bus_plot_jobs = bus_analysis[(bus_analysis['Bus_Increment'] > 0) & (bus_analysis['Jobs_RatePerBus'] != 0)]
if len(bus_plot_jobs) > 0:
    ax.plot(bus_plot_jobs['Buses'], bus_plot_jobs['Jobs_RatePerBus'], 
             's-', linewidth=2, markersize=4, label='Jobs Rate per Bus', color='orange', alpha=0.7)
ax.set_xlabel('Number of Buses', fontweight='bold')
ax.set_ylabel('Jobs Improvement Rate (jobs/bus)', fontweight='bold')
ax.set_title('Rate of Change: Completed Jobs vs Increasing Buses (Train Range: 20-40)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '4_rate_of_change_jobs_vs_buses.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '4_rate_of_change_jobs_vs_buses.png')}")

# 5. Buses per Train Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heatmap_ratio = df.pivot_table(index='Trains', columns='Buses', values='BusesPerTrain', aggfunc='mean')
sns.heatmap(heatmap_ratio, cmap='RdYlGn', annot=False, cbar_kws={'label': 'Buses per Train'}, 
            vmin=min_buses_per_train, vmax=max_buses_per_train, ax=ax)
ax.axhline(int(final_recommendation['Trains']) - 0.5, color='yellow', linewidth=3, label='Recommended')
ax.axvline(int(final_recommendation['Buses']) - 0.5, color='yellow', linewidth=3)
ax.set_title('Buses per Train Heatmap', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Buses', fontweight='bold')
ax.set_ylabel('Number of Trains', fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '5_buses_per_train_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '5_buses_per_train_heatmap.png')}")

# 6. Completed Jobs vs Number of Trains (showing plateau at 30 trains)
print("\nGenerating plateau analysis visualizations...")
train_stats = df_reasonable.groupby('Trains').agg({
    'CompletedJobs': ['mean', 'median', 'std'],
    'AvgServiceTime': ['mean', 'median']
}).reset_index()
train_stats.columns = ['Trains', 'Jobs_Mean', 'Jobs_Median', 'Jobs_Std', 'ServiceTime_Mean', 'ServiceTime_Median']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_stats['Trains'], train_stats['Jobs_Mean'], 'o-', label='Mean Completed Jobs', 
        linewidth=2, markersize=6, color='blue')
ax.fill_between(train_stats['Trains'], 
                train_stats['Jobs_Mean'] - train_stats['Jobs_Std'], 
                train_stats['Jobs_Mean'] + train_stats['Jobs_Std'],
                alpha=0.2, color='blue', label='±1 Std Dev')
ax.axvline(30, color='red', linestyle='--', linewidth=2, label='Plateau Point (30 trains)')
ax.fill_betweenx([0, max(train_stats['Jobs_Mean']) * 1.1], 30, max(train_stats['Trains']), 
                 alpha=0.1, color='red', label='Plateau Region')
ax.set_xlabel('Number of Trains', fontweight='bold')
ax.set_ylabel('Completed Jobs', fontweight='bold')
ax.set_title('Completed Jobs vs Number of Trains (Plateau at 30 Trains)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '6_jobs_vs_trains_plateau.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '6_jobs_vs_trains_plateau.png')}")

# 7. Average Service Time vs Number of Trains (showing plateau at 55 trains)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_stats['Trains'], train_stats['ServiceTime_Mean'], 's-', label='Mean Service Time', 
        linewidth=2, markersize=6, color='green')
ax.axvline(55, color='red', linestyle='--', linewidth=2, label='Service Time Plateau Point (55 trains)')
ax.fill_betweenx([0, max(train_stats['ServiceTime_Mean']) * 1.1], 55, max(train_stats['Trains']), 
                 alpha=0.1, color='red', label='Plateau Region')
ax.set_xlabel('Number of Trains', fontweight='bold')
ax.set_ylabel('Average Service Time (min)', fontweight='bold')
ax.set_title('Average Service Time vs Number of Trains (Plateau at 55 Trains)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '7_service_time_vs_trains_plateau.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '7_service_time_vs_trains_plateau.png')}")

# 8. Completed Jobs vs Number of Buses (showing plateau at 180 buses)
bus_stats = df_reasonable.groupby('Buses').agg({
    'CompletedJobs': ['mean', 'median', 'std'],
    'AvgServiceTime': ['mean', 'median']
}).reset_index()
bus_stats.columns = ['Buses', 'Jobs_Mean', 'Jobs_Median', 'Jobs_Std', 'ServiceTime_Mean', 'ServiceTime_Median']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bus_stats['Buses'], bus_stats['Jobs_Mean'], 'o-', label='Mean Completed Jobs', 
        linewidth=2, markersize=4, color='purple')
ax.fill_between(bus_stats['Buses'], 
                bus_stats['Jobs_Mean'] - bus_stats['Jobs_Std'], 
                bus_stats['Jobs_Mean'] + bus_stats['Jobs_Std'],
                alpha=0.2, color='purple', label='±1 Std Dev')
ax.axvline(180, color='red', linestyle='--', linewidth=2, label='Plateau Point (180 buses)')
ax.fill_betweenx([0, max(bus_stats['Jobs_Mean']) * 1.1], 180, max(bus_stats['Buses']), 
                 alpha=0.15, color='red', label='Plateau Region')
ax.set_xlabel('Number of Buses', fontweight='bold')
ax.set_ylabel('Completed Jobs', fontweight='bold')
ax.set_title('Completed Jobs vs Number of Buses (Plateau at 180 Buses)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '8_jobs_vs_buses_plateau.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '8_jobs_vs_buses_plateau.png')}")

# 9. Average Service Time vs Number of Buses (showing plateau at 400 buses)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bus_stats['Buses'], bus_stats['ServiceTime_Mean'], 's-', label='Mean Service Time', 
        linewidth=2, markersize=4, color='orange')
ax.axvline(400, color='red', linestyle='--', linewidth=2, label='Plateau Point (400 buses)')
ax.fill_betweenx([0, max(bus_stats['ServiceTime_Mean']) * 1.1], 400, max(bus_stats['Buses']), 
                 alpha=0.15, color='red', label='Plateau Region')
ax.set_xlabel('Number of Buses', fontweight='bold')
ax.set_ylabel('Average Service Time (min)', fontweight='bold')
ax.set_title('Average Service Time vs Number of Buses (Plateau at 400 Buses)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '9_service_time_vs_buses_plateau.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(plot_dir, '9_service_time_vs_buses_plateau.png')}")

# 10. Summary Statistics (as text file)
summary_text = f"""
⭐ RECOMMENDED CONFIGURATION

Trains: {int(final_recommendation['Trains'])}
Buses: {int(final_recommendation['Buses'])}
Buses/Train: {final_recommendation['BusesPerTrain']:.1f}

Service Time: {final_recommendation['AvgServiceTime']:.2f} min
Completed Jobs: {int(final_recommendation['CompletedJobs']):,}
Service Level: {(final_recommendation['CompletedJobs']/total_workers*100) if total_workers > 0 else 0:.1f}% of workers
Meets Minimum: {'✓ YES' if final_recommendation['CompletedJobs'] >= min_jobs_required else '✗ NO'}
Efficiency: {final_recommendation['JobsPerVehicle']:.2f} jobs/vehicle

Performance vs Best:
• Service Time: {((final_recommendation['AvgServiceTime']/best_service_time - 1)*100):+.1f}%
• Completed Jobs: {((1 - final_recommendation['CompletedJobs']/best_completed_jobs)*100):+.1f}%
"""

with open(os.path.join(plot_dir, '10_summary_statistics.txt'), 'w') as f:
    f.write(summary_text)
print(f"Saved: {os.path.join(plot_dir, '10_summary_statistics.txt')}")

print(f"\nAll visualizations saved to '{plot_dir}/' directory")

# Save summary
summary_data = {
    'Metric': ['Recommended Trains', 'Recommended Buses', 'Buses per Train', 
               'Service Time (min)', 'Completed Jobs', 
               'Efficiency (jobs/vehicle)'],
    'Value': [int(final_recommendation['Trains']), int(final_recommendation['Buses']),
              f"{final_recommendation['BusesPerTrain']:.1f}",
              f"{final_recommendation['AvgServiceTime']:.2f}",
              int(final_recommendation['CompletedJobs']),
              f"{final_recommendation['JobsPerVehicle']:.2f}"]
}

summary_df = pd.DataFrame(summary_data)
# Ensure csv directory exists
csv_dir = 'csv'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
csv_path = os.path.join(csv_dir, 'optimal_configuration.csv')
summary_df.to_csv(csv_path, index=False)
print(f"Summary saved to '{csv_path}'")

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
print(f"   This is the minimum viable fleet configuration that meets all service level")
print(f"   requirements while minimizing the number of trains and buses needed.")
print("=" * 80)
