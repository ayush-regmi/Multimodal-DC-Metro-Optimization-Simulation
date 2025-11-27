# CMIS202 Capstone Project: High-Speed Rail and Bus Transportation Simulation

## Project Overview
This Java-based simulation models a high-speed rail and bus transportation network using queueing theory. It demonstrates how passengers flow through a multi-modal transit system and evaluates system performance under different loads and configurations.

## Features
- Distribution of worker departures and destinations based on real census data from the DC area, and can be quickly modified via a configuration file.
- Time-based simulation with peak hour modeling
- Realistic train headway constraints and station dwell times
- Bus route optimization based on worker population

## Running the Simulation

### Local Execution

1. **Compile the Java source files:**
   ```bash
   javac -d out/production/MultiModal-DC-Metro-Optimization-Simulation src/*.java
   ```

2. **Run the simulation:**
   ```bash
   java -cp out/production/MultiModal-DC-Metro-Optimization-Simulation Main
   ```
   
   Or with a custom station configuration file:
   ```bash
   java -cp out/production/MultiModal-DC-Metro-Optimization-Simulation Main /path/to/stations.csv
   ```

3. **Results will be saved to:** `csv/results.csv`

4. **Run optimization analysis:**
   ```bash
   pip install -r requirements.txt
   python optimized.py
   ```

### Cloud Execution (GitHub Actions) - Recommended for Long Runs

The simulation can be run on GitHub Actions for free, which is ideal for the full 200-configuration run (estimated 1.5-3 hours).

**Steps:**

1. **Push your code to GitHub** (if not already done)

2. **Trigger the workflow:**
   - Go to your repository on GitHub
   - Click on the "Actions" tab
   - Select "Run Simulation" workflow from the left sidebar
   - Click "Run workflow" button
   - Select the branch (usually `main` or `master`)
   - Click "Run workflow"

3. **Monitor progress:**
   - Click on the running workflow to see live logs
   - The simulation will show progress for each configuration

4. **Download results:**
   - Once complete, scroll to the bottom of the workflow run
   - Under "Artifacts", click "simulation-results"
   - Download the ZIP file containing:
     - `csv/results.csv` - Full simulation results
     - `optimal_configuration.csv` - Recommended configuration
     - `optimization_analysis.png` - Visualization (if generated)

**Benefits of Cloud Execution:**
- ✅ Free (GitHub Actions provides 2,000 free minutes/month for private repos, unlimited for public)
- ✅ Faster than most local machines
- ✅ Doesn't tie up your computer
- ✅ Results automatically saved as artifacts
- ✅ Reproducible environment

**Note:** The workflow will automatically trigger on pushes to `main`/`master` branch if you modify source files, or you can manually trigger it anytime.
