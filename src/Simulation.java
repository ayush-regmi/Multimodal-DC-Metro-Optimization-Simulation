import java.util.*;

public class Simulation {
    private double currentTime;
    private LoopingQueue<Station> globalStationQueue;
    VehicleInfo trainInfo;
    SimulationConfig simConfig;

    public Simulation(String stationConfigFile, SimulationConfig simConfigIn) {
        simConfig = simConfigIn;
        globalStationQueue = new LoopingQueue<Station>();
        currentTime = 0.0;
        Config config = new Config();
        List<Config.StationConfig> stationConfigs = config.getStationConfigs(stationConfigFile);
        
        // Calculate bus distribution based on worker population
        int[] busesPerStation = distributeBusesByWorkerPopulation(stationConfigs, simConfig.numBusses);
        
        for(int i = 0; i < stationConfigs.size(); i++) {
            Config.StationConfig s = stationConfigs.get(i);
            globalStationQueue.enqueue(new Station(
                    s.getStationName(),
                    s.getOriginDistance(),
                    s.getPopulation(),
                    s.getNumWorkers(),
                    new VehicleInfo(
                            simConfig.busCapacity,
                            busesPerStation[i],
                            simConfig.busSpeed
                    )
            ));
        }
        trainInfo = new VehicleInfo(simConfig.trainCapacity, simConfig.numTrains, simConfig.trainSpeed);
    }

    /**
     * Distributes buses to stations based on worker population percentage.
     * Larger stations (more workers) get more buses proportionally.
     * 
     * @param stationConfigs List of station configurations
     * @param totalBuses Total number of buses to distribute
     * @return Array of buses allocated to each station
     */
    private int[] distributeBusesByWorkerPopulation(List<Config.StationConfig> stationConfigs, int totalBuses) {
        int[] busesPerStation = new int[stationConfigs.size()];
        
        // Calculate total workers across all stations
        int totalWorkers = 0;
        for(Config.StationConfig s : stationConfigs) {
            totalWorkers += s.getNumWorkers();
        }
        
        if(totalWorkers == 0) {
            // If no workers, distribute evenly
            int busesPerStationEven = totalBuses / stationConfigs.size();
            for(int i = 0; i < stationConfigs.size(); i++) {
                busesPerStation[i] = busesPerStationEven;
            }
            // Distribute remainder
            int remainder = totalBuses % stationConfigs.size();
            for(int i = 0; i < remainder; i++) {
                busesPerStation[i]++;
            }
            return busesPerStation;
        }
        
        // Calculate percentage and distribute buses proportionally
        int allocatedBuses = 0;
        for(int i = 0; i < stationConfigs.size(); i++) {
            double workerPercentage = (double) stationConfigs.get(i).getNumWorkers() / totalWorkers;
            busesPerStation[i] = (int) Math.round(totalBuses * workerPercentage);
            allocatedBuses += busesPerStation[i];
        }
        
        // Handle rounding differences - ensure exactly totalBuses are allocated
        int difference = totalBuses - allocatedBuses;
        if(difference != 0) {
            // Distribute remainder to stations with most workers
            // Sort indices by worker count (descending)
            Integer[] indices = new Integer[stationConfigs.size()];
            for(int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            java.util.Arrays.sort(indices, (a, b) -> 
                Integer.compare(stationConfigs.get(b).getNumWorkers(), 
                               stationConfigs.get(a).getNumWorkers()));
            
            // Add/subtract remainder
            for(int i = 0; i < Math.abs(difference); i++) {
                int idx = indices[i % indices.length];
                if(difference > 0) {
                    busesPerStation[idx]++;
                } else {
                    busesPerStation[idx] = Math.max(0, busesPerStation[idx] - 1);
                }
            }
        }
        
        // Ensure minimum of 1 bus per station (if totalBuses >= number of stations)
        if(totalBuses >= stationConfigs.size()) {
            for(int i = 0; i < stationConfigs.size(); i++) {
                if(busesPerStation[i] == 0) {
                    busesPerStation[i] = 1;
                    // Take one bus from the station with most buses
                    int maxBusesIdx = 0;
                    for(int j = 1; j < stationConfigs.size(); j++) {
                        if(busesPerStation[j] > busesPerStation[maxBusesIdx]) {
                            maxBusesIdx = j;
                        }
                    }
                    busesPerStation[maxBusesIdx]--;
                }
            }
        }
        
        return busesPerStation;
    }

    /**
     * Runs the simulation for a specified duration in minutes.
     * 
     * @param simulationDurationMinutes Duration of simulation in minutes (e.g., 1440 for 24 hours)
     * @return OutputDataConfig containing simulation results
     */
    public OutputDataConfig run(double simulationDurationMinutes) {
        List<BatchServerQueue> trains = new ArrayList<>();
        int numStations = globalStationQueue.getLength();
        double minHeadway = 3.0; // 3 minutes minimum headway (in minutes, consistent with timeToTravel units)
        
        for(int i = 0; i < simConfig.numTrains; i++) {
            LoopingQueue<Station> newQueue = globalStationQueue.cloneQueue();
            
            // Distribute trains evenly across the route
            // If trains <= stations: divide stations among trains
            // If trains > stations: space trains evenly using modulo (prevents all starting at same position)
            int startingOffset;
            if (numStations >= simConfig.numTrains) {
                // Divide stations among trains
                int stationsPerTrain = numStations / simConfig.numTrains;
                startingOffset = i * stationsPerTrain;
            } else {
                // Space trains evenly throughout the loop using modulo
                // This ensures trains are distributed across all stations
                startingOffset = (i * numStations) / simConfig.numTrains;
            }
            
            // Position train at starting offset
            for(int j = 0; j < startingOffset; j++) {
                newQueue.dequeue();
            }
            BatchServerQueue train = new BatchServerQueue(trainInfo, newQueue);
            
            // Set initial time offset to space trains out evenly
            // This prevents all trains from starting at the same time
            // Distribute trains evenly across a time window based on min headway
            double initialTimeOffset = (i * minHeadway) / Math.max(1, simConfig.numTrains);
            train.setTimeOffset(initialTimeOffset);
            
            trains.add(train);
        }
        
        // Track last departure time for each train to enforce headway
        double[] lastDepartureTime = new double[trains.size()];
        
        double endTime = simulationDurationMinutes;
        int iterationCount = 0;
        
        // Get all stations for bus processing
        List<Station> allStations = new ArrayList<>();
        LoopingQueue<Station> tempQueue = globalStationQueue.cloneQueue();
        for(int i = 0; i < numStations; i++) {
            allStations.add(tempQueue.dequeue());
        }
        CityInfoHolder[] cityInfo = globalStationQueue.getStationNames();
        
        // Time-based simulation: run until we reach the end time
        while (currentTime < endTime) {
            iterationCount++;
            if (iterationCount % 100 == 0) {
                double progress = (currentTime / endTime) * 100;
                System.out.print("\r" + String.format("Simulation progress: %.1f%% (Time: %.1f/%.1f min)", 
                    progress, currentTime, endTime) + " " + System.currentTimeMillis());
            }
            
            // Process buses at all stations continuously (independent of train arrivals)
            // This allows buses to pick up passengers regularly, not just when trains arrive
            for (Station station : allStations) {
                station.getBusArrivals(currentTime, cityInfo);
            }
            
            double maxTravelTime = 0.0;
            Map<BatchServerQueue, Double> travelTimes = new HashMap<>();

            // First pass: calculate travel times and find max
            for (int trainIdx = 0; trainIdx < trains.size(); trainIdx++) {
                BatchServerQueue t = trains.get(trainIdx);
                
                // Enforce headway constraint - ensure minimum time between train departures
                double trainCurrentTime = currentTime + t.getTimeOffset();
                double timeSinceLastDeparture = trainCurrentTime - lastDepartureTime[trainIdx];
                
                if(timeSinceLastDeparture < minHeadway) {
                    // Wait until minimum headway is met
                    double waitTime = minHeadway - timeSinceLastDeparture;
                    t.setTimeOffset(t.getTimeOffset() + waitTime);
                    trainCurrentTime = currentTime + t.getTimeOffset();
                }
                
                // Check if this train would exceed simulation time
                double travelTime = t.stopAtStation(trainCurrentTime);
                travelTimes.put(t, travelTime);
                maxTravelTime = Math.max(travelTime, maxTravelTime);
                
                // Update last departure time
                lastDepartureTime[trainIdx] = trainCurrentTime;
            }
            
            // Check if next iteration would exceed simulation time
            if (currentTime + maxTravelTime > endTime) {
                // Stop before exceeding time limit
                break;
            }
            
            currentTime += maxTravelTime;

            // Second pass: update offsets
            for (BatchServerQueue t : trains) {
                double travelTime = travelTimes.get(t);
                t.setTimeOffset(t.getTimeOffset() + travelTime - maxTravelTime);
            }
        }
        
        System.out.print("\r" + String.format("Simulation complete: %.1f minutes simulated (%d iterations)", 
            currentTime, iterationCount) + "                    ");

        // Diagnostic: Collect job flow statistics
        long totalJobsGenerated = allStations.stream().mapToLong(Station::getJobsGenerated).sum();
        long totalJobsPickedUpByBuses = allStations.stream().mapToLong(Station::getJobsPickedUpByBuses).sum();
        long totalJobsPickedUpByTrains = trains.stream().mapToLong(BatchServerQueue::getJobsPickedUpByTrain).sum();
        long totalJobsRejectedWrongDirection = trains.stream().mapToLong(BatchServerQueue::getJobsRejectedWrongDirection).sum();
        int totalBusStopWaiters = allStations.stream().mapToInt(Station::getCurrentBusStopWaitersSize).sum();
        int totalStationWaiters = allStations.stream().mapToInt(Station::getCurrentStationWaitersSize).sum();
        int maxBusStopWaiters = allStations.stream().mapToInt(Station::getMaxBusStopWaitersSize).max().orElse(0);
        int maxStationWaiters = allStations.stream().mapToInt(Station::getMaxStationWaitersSize).max().orElse(0);
        
        // Train capacity utilization statistics
        long totalTrainStops = trains.stream().mapToLong(BatchServerQueue::getTotalStops).sum();
        long stopsAtFullCapacity = trains.stream().mapToLong(BatchServerQueue::getStopsAtFullCapacity).sum();
        double avgCapacityUtilization = trains.stream().mapToDouble(BatchServerQueue::getCapacityUtilization).average().orElse(0.0);
        
        // Print diagnostic information
        System.out.println("\n=== DIAGNOSTIC INFORMATION ===");
        System.out.println(String.format("Jobs Generated: %,d", totalJobsGenerated));
        System.out.println(String.format("Jobs Picked Up by Buses: %,d (%.1f%%)", 
            totalJobsPickedUpByBuses, 
            totalJobsGenerated > 0 ? (100.0 * totalJobsPickedUpByBuses / totalJobsGenerated) : 0.0));
        System.out.println(String.format("Jobs Picked Up by Trains: %,d (%.1f%% of bus pickups)", 
            totalJobsPickedUpByTrains,
            totalJobsPickedUpByBuses > 0 ? (100.0 * totalJobsPickedUpByTrains / totalJobsPickedUpByBuses) : 0.0));
        System.out.println(String.format("Jobs Rejected (Wrong Direction): %,d", totalJobsRejectedWrongDirection));
        System.out.println(String.format("Remaining at Bus Stops: %,d (Max: %,d)", totalBusStopWaiters, maxBusStopWaiters));
        System.out.println(String.format("Remaining at Stations: %,d (Max: %,d)", totalStationWaiters, maxStationWaiters));
        System.out.println(String.format("Train Capacity Utilization: %.1f%% (%.1f%% of stops at full capacity)", 
            avgCapacityUtilization * 100.0,
            totalTrainStops > 0 ? (100.0 * stopsAtFullCapacity / totalTrainStops) : 0.0));

        // collecting metrics across all trains
        int totalCompletedJobs = trains.stream().mapToInt(BatchServerQueue::getCompletedJobs).sum();
        double cumulativeServiceTime = trains.stream().mapToDouble(BatchServerQueue::getTotalServiceTime).sum();
        double avgServiceTime = totalCompletedJobs > 0 ? cumulativeServiceTime / totalCompletedJobs : 0.0;
        double longestServiceTime = trains.stream().mapToDouble(BatchServerQueue::getLongestServiceTime).max().orElse(0.0);
        
        System.out.println(String.format("Jobs Completed: %,d (%.1f%% of generated, %.1f%% of train pickups)", 
            totalCompletedJobs,
            totalJobsGenerated > 0 ? (100.0 * totalCompletedJobs / totalJobsGenerated) : 0.0,
            totalJobsPickedUpByTrains > 0 ? (100.0 * totalCompletedJobs / totalJobsPickedUpByTrains) : 0.0));
        
        // Calculate optimal train count based on job generation
        long expectedJobs = (long)(895266 * 0.70); // Approximate total workers (70% commute)
        System.out.println("\n=== JOB GENERATION ANALYSIS ===");
        System.out.println("Expected Jobs (70% of workers): " + expectedJobs);
        System.out.println("Actual Jobs Generated: " + totalJobsGenerated);
        if (totalJobsGenerated > expectedJobs * 1.5) {
            System.out.println("WARNING: Job generation is " + String.format("%.1f", (double)totalJobsGenerated / expectedJobs) + "x expected!");
        }
        
        // Calculate train requirements
        int trainCapacity = trainInfo.getVehicleCapacity();
        int numStationsForCalc = globalStationQueue.getLength();
        // Estimate: each train loops through all stations
        // Average time per station (travel + dwell): ~3.5 minutes
        double avgStationTime = 3.5;
        double loopTime = numStationsForCalc * avgStationTime; // minutes per loop
        double loopsPerDay = 1440.0 / loopTime; // loops per 24-hour day
        double trainStationVisitsPerDay = trains.size() * loopsPerDay * numStationsForCalc;
        
        if (trainStationVisitsPerDay > 0 && totalJobsPickedUpByBuses > 0) {
            double jobsPerVisit = (double)totalJobsPickedUpByBuses / trainStationVisitsPerDay;
            double utilization = (jobsPerVisit / trainCapacity) * 100.0;
            System.out.println("\n=== TRAIN CAPACITY ANALYSIS ===");
            System.out.println("Train-Station Visits per Day: " + String.format("%.0f", trainStationVisitsPerDay));
            System.out.println("Jobs per Visit: " + String.format("%.1f", jobsPerVisit));
            System.out.println("Current Utilization: " + String.format("%.1f", utilization) + "%");
            
            // Calculate optimal train count for 80% utilization
            double targetUtilization = 0.80;
            int optimalTrains80 = (int)Math.ceil((totalJobsPickedUpByBuses / trainStationVisitsPerDay) / (trainCapacity * targetUtilization) * trains.size());
            System.out.println("Recommended Trains (for 80% utilization): " + optimalTrains80);
            
            // Calculate optimal train count for 50% utilization (more realistic)
            targetUtilization = 0.50;
            int optimalTrains50 = (int)Math.ceil((totalJobsPickedUpByBuses / trainStationVisitsPerDay) / (trainCapacity * targetUtilization) * trains.size());
            System.out.println("Recommended Trains (for 50% utilization): " + optimalTrains50);
        }
        
        System.out.println("==============================\n");

        return new OutputDataConfig(
                simConfig.numTrains,
                simConfig.numBusses,
                avgServiceTime,
                longestServiceTime,
                totalCompletedJobs
        );
    }
}
