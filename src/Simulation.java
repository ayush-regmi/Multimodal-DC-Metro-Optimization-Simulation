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

    public OutputDataConfig run(int stops) {
        List<BatchServerQueue> trains = new ArrayList<>();
        for(int i = 0; i < simConfig.numTrains; i++) {
            LoopingQueue<Station> newQueue = globalStationQueue.cloneQueue();
            int stationsPerTrain = globalStationQueue.length / simConfig.numTrains;
            for(int j = 0; j < i * stationsPerTrain; j++) { newQueue.dequeue(); }
            trains.add(new BatchServerQueue(trainInfo, newQueue));
        }
        
        // Track last departure time for each train to enforce headway
        double[] lastDepartureTime = new double[trains.size()];
        double minHeadway = 3.0; // 3 minutes minimum headway (in minutes, consistent with timeToTravel units)
        
        for (int i = 0; i <= stops; i++) {
            System.out.print("\r" + i + "/" + stops + " tasks completed. " + System.currentTimeMillis());
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
                
                double travelTime = t.stopAtStation(trainCurrentTime);
                travelTimes.put(t, travelTime);
                maxTravelTime = Math.max(travelTime, maxTravelTime);
                
                // Update last departure time
                lastDepartureTime[trainIdx] = trainCurrentTime;
            }
            currentTime += maxTravelTime;

            // Second pass: update offsets
            for (BatchServerQueue t : trains) {
                double travelTime = travelTimes.get(t);
                t.setTimeOffset(t.getTimeOffset() + travelTime - maxTravelTime);
            }
        }

        // collecting metrics across all trains
        int totalCompletedJobs = trains.stream().mapToInt(BatchServerQueue::getCompletedJobs).sum();
        double cumulativeServiceTime = trains.stream().mapToDouble(BatchServerQueue::getTotalServiceTime).sum();
        double avgServiceTime = totalCompletedJobs > 0 ? cumulativeServiceTime / totalCompletedJobs : 0.0;
        double longestServiceTime = trains.stream().mapToDouble(BatchServerQueue::getLongestServiceTime).max().orElse(0.0);

        return new OutputDataConfig(
                simConfig.numTrains,
                simConfig.numBusses,
                avgServiceTime,
                longestServiceTime,
                totalCompletedJobs
        );
    }
}
