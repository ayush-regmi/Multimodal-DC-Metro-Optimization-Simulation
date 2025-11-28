import java.util.Random;

public class Station {
    private String name;
    private double distanceFromOriginStation; //In kilometers
    private int population;
    private int numWorkers;
    private VehicleInfo busInfo;
    public Queue<Job> stationWaiters;

    private final double lambda;
    private Queue<Job> busStopWaiters = new Queue<>();
    private Random r;
    
    // Diagnostic counters
    private long jobsGenerated = 0;
    private long jobsPickedUpByBuses = 0;
    private int maxBusStopWaitersSize = 0;
    private int maxStationWaitersSize = 0;
    
    // Job generation diagnostics
    private double totalTimeGenerated = 0.0; // Track total time period for which jobs were generated
    private int generationCalls = 0; // Track how many times generateBusStopWaiters is called
    private double minInterArrivalTime = Double.MAX_VALUE;
    private double maxInterArrivalTime = 0.0;

    public String getName() { return name; }
    public int getPopulation() { return population; }
    public int getNumWorkers() { return numWorkers; }
    public double getDistanceFromOriginStation() { return distanceFromOriginStation; }

    private double lastPickupTime;

    public double getLastPickupTime() { return lastPickupTime; }
    
    // Diagnostic getters
    public long getJobsGenerated() { return jobsGenerated; }
    public long getJobsPickedUpByBuses() { return jobsPickedUpByBuses; }
    public int getMaxBusStopWaitersSize() { return maxBusStopWaitersSize; }
    public int getMaxStationWaitersSize() { return maxStationWaitersSize; }
    public int getCurrentBusStopWaitersSize() { return busStopWaiters.getLength(); }
    public int getCurrentStationWaitersSize() { return stationWaiters.getLength(); }
    
    // Additional diagnostic getters
    public double getTotalTimeGenerated() { return totalTimeGenerated; }
    public int getGenerationCalls() { return generationCalls; }
    public double getMinInterArrivalTime() { return minInterArrivalTime == Double.MAX_VALUE ? 0.0 : minInterArrivalTime; }
    public double getMaxInterArrivalTime() { return maxInterArrivalTime; }
    public double getExpectedJobsForTimePeriod() {
        // Calculate expected jobs based on lambda and total time generated
        // This accounts for time-of-day multipliers (approximate)
        return lambda * totalTimeGenerated * 1.22; // 1.22 is average multiplier
    }

    public Station(String stationName, double originDistance, int pop, int numWorkers, VehicleInfo busInfoIn) {
        r = new Random();
        name = stationName;
        distanceFromOriginStation = originDistance;
        population = pop;
        busInfo = busInfoIn;
        stationWaiters = new Queue<Job>();
        this.numWorkers = numWorkers;

        lastPickupTime = 0;
        // Base lambda calculation - arrivals per minute
        // Based on numWorkers (commuters), not total population
        // Assumptions:
        // - 70% of workers commute per day (one trip each)
        // - Commuters spread uniformly over 24 hours (base rate)
        // - Time-of-day multipliers adjust the rate (peak 2.5x, off-peak 1.0x, late night 0.3x)
        // - Formula: base_lambda = (numWorkers * commute_rate) / (24_hours * 60_min)
        // - This gives the average arrival rate; time multipliers are applied during generation
        double commuteRate = 0.70; // 70% of workers commute
        double totalMinutesPerDay = 24.0 * 60.0; // 1440 minutes
        lambda = (getNumWorkers() * commuteRate) / totalMinutesPerDay;
    }
    
    /**
     * Calculates the demand multiplier based on time of day.
     * Peak hours (7-9 AM and 4-6 PM) have higher demand.
     * 
     * @param currentTime Time in minutes from simulation start
     * @return Multiplier for arrival rate (1.0 = base, >1.0 = peak hours)
     */
    private double getTimeOfDayMultiplier(double currentTime) {
        // Convert minutes to hours of day (0-24)
        double hoursOfDay = (currentTime % 1440.0) / 60.0; // 1440 minutes = 24 hours
        
        // Morning peak: 7:00 AM - 9:00 AM (420-540 minutes)
        // Evening peak: 4:00 PM - 6:00 PM (960-1080 minutes)
        boolean isMorningPeak = (hoursOfDay >= 7.0 && hoursOfDay < 9.0);
        boolean isEveningPeak = (hoursOfDay >= 16.0 && hoursOfDay < 18.0);
        
        if (isMorningPeak || isEveningPeak) {
            // Peak hours: 2.5x base demand
            return 2.5;
        } else if (hoursOfDay >= 6.0 && hoursOfDay < 7.0) {
            // Pre-morning peak: 1.5x base demand
            return 1.5;
        } else if (hoursOfDay >= 9.0 && hoursOfDay < 10.0) {
            // Post-morning peak: 1.5x base demand
            return 1.5;
        } else if (hoursOfDay >= 15.0 && hoursOfDay < 16.0) {
            // Pre-evening peak: 1.5x base demand
            return 1.5;
        } else if (hoursOfDay >= 18.0 && hoursOfDay < 19.0) {
            // Post-evening peak: 1.5x base demand
            return 1.5;
        } else if (hoursOfDay >= 22.0 || hoursOfDay < 5.0) {
            // Late night/early morning: 0.3x base demand
            return 0.3;
        } else {
            // Off-peak hours: 1.0x base demand
            return 1.0;
        }
    }

    public void generateBusStopWaiters(double startTime, double endTime, CityInfoHolder[] cityInfo) {
        // Maximum queue size to prevent memory issues (safety limit)
        final int MAX_QUEUE_SIZE = 1000000; // 1 million jobs max per queue
        
        // Don't generate more jobs if queue is already too large
        if (busStopWaiters.getLength() >= MAX_QUEUE_SIZE) {
            return; // Queue is full, stop generating
        }
        
        // Safety check: prevent generating jobs if time period is invalid
        if (endTime <= startTime || endTime - startTime > 1440.0) {
            return; // Invalid time period or longer than 24 hours
        }
        
        // Diagnostic: Track generation calls and time periods
        generationCalls++;
        double timePeriod = endTime - startTime;
        totalTimeGenerated += timePeriod;
        
        // Maximum generation rate limit: prevent generating more than X jobs per minute
        // This is a safety limit to prevent runaway generation
        final double MAX_JOBS_PER_MINUTE = lambda * 5.0; // 5x base rate maximum
        final int MAX_JOBS_FOR_PERIOD = (int)(MAX_JOBS_PER_MINUTE * timePeriod);
        
        double localCurrentTime = startTime;
        double currentMultiplier = -1.0; // Track current multiplier to avoid recreating distribution
        ExponentialDistribution currentDistribution = null;
        int jobsGeneratedThisCall = 0;
        
        while (localCurrentTime < endTime) {
            // Check queue size limit to prevent memory exhaustion
            if (busStopWaiters.getLength() >= MAX_QUEUE_SIZE) {
                break; // Stop generating if queue is too large
            }
            
            // Check maximum generation rate limit
            if (jobsGeneratedThisCall >= MAX_JOBS_FOR_PERIOD) {
                break; // Stop generating if we've exceeded the maximum rate
            }
            
            // Get time-of-day multiplier for current time
            double timeMultiplier = getTimeOfDayMultiplier(localCurrentTime);
            
            // Only create new distribution if multiplier changed
            if (timeMultiplier != currentMultiplier || currentDistribution == null) {
                // Adjust arrival rate based on time of day
                // Higher multiplier = more frequent arrivals (shorter inter-arrival time)
                double adjustedLambda = lambda * timeMultiplier;
                
                // Cap lambda to prevent excessive generation rates
                // Even during peak, don't exceed 5x base rate
                double maxLambda = lambda * 5.0;
                if (adjustedLambda > maxLambda) {
                    adjustedLambda = maxLambda;
                }
                
                // Reuse Random object from Station to avoid creating new ones
                currentDistribution = new ExponentialDistribution(adjustedLambda, r);
                currentMultiplier = timeMultiplier;
            }
            
            double nextArrival = currentDistribution.sample();
            
            // Track inter-arrival times for diagnostics
            if (nextArrival < minInterArrivalTime) {
                minInterArrivalTime = nextArrival;
            }
            if (nextArrival > maxInterArrivalTime) {
                maxInterArrivalTime = nextArrival;
            }
            
            // Sanity check: if inter-arrival time is extremely small, cap it
            // This prevents generating millions of jobs in a very short time
            final double MIN_INTER_ARRIVAL_TIME = 0.0001; // 0.006 seconds minimum
            if (nextArrival < MIN_INTER_ARRIVAL_TIME) {
                nextArrival = MIN_INTER_ARRIVAL_TIME;
            }
            
            Job job = new Job(localCurrentTime, getName(), pickStation(cityInfo));
            busStopWaiters.enqueue(job);
            jobsGenerated++;
            jobsGeneratedThisCall++;
            
            // Track maximum queue size
            int currentSize = busStopWaiters.getLength();
            if (currentSize > maxBusStopWaitersSize) {
                maxBusStopWaitersSize = currentSize;
            }

            localCurrentTime += nextArrival;
            if (localCurrentTime >= endTime) break;
        }
    }

    public void getBusArrivals(double currentTime, CityInfoHolder[] cityInfo) {
        // Safety check: prevent generating jobs if time hasn't advanced
        if (currentTime <= getLastPickupTime()) {
            return; // No time has passed, don't generate jobs
        }
        
        generateBusStopWaiters(getLastPickupTime(), currentTime, cityInfo);
        double busTime = getLastPickupTime();
        
        // Maximum queue size for station waiters to prevent memory issues
        final int MAX_STATION_QUEUE_SIZE = 1000000; // 1 million jobs max

        while (busTime < currentTime) {
            int numVehicles = busInfo.getNumVehicles();
            int capacityPerBus = busInfo.getVehicleCapacity();

            for (int i = 0; i < numVehicles; i++) {
                int count = 0;

                while (!busStopWaiters.isQueueEmpty() && count < capacityPerBus) {
                    // Check station queue size limit to prevent memory issues
                    if (stationWaiters.getLength() >= MAX_STATION_QUEUE_SIZE) {
                        break; // Stop picking up if station queue is too large
                    }
                    
                    Job job = busStopWaiters.current.value;
                    if (job.getTimeOfCreation() <= busTime) {
                        stationWaiters.enqueue(busStopWaiters.dequeue());
                        count++;
                        jobsPickedUpByBuses++;
                        
                        // Track maximum station waiters size
                        int currentSize = stationWaiters.getLength();
                        if (currentSize > maxStationWaitersSize) {
                            maxStationWaitersSize = currentSize;
                        }
                    } else {
                        break;
                    }
                }
            }

            // Calculate realistic bus travel time based on station area and population
            // Buses pick up passengers from local bus stops and bring them to the train station
            // Travel distance is based on station area (estimated from population density)
            // Average bus route distance: 2-5 km for local bus routes to train stations
            double averageBusRouteDistance = calculateBusRouteDistance(cityInfo);
            double travelTime = (averageBusRouteDistance / busInfo.getVehicleSpeed()) * 60; // Convert to minutes
            
            // Add bus stop dwell time (10-20 seconds per stop, average 3-5 stops per route)
            double busDwellTime = (15.0 * 4.0) / 60.0; // 15 seconds * 4 stops = 1 minute
            travelTime += busDwellTime;
            
            busTime += travelTime;
        }
        lastPickupTime = currentTime; //This line of code fixed an issue I dealt with 5 hours because I am dumb -C
    }

    /**
     * Calculates realistic bus route distance based on station characteristics.
     * Larger stations (more population) have larger service areas, so longer bus routes.
     * 
     * @param cityInfo Array of city information for context
     * @return Average bus route distance in kilometers
     */
    private double calculateBusRouteDistance(CityInfoHolder[] cityInfo) {
        // Base distance: 2 km for small stations
        // Scale with population: larger stations have more bus stops and longer routes
        // Typical local bus route to train station: 2-5 km
        
        double baseDistance = 2.0; // km
        double maxDistance = 5.0; // km
        
        // Normalize population to 0-1 scale (using a reasonable max population)
        // Assuming max population around 700k (DC), scale accordingly
        double maxPopulation = 700000.0;
        double populationFactor = Math.min(1.0, (double) population / maxPopulation);
        
        // Calculate distance: base + (max - base) * population factor
        double routeDistance = baseDistance + (maxDistance - baseDistance) * populationFactor;
        
        // Add some variation based on station density
        // Urban stations (DC) have shorter routes, suburban stations have longer routes
        if (distanceFromOriginStation > 50) {
            // Suburban station - slightly longer routes
            routeDistance *= 1.2;
        } else if (distanceFromOriginStation < 20) {
            // Urban station - slightly shorter routes
            routeDistance *= 0.9;
        }
        
        // Ensure within reasonable bounds
        return Math.max(1.5, Math.min(6.0, routeDistance));
    }

    private String pickStation(CityInfoHolder[] cityInfo) {
        int totalWorkers = 0;
        for(CityInfoHolder c : cityInfo) {
            if(!c.getName().equals(getName())) totalWorkers += c.getNumWorkers();
        }

        if (totalWorkers == 0) return null;

        int randInt = r.nextInt(totalWorkers);
        for(CityInfoHolder c : cityInfo) {
            if(!c.getName().equals(getName())) {
                randInt -= c.getNumWorkers();
                if(randInt < 0) return c.getName();
            }
        }
        return null;
    }

    public static UnitTestResult UnitTest() {
        UnitTestResult result = new UnitTestResult("Station");

        // Test 1: Constructor initializes fields correctly
        try {
            Station station = new Station("TestStation", 5.5, 2000, 500, new VehicleInfo(20, 10, 30));
            assert station.getName().equals("TestStation") : "Name not initialized correctly";
            assert station.getDistanceFromOriginStation() == 5.5 : "Distance not initialized correctly";
            assert station.getPopulation() == 2000 : "Population not initialized correctly";
            assert station.getNumWorkers() == 500 : "NumWorkers not initialized correctly";
            assert station.getLastPickupTime() == 0 : "LastPickupTime not initialized to 0";
            result.recordNewTask(true);
        } catch (AssertionError e) {
            result.recordNewTask(false);
        }

        // Test 2: pickStation selects the only other city
        try {
            Station station = new Station("A", 0, 100, 50, new VehicleInfo(10, 5, 20));
            CityInfoHolder[] cities = { new CityInfoHolder("A", 50, 0), new CityInfoHolder("B", 100, 0) };
            String picked = station.pickStation(cities);
            assert picked != null && picked.equals("B") : "pickStation did not select the only other city";
            result.recordNewTask(true);
        } catch (AssertionError e) {
            result.recordNewTask(false);
        }

        // Test 3: getBusArrivals updates lastPickupTime
        try {
            Station station = new Station("Test", 0, 1000, 500, new VehicleInfo(10, 5, 50));
            CityInfoHolder[] cities = { new CityInfoHolder("Test", 500, 0) };
            station.getBusArrivals(10.0, cities);
            assert station.getLastPickupTime() > 0 : "lastPickupTime not updated after getBusArrivals";
            result.recordNewTask(true);
        } catch (AssertionError e) {
            result.recordNewTask(false);
        }

        // Test 4: generateBusStopWaiters with zero time range creates no jobs
        try {
            Station station = new Station("Test", 0, 1000, 500, new VehicleInfo(10, 5, 50));
            CityInfoHolder[] cities = { new CityInfoHolder("Test", 500, 0) };
            station.getBusArrivals(0.0, cities);
            assert station.stationWaiters.isQueueEmpty() : "stationWaiters should be empty when time range is zero";
            result.recordNewTask(true);
        } catch (AssertionError e) {
            result.recordNewTask(false);
        }

        // Test 5: Bus capacity limits jobs moved to stationWaiters (example with capacity 2)
        try {
            VehicleInfo busInfo = new VehicleInfo(2, 5, 50);
            Station station = new Station("Test", 0, 100000, 500, busInfo); // High population to generate jobs
            CityInfoHolder[] cities = { new CityInfoHolder("Test", 500, 0), new CityInfoHolder("Other", 500, 0) };

            // Assuming generateBusStopWaiters produces jobs; process up to time 10
            station.getBusArrivals(10.0, cities);

            // Check if the number of jobs in stationWaiters does not exceed possible bus arrivals
            // This is a heuristic check as exact numbers depend on random distributions
            int jobCount = 0;
            while (!station.stationWaiters.isQueueEmpty()) {
                station.stationWaiters.dequeue();
                jobCount++;
            }
            int maxPossible = (int) (10.0 / busInfo.getTravelTime(5, 18)) * busInfo.getVehicleCapacity();
            assert jobCount <= maxPossible : "stationWaiters exceeded expected job count based on bus capacity";
            result.recordNewTask(true);
        } catch (AssertionError e) {
            result.recordNewTask(false);
        }

        return result;
    }
}
