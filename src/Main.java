import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) {
        //out.print("Please enter your station configuration file directory(if you don't have one, press enter to use the default):");

        //Scanner ln = new Scanner(System.in);
        //String stationConfigFile = ln.nextLine();
        //ln.close();
        //if(stationConfigFile.isEmpty()) {
        //    Path defaultStationsPath = Paths.get("csv", "stations.csv");
        // stationConfigFile = defaultStationsPath.toFile().getAbsolutePath();
       // }

       // for cloud run
       String stationConfigFileArg = args.length > 0 ? args[0] : "";
       final String stationConfigFile;
        if(stationConfigFileArg.isEmpty()) {
            Path defaultStationsPath = Paths.get("csv", "stations.csv");
            stationConfigFile = defaultStationsPath.toFile().getAbsolutePath();
        } else {
            stationConfigFile = stationConfigFileArg;
        }

        int numTrainsRange = 80;
        int numBusesRange = 800;
        
        // Use step sizes to reduce search space
        int trainStep = 1;  
        int busStep = 10;    // Test every 10th bus
        
        // Calculate number of configurations
        int numTrainValues = (numTrainsRange + trainStep - 1) / trainStep;  // Ceiling division
        int numBusValues = (numBusesRange + busStep - 1) / busStep;  // Ceiling division
        int totalConfigurations = numTrainValues * numBusValues - 320;
        
        String separator = "============================================================";
        System.out.println("\n" + separator);
        System.out.println("SIMULATION CONFIGURATION");
        System.out.println(separator);
        System.out.println("Train range: 1-" + numTrainsRange + " (step: " + trainStep + ")");
        System.out.println("Bus range: 1-" + numBusesRange + " (step: " + busStep + ")");
        System.out.println("Total configurations: " + totalConfigurations);
        System.out.println(separator + "\n");
        
        int[][] vehicleNumber = new int[totalConfigurations][2];
        int index = 0;
        for(int train = 5; train <= numTrainsRange; train += trainStep) {
            for(int bus = 10; bus <= numBusesRange; bus += busStep) {
                vehicleNumber[index][0] = train;
                vehicleNumber[index][1] = bus;
                index++;
            }
        }

        List<OutputDataConfig> results = new ArrayList<>();
        double simulationDuration = 1680.0; // 28 hours in minutes (24h generation + 4h clearance)
        
        // Determine optimal thread pool size (use available processors)
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Using " + numThreads + " threads for parallel execution\n");
        
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<OutputDataConfig>> futures = new ArrayList<>();
        
        // Submit all configuration tasks to thread pool
        int configNumber = 0;
        for (int[] num : vehicleNumber) {
            configNumber++;
            final int finalConfigNumber = configNumber;
            final int numTrains = num[0];
            final int numBuses = num[1];
            
            // Create a Callable task for each configuration
            Callable<OutputDataConfig> task = () -> {
                System.out.println("[" + finalConfigNumber + "/" + totalConfigurations + "] Starting: " + 
                                 numTrains + " trains, " + numBuses + " buses");
                
                SimulationConfig simulationConfig = new SimulationConfig(
                        numTrains,
                        1000,
                        250,
                        numBuses,
                        50,
                        75
                );

                Simulation simulation = new Simulation(stationConfigFile, simulationConfig);
                OutputDataConfig result = simulation.run(simulationDuration);
                
                System.out.println("[" + finalConfigNumber + "/" + totalConfigurations + "] Completed: " + 
                                 numTrains + " trains, " + numBuses + " buses");
                
                return result;
            };
            
            futures.add(executor.submit(task));
        }
        
        // Collect results from all futures (maintains order)
        for (int i = 0; i < futures.size(); i++) {
            try {
                results.add(futures.get(i).get());
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Error executing configuration " + (i + 1) + ": " + e.getMessage());
                e.printStackTrace();
                // Create a placeholder result to maintain order
                int[] num = vehicleNumber[i];
                results.add(new OutputDataConfig(num[0], num[1], 0.0, 0.0, 0));
            }
        }
        
        // Shutdown executor
        executor.shutdown();

        System.out.println("\nSimulation results: ");
        results.forEach(System.out::println);

        String csvPath = "csv/results.csv";
        writeResultsCsv(csvPath, results);

    }

    private static void writeResultsCsv(String filePath, List<OutputDataConfig> results) {
        // Ensure the directory exists
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        // header row
        String header = "Trains,Buses,AvgServiceTime,LongestServiceTime,CompletedJobs";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                writer.write(header);
                writer.newLine();

                for (OutputDataConfig o : results) {
                    // build a CSV line from the fields
                    String line = String.format(
                            "%d,%d,%.2f,%.2f,%d",
                            o.numTrains,
                            o.numBuses,
                            o.avgServiceTime,
                            o.longestServiceTime,
                            o.totalCompletedJobs
                    );
                    writer.write(line);
                    writer.newLine();
                }

            } catch (IOException e) {
                System.err.println("Error writing to CSV file: " + e.getMessage());
            }
    }
}

