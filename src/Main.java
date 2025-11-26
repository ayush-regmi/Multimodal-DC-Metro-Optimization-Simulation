import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;


import static java.lang.System.out;

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
       String stationConfigFile = args.length > 0 ? args[0] : "";
        if(stationConfigFile.isEmpty()) {
            Path defaultStationsPath = Paths.get("csv", "stations.csv");
            stationConfigFile = defaultStationsPath.toFile().getAbsolutePath();
        }

        int numTrainsRange = 20;
        int numBusesRange = 500;
        
        // Use step sizes to reduce search space
        int trainStep = 2;   // Test every 2nd train (1, 3, 5, 7, 9, ...)
        int busStep = 25;    // Test every 25th bus (1, 26, 51, 76, ...)
        
        // Calculate number of configurations
        int numTrainValues = (numTrainsRange + trainStep - 1) / trainStep;  // Ceiling division
        int numBusValues = (numBusesRange + busStep - 1) / busStep;  // Ceiling division
        int totalConfigurations = numTrainValues * numBusValues;
        
        String separator = "============================================================";
        System.out.println("\n" + separator);
        System.out.println("SIMULATION CONFIGURATION");
        System.out.println(separator);
        System.out.println("Train range: 1-" + numTrainsRange + " (step: " + trainStep + ")");
        System.out.println("Bus range: 1-" + numBusesRange + " (step: " + busStep + ")");
        System.out.println("Total configurations: " + totalConfigurations);
        System.out.println("Simulation duration: 24 hours (1440 minutes)");
        System.out.println(separator + "\n");
        
        int[][] vehicleNumber = new int[totalConfigurations][2];
        int index = 0;
        for(int train = 1; train <= numTrainsRange; train += trainStep) {
            for(int bus = 1; bus <= numBusesRange; bus += busStep) {
                vehicleNumber[index][0] = train;
                vehicleNumber[index][1] = bus;
                index++;
            }
        }

        List<OutputDataConfig> results = new ArrayList<>();
        double simulationDuration = 1440.0; // 24 hours in minutes
        
        int configNumber = 0;
        for (int[] num : vehicleNumber) {
            configNumber++;
            int numTrains = num[0];
            int numBuses  = num[1];
            
            System.out.println("\n[" + configNumber + "/" + totalConfigurations + "] Testing: " + 
                             numTrains + " trains, " + numBuses + " buses");

            SimulationConfig simulationConfig = new SimulationConfig(
                    numTrains,
                    500,
                    250,
                    numBuses,
                    50,
                    75
            );

            Simulation simulation = new Simulation(stationConfigFile, simulationConfig);
            results.add(simulation.run(simulationDuration));
            System.out.println();
        }

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

