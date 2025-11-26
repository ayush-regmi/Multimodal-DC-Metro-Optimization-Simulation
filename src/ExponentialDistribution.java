import java.util.Random;

public class ExponentialDistribution extends RandomDistribution {
    private double lambda;
    private Random random; // Reuse Random object for better performance

    public ExponentialDistribution(double lambda) { 
        this.lambda = lambda;
        this.random = new Random();
    }
    
    // Constructor that accepts a Random object for reuse
    public ExponentialDistribution(double lambda, Random random) {
        this.lambda = lambda;
        this.random = random;
    }

    @Override public double sample() {
        // Reuse existing Random object instead of creating new one
        return -(1 / lambda) * Math.log(random.nextDouble());
    }
}