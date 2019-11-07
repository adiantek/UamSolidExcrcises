package speakerrecognition.math;


import java.util.Arrays;

public final class Statistics {

    public static double getMean(double[] data) {
        return Arrays.stream(data).average().orElse(0);
    }

    public static double[] getMean(double[][] data) {
        int numOfRows = data.length;
        int numOfCols = data[0].length;

        double[] v = new double[numOfCols];
        for (int j = 0; j < numOfCols; j++) {
            for (double[] datum : data) {
                v[j] += datum[j];
            }
            v[j] /= numOfRows;
        }
        return v;
    }

    public static double[] getVariance(double[][] data) {
        int numOfRows = data.length;
        int numOfCols = data[0].length;

        double[] means = getMean(data);
        double[] v = new double[numOfCols];

        for (int j = 0; j < numOfCols; j++) {
            for (double[] datum : data) {
                v[j] += (datum[j] - means[j]) * (datum[j] - means[j]);
            }
            v[j] /= numOfRows;
        }

        return v;
    }
}