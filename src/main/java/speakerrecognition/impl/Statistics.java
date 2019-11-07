package speakerrecognition.impl;


public final class Statistics {

    public static double getMean(double[] data) {
        double sum = 0.0;
        for (double a : data)
            sum += a;
        return sum / data.length;
    }

    public static double[] getMean(double[][] data) {
        int numOfRows = data.length;
        int numOfCols = data[0].length;

        double sum[] = new double[numOfCols];
        for (int j = 0; j < numOfCols; j++) {
            for (double[] datum : data) {
                //System.out.println(Double.toString(data[i][j]));
                sum[j] += datum[j];
            }
            sum[j] /= numOfRows;
        }
        //System.out.println("sumaaa");
        return sum;
    }

    public static double[] getVariance(double[][] data) {
        int numOfRows = data.length;
        int numOfCols = data[0].length;

        double[] means = Statistics.getMean(data);
        double[] temp = new double[numOfCols];

        for (int j = 0; j < numOfCols; j++) {
            for (double[] datum : data) {
                temp[j] += Math.pow((datum[j] - means[j]), 2);
            }
            temp[j] /= numOfRows;
        }

        return temp;
    }

}
