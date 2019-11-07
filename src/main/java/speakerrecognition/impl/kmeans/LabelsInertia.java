package speakerrecognition.impl.kmeans;

import speakerrecognition.math.Matrices;

public class LabelsInertia {
    private int[] labels = null;
    private double[] distances = null;
    private double inertia = 0;

    public int[] getLabels() {
        return this.labels.clone();
    }

    public double getInertia() {
        return this.inertia;
    }

    public double[] getDistances() {
        return this.distances.clone();
    }

    LabelsInertia(double[][] X, double[] x_squared_norms, double[][] centers, double[] distances) {
        this.distances = distances;

        int n_samples = X.length;
        int[] labels = new int[n_samples];
        labels = Matrices.addValue(labels, -1);

        LabelsPrecomputeDence result = new LabelsPrecomputeDence(X, x_squared_norms, centers, this.distances);
        this.labels = result.labels.clone();
        this.inertia = result.inertia;
        this.distances = result.distances.clone();

    }
}
