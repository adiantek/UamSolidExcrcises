package speakerrecognition.impl.kmeans;

import speakerrecognition.math.Matrices;

public class LabelsPrecomputeDence {
    public int[] labels = null;
    public double[] distances = null;
    public double inertia = 0;

    LabelsPrecomputeDence(double[][] X, double[] x_squared_norms, double[][] centers, double[] distances) {
        this.distances = distances; ////////// huston, problem - k_means.py line 490, niejawne zwracanie

        int n_samples = X.length;
        int k = centers.length;
        double[][] all_distances = Matrices.euclidean_distances(centers, X, x_squared_norms);
        this.labels = new int[n_samples];
        this.labels = Matrices.addValue(this.labels, -1);
        double[] mindist = new double[n_samples];
        mindist = Matrices.addValue(mindist, Double.POSITIVE_INFINITY);

        for (int center_id = 0; center_id < k; center_id++) {
            double[] dist = all_distances[center_id];
            for (int i = 0; i < labels.length; i++) {
                if (dist[i] < mindist[i]) {
                    this.labels[i] = center_id;
                }
                mindist[i] = Math.min(dist[i], mindist[i]);
            }
        }
        if (n_samples == this.distances.length)
            this.distances = mindist;
        this.inertia = Matrices.sum(mindist);
    }
}
