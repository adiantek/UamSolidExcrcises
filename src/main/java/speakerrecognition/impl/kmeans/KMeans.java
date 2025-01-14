package speakerrecognition.impl.kmeans;


import speakerrecognition.math.Matrices;
import speakerrecognition.math.Statistics;

import java.util.Arrays;

public class KMeans {
    private int numOfClusters;
    private int numOfRows;
    private int numOfCols;
    private double[][] data;
    private double tol = 0.0001;
    private double[][] best_cluster_centers;
    private double best_inertia = Double.MAX_VALUE;

    public KMeans(double[][] x, int numOfClust) {
        this.numOfClusters = numOfClust;
        double tol = 0.0001;
        this.tol = tolerance(x, tol);
        this.numOfRows = x.length;
        this.numOfCols = x[0].length;
        this.data = Matrices.copy2dArray(x);
        this.best_cluster_centers = new double[numOfClust][x[0].length];
    }

    public void fit() {
        double[][] clusterCenters;
        double inertia = 0;
        double[] X_mean = Statistics.getMean(data);
        for (int i = 0; i < this.numOfRows; i++) {
            for (int j = 0; j < this.numOfCols; j++) {
                data[i][j] -= X_mean[j];
            }
        }
        double[] xSquaredNorms = Matrices.einsum(data);
        int n_init = 10;
        for (int i = 0; i < n_init; i++) {
            int max_iter = 300;
            KmeansSingle kmeansSingle = new KmeansSingle(this.data, this.numOfClusters, xSquaredNorms, max_iter, this.tol, numOfRows, numOfCols);
            clusterCenters = kmeansSingle.getBestCenters().clone();
            inertia = kmeansSingle.getBestInertia();
            if (inertia < this.best_inertia) {
                this.best_inertia = inertia;
                this.best_cluster_centers = clusterCenters.clone();
            }
        }
        this.best_cluster_centers = Matrices.addValue(this.best_cluster_centers, X_mean);
    }

    public double[][] getCenters() {
        return this.best_cluster_centers;
    }

    static double[][] centersDense(double[][] X, int[] labels, int n_clusters) {
        double[][] result = new double[n_clusters][X[0].length];
        for (int j = 0; j < X[0].length; j++) {
            double[] sum = new double[n_clusters];
            for (int k = 0; k < n_clusters; k++) {
                int samples_num = 0;
                for (int z = 0; z < labels.length; z++) {
                    if (labels[z] == k) {
                        sum[k] += X[z][j];
                        samples_num += 1;
                    }
                }
                sum[k] /= samples_num;
            }
            for (int i = 0; i < n_clusters; i++)
                result[i][j] = sum[i];
        }
        return result;
    }

    static double[][] initCentroids(double[][] data, int n_clusters, double[] x_sq_norms, int numOfRows, int numOfCols) {
        double[][] centers = new double[n_clusters][numOfCols];

        int n_local_trials = 2 + (int) (Math.log(n_clusters));
        /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
        int center_id = (int) Math.floor(Math.random() * numOfRows);
        if (numOfCols >= 0) System.arraycopy(data[center_id], 0, centers[0], 0, numOfCols);
        double[] closest_dist_sq = Matrices.euclideanDistances(centers[0], data, x_sq_norms);
        double current_pot = Matrices.sum(closest_dist_sq);

        for (int c = 1; c < n_clusters; c++) {
            /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
            double[] rand_vals = Matrices.genRandMatrix(current_pot, n_local_trials);
            double[] closest_dist_sq_cumsum = Matrices.cumsum(closest_dist_sq);
            int[] candidate_ids = Matrices.searchsorted(closest_dist_sq_cumsum, rand_vals);
            double[][] data_candidates = new double[n_local_trials][numOfCols];

            for (int z = 0; z < n_local_trials; z++) {
                if (numOfCols >= 0)
                    System.arraycopy(data[candidate_ids[z]], 0, data_candidates[z], 0, numOfCols);
            }

            int best_candidate = -1;
            double best_pot = Double.POSITIVE_INFINITY;
            double[] best_dist_sq = null;

            double[][] distance_to_candidates = Matrices.euclideanDistances(data_candidates, data, x_sq_norms);

            for (int trial = 0; trial < n_local_trials; trial++) {
                double[] new_dist_sq = Matrices.minimum(closest_dist_sq, Matrices.select_row(distance_to_candidates, trial));
                double new_pot = Matrices.sum(new_dist_sq);

                if (best_candidate == -1 || new_pot < best_pot) {
                    best_candidate = candidate_ids[trial];
                    best_pot = new_pot;
                    best_dist_sq = Arrays.copyOf(new_dist_sq, new_dist_sq.length);
                }
            }
            double[] center_temp = Arrays.copyOf(data[best_candidate], data[best_candidate].length);
            System.arraycopy(center_temp, 0, centers[c], 0, centers[0].length);
            current_pot = best_pot;
            closest_dist_sq = Arrays.copyOf(best_dist_sq, best_dist_sq.length);
        }

        return centers;

    }


    private double tolerance(double[][] x, double tol) {
        double[] temp = Statistics.getVariance(x);

        for (int i = 0; i < temp.length; i++) {
            temp[i] = temp[i] * tol;
        }
        return Statistics.getMean(temp);
    }

}
