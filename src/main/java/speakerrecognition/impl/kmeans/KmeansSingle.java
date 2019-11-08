package speakerrecognition.impl.kmeans;

import speakerrecognition.math.Matrices;

class KmeansSingle {
    private double[][] best_centers = null;
    private double best_inertia = Double.MAX_VALUE;

    KmeansSingle(double[][] data, int n_clusters, double[] x_sq_norms, int max_iter, double tol, int numOfRows, int numOfCols) {
        double[][] centers = KMeans.initCentroids(data, n_clusters, x_sq_norms, numOfRows, numOfCols);
        double[] distances = new double[data.length];
        for (int i = 0; i < max_iter; i++) {
            double[][] centers_old = centers.clone();
            LabelsInertia labelsInertia = new LabelsInertia(data, x_sq_norms, centers, distances);
            int[] labels = labelsInertia.getLabels().clone();
            double inertia = labelsInertia.getInertia();
            distances = labelsInertia.getDistances().clone();
            centers = KMeans.centersDense(data, labels, n_clusters);
            if (inertia < best_inertia) {
                this.best_centers = centers.clone();
                this.best_inertia = inertia;
            }
            if (Matrices.squared_norm(Matrices.substractMatrices(centers_old, centers)) <= tol) {
                break;
            }
        }
    }

    double[][] getBestCenters() {
        return this.best_centers;
    }

    double getBestInertia() {
        return this.best_inertia;
    }
}
