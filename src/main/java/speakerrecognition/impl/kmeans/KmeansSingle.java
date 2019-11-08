package speakerrecognition.impl.kmeans;

import speakerrecognition.math.Matrices;

class KmeansSingle {
    private double[][] bestCenters = null;
    private double bestInertia = Double.MAX_VALUE;

    KmeansSingle(double[][] data, int nClusters, double[] xSqNorms, int max_iter, double tol, int numOfRows, int numOfCols) {
        double[][] centers = KMeans.initCentroids(data, nClusters, xSqNorms, numOfRows, numOfCols);
        double[] distances = new double[data.length];
        for (int i = 0; i < max_iter; i++) {
            double[][] centersOld = centers.clone();
            LabelsInertia labelsInertia = new LabelsInertia(data, xSqNorms, centers, distances);
            int[] labels = labelsInertia.getLabels().clone();
            double inertia = labelsInertia.getInertia();
            distances = labelsInertia.getDistances().clone();
            centers = KMeans.centersDense(data, labels, nClusters);
            if (inertia < bestInertia) {
                this.bestCenters = centers.clone();
                this.bestInertia = inertia;
            }
            if (Matrices.squared_norm(Matrices.substractMatrices(centersOld, centers)) <= tol) {
                break;
            }
        }
    }

    double[][] getBestCenters() {
        return this.bestCenters;
    }

    double getBestInertia() {
        return this.bestInertia;
    }
}
