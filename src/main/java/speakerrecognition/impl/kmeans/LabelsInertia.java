package speakerrecognition.impl.kmeans;

class LabelsInertia {
    private int[] labels;
    private double[] distances;
    private double inertia;

    int[] getLabels() {
        return this.labels.clone();
    }

    double getInertia() {
        return this.inertia;
    }

    double[] getDistances() {
        return this.distances.clone();
    }

    LabelsInertia(double[][] X, double[] xSquaredNorms, double[][] centers, double[] distances) {
        this.distances = distances;
        LabelsPrecomputeDence result = new LabelsPrecomputeDence(X, xSquaredNorms, centers, this.distances);
        this.labels = result.getLabels().clone();
        this.inertia = result.getInertia();
        this.distances = result.getDistances().clone();
    }
}
