package speakerrecognition.impl.gmm;

import speakerrecognition.impl.kmeans.KMeans;
import speakerrecognition.math.Matrices;
import speakerrecognition.math.Statistics;

public class GMM {
    private static final double EPS = 2.2204460492503131e-16;
    private int numOfComponents;
    private double[][] observations;
    private double min_covar = 0.001;
    private double currentLogLikelihood = 0;

    private double[][] means;
    private double[] weights;
    private double[][] covars;

    private double[][] best_means = null;
    private double[] best_weights = null;
    private double[][] best_covars = null;

    public GMM(double[][] data, int compNum) {
        this.observations = data;
        this.numOfComponents = compNum;
        this.means = new double[compNum][data[0].length];
        this.weights = new double[data.length];
        this.covars = new double[compNum][data[0].length];
    }

    public void fit() {
        double change;

        double[][] cv;
        double maxLogProb = Double.NEGATIVE_INFINITY;

        int nInit = 10;
        for (int i = 0; i < nInit; i++) {
            KMeans kMeans = new KMeans(this.observations, this.numOfComponents);
            kMeans.fit();
            this.means = kMeans.getCenters();
            this.weights = Matrices.fillWith(this.weights, (double) 1 / this.numOfComponents);

            this.covars = Matrices.cov(Matrices.transpose(this.observations)); //np.cov(X.T), gmm.py line 450
            cv = Matrices.eye(this.observations[0].length, this.min_covar); //self.min_covar * np.eye(X.shape[1])
            this.covars = Matrices.addMatrices(this.covars, cv);
            this.covars = Matrices.duplicate(Matrices.chooseDiagonalValues(this.covars), this.numOfComponents);

            int nIter = 10;
            for (int j = 0; j < nIter; j++) {
                double prev_log_likelihood = currentLogLikelihood;
                ScoreSamples score_samples = new ScoreSamples(this.observations, this.means, this.covars, this.weights, this.numOfComponents);
                double[] log_likelihoods = score_samples.getLogprob();
                double[][] responsibilities = score_samples.getResponsibilities();
                currentLogLikelihood = Statistics.getMean(log_likelihoods);
                if (!Double.isNaN(prev_log_likelihood)) {
                    change = Math.abs(currentLogLikelihood - prev_log_likelihood);
                    double tol = 0.001;
                    if (change < tol) {
                        break;
                    }
                }
                doMstep(this.observations, responsibilities);
            }

            if (currentLogLikelihood > maxLogProb) {
                maxLogProb = currentLogLikelihood;
                this.best_means = this.means;
                this.best_covars = this.covars;
                this.best_weights = this.weights;

            }
        }

        if (Double.isInfinite(maxLogProb))
            throw new IllegalArgumentException("EM algorithm was never able to compute a valid likelihood given initial parameters");
    }

    public double[][] getMeans() {
        return this.best_means;
    }

    public double[][] getCovars() {
        return this.best_covars;
    }

    public double[] getWeights() {
        return this.best_weights;
    }

    private void doMstep(double[][] data, double[][] responsibilities) {
        double[] weights = Matrices.sum(responsibilities, 0);
        double[][] weighted_X_sum = Matrices.multiplyByMatrix(Matrices.transpose(responsibilities), data);
        double[] inverse_weights = Matrices.invertElements(Matrices.addValue(weights, 10 * EPS));
        this.weights = Matrices.addValue(Matrices.multiplyByValue(weights, 1.0 / (Matrices.sum(weights) + 10 * EPS)), EPS);
        this.means = Matrices.multiplyByValue(weighted_X_sum, inverse_weights);
        this.covars = covarMstepDiag(this.means, data, responsibilities, weighted_X_sum, inverse_weights, this.min_covar);
    }

    private static double[][] covarMstepDiag(double[][] means, double[][] X, double[][] responsibilities, double[][] weighted_X_sum, double[] norm, double min_covar) {
        double[][] avg_X2 = Matrices.multiplyByValue(Matrices.multiplyByMatrix(Matrices.transpose(responsibilities), Matrices.multiplyMatricesElByEl(X, X)), norm);
        double[][] avg_means2 = Matrices.power(means, 2);
        double[][] avg_X_means = Matrices.multiplyByValue(Matrices.multiplyMatricesElByEl(means, weighted_X_sum), norm);
        return Matrices.addValue(Matrices.addMatrices(Matrices.substractMatrices(avg_X2, Matrices.row_mul(avg_X_means, 2)), avg_means2), min_covar);
    }
}
