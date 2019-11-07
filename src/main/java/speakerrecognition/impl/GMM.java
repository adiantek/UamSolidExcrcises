package speakerrecognition.impl;


// https://commons.apache.org/proper/commons-math/apidocs/org/apache/commons/math3/distribution/fitting/MultivariateNormalMixtureExpectationMaximization.html
// https://www.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf

import speakerrecognition.math.Matrices;
import speakerrecognition.math.Statistics;

public class GMM {
    private static final double EPS = 2.2204460492503131e-16;
    private int numOfComponents;
    private double[][] observations;
    private double min_covar = 0.001;
    private double current_log_likelihood = 0;

    private double[][] means = null;
    private double[] weights = null;
    private double[][] covars = null;

    private double[][] best_means = null;
    private double[] best_weights = null;
    private double[][] best_covars = null;

    //private MultivariateNormalMixtureExpectationMaximization gmm = null;


    GMM(double[][] data, int compNum) {
        this.observations = data;
        this.numOfComponents = compNum;
        this.means = new double[compNum][data[0].length];
        this.weights = new double[data.length];
        this.covars = new double[compNum][data[0].length];
        //this.gmm = new MultivariateNormalMixtureExpectationMaximization(data);
    }

    public void fit() {
        double change = 0;

        try {

            double[][] cv;
            double max_log_prob = Double.NEGATIVE_INFINITY;

            int n_init = 10;
            for (int i = 0; i < n_init; i++) {
                KMeans kMeans = new KMeans(this.observations, this.numOfComponents);
                kMeans.fit();
                this.means = kMeans.get_centers();
                this.weights = Matrices.fillWith(this.weights, (double) 1 / this.numOfComponents);

                this.covars = Matrices.cov(Matrices.transpose(this.observations)); //np.cov(X.T), gmm.py line 450
                cv = Matrices.eye(this.observations[0].length, this.min_covar); //self.min_covar * np.eye(X.shape[1])
                this.covars = Matrices.addMatrixes(this.covars, cv);
                this.covars = Matrices.duplicate(Matrices.chooseDiagonalValues(this.covars), this.numOfComponents);

                int n_iter = 10;
                for (int j = 0; j < n_iter; j++) {
                    double prev_log_likelihood = current_log_likelihood;
                    Score_samples score_samples = new Score_samples(this.observations, this.means, this.covars, this.weights);
                    double[] log_likelihoods = score_samples.getLogprob();
                    double[][] responsibilities = score_samples.getResponsibilities();
                    current_log_likelihood = Statistics.getMean(log_likelihoods);

                    if (!Double.isNaN(prev_log_likelihood)) {
                        change = Math.abs(current_log_likelihood - prev_log_likelihood);
                        double tol = 0.001;
                        if (change < tol) {
                            boolean converged = true;
                            break;
                        }

                    }

                    /// do m-step - gmm.py line 509
                    do_mstep(this.observations, responsibilities);

                }

                if (current_log_likelihood > max_log_prob) {
                    max_log_prob = current_log_likelihood;
                    this.best_means = this.means;
                    this.best_covars = this.covars;
                    this.best_weights = this.weights;

                }
            }

            if (Double.isInfinite(max_log_prob))
                System.out.println("EM algorithm was never able to compute a valid likelihood given initial parameters");
        } catch (Exception myEx) {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }

    }

    public double[][] get_means() {
        return this.best_means;
    }

    public double[][] get_covars() {
        return this.best_covars;
    }

    public double[] get_weights() {
        return this.best_weights;
    }

    private void do_mstep(double[][] data, double[][] responsibilities) {
        try {
            double[] weights = Matrices.sum(responsibilities, 0);
            double[][] weighted_X_sum = Matrices.multiplyByMatrix(Matrices.transpose(responsibilities), data);
            double[] inverse_weights = Matrices.invertElements(Matrices.addValue(weights, 10 * EPS));
            this.weights = Matrices.addValue(Matrices.multiplyByValue(weights, 1.0 / (Matrices.sum(weights) + 10 * EPS)), EPS);
            this.means = Matrices.multiplyByValue(weighted_X_sum, inverse_weights);
            this.covars = covar_mstep_diag(this.means, data, responsibilities, weighted_X_sum, inverse_weights, this.min_covar);
        } catch (Exception myEx) {
            myEx.printStackTrace();
            System.exit(1);
        }

    }

    private double[][] covar_mstep_diag(double[][] means, double[][] X, double[][] responsibilities, double[][] weighted_X_sum, double[] norm, double min_covar) {
        double[][] temp = null;
        try {
            double[][] avg_X2 = Matrices.multiplyByValue(Matrices.multiplyByMatrix(Matrices.transpose(responsibilities), Matrices.multiplyMatrixesElByEl(X, X)), norm);
            double[][] avg_means2 = Matrices.power(means, 2);
            double[][] avg_X_means = Matrices.multiplyByValue(Matrices.multiplyMatrixesElByEl(means, weighted_X_sum), norm);
            temp = Matrices.addValue(Matrices.addMatrixes(Matrices.substractMatrixes(avg_X2, Matrices.multiplyByValue(avg_X_means, 2)), avg_means2), min_covar);
        } catch (Exception myEx) {
            System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
        return temp;
    }

    private class Score_samples {
        private double[][] means = null;
        private double[][] covars = null;
        /////out matrixes////
        private double[] logprob = null;
        private double[][] responsibilities = null;
        /////////////////////


        Score_samples(double[][] X, double[][] means, double[][] covars, double[] weights) {
            this.responsibilities = new double[X.length][GMM.this.numOfComponents];
            this.means = means;
            this.covars = covars;


            try {
                double[][] lpr = log_multivariate_normal_density(X, this.means, this.covars);
                lpr = Matrices.addValue(lpr, Matrices.makeLog(weights));
                this.logprob = Matrices.logsumexp(lpr);
                // gmm.py line 321
                this.responsibilities = Matrices.exp(Matrices.substractValue(lpr, logprob));
            } catch (Exception myEx) {
                //System.out.println("An exception encourred: " + myEx.getMessage());
                myEx.printStackTrace();
                System.exit(1);
            }

        }

        public double[] getLogprob() {
            return this.logprob;
        }

        public double[][] getResponsibilities() {
            return this.responsibilities;
        }

        private double[][] log_multivariate_normal_density(double[][] data, double[][] means, double[][] covars) {
            //diagonal type
            double[][] lpr = new double[data.length][means.length];
            int n_samples = data.length;
            int n_dim = data[0].length;

            try {
                double[] sumLogCov = Matrices.sum(Matrices.makeLog(covars), 1); //np.sum(np.log(covars), 1)
                double[] sumDivMeanCov = Matrices.sum(Matrices.divideElements(Matrices.power(this.means, 2), this.covars), 1); //np.sum((means ** 2) / covars, 1)
                double[][] dotXdivMeanCovT = Matrices.multiplyByValue(Matrices.multiplyByMatrix(data, Matrices.transpose(Matrices.divideElements(means, covars))), -2); //- 2 * np.dot(X, (means / covars).T)
                double[][] dotXdivOneCovT = Matrices.multiplyByMatrix(Matrices.power(data, 2), Matrices.transpose(Matrices.invertElements(covars)));


                sumLogCov = Matrices.addValue(sumLogCov, n_dim * Math.log(2 * Math.PI)); //n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                sumDivMeanCov = Matrices.addMatrixes(sumDivMeanCov, sumLogCov); // n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1)
                dotXdivOneCovT = Matrices.sum(dotXdivOneCovT, dotXdivMeanCovT); //- 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T)
                dotXdivOneCovT = Matrices.addValue(dotXdivOneCovT, sumDivMeanCov); // (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1) - 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))
                lpr = Matrices.multiplyByValue(dotXdivOneCovT, -0.5);
            } catch (Exception myEx) {
                System.out.println("An exception encourred: " + myEx.getMessage());
                myEx.printStackTrace();
                System.exit(1);
            }

            return lpr;
        }
    }
}
