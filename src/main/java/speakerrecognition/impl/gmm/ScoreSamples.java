package speakerrecognition.impl.gmm;

import speakerrecognition.math.Matrices;

class ScoreSamples {
    private double[][] means;
    private double[][] covars;
    private double[] logprob;
    private double[][] responsibilities;


    ScoreSamples(double[][] X, double[][] means, double[][] covars, double[] weights, int numOfComponents) {
        this.responsibilities = new double[X.length][numOfComponents];
        this.means = means;
        this.covars = covars;

        double[][] lpr = logMultivariateNormalDensity(X, this.means, this.covars);
        lpr = Matrices.addValue(lpr, Matrices.makeLog(weights));
        this.logprob = Matrices.logsumexp(lpr);
        this.responsibilities = Matrices.exp(Matrices.substractValue(lpr, logprob));
    }

    double[] getLogprob() {
        return this.logprob;
    }

    double[][] getResponsibilities() {
        return this.responsibilities;
    }

    private double[][] logMultivariateNormalDensity(double[][] data, double[][] means, double[][] covars) {
        //diagonal type
        double[][] lpr = new double[data.length][means.length];
        int n_dim = data[0].length;

        double[] sumLogCov = Matrices.sum(Matrices.makeLog(covars), 1); //np.sum(np.log(covars), 1)
        double[] sumDivMeanCov = Matrices.sum(Matrices.divideElements(Matrices.power(this.means, 2), this.covars), 1); //np.sum((means ** 2) / covars, 1)
        double[][] dotXdivMeanCovT = Matrices.row_mul(Matrices.multiplyByMatrix(data, Matrices.transpose(Matrices.divideElements(means, covars))), -2); //- 2 * np.dot(X, (means / covars).T)
        double[][] dotXdivOneCovT = Matrices.multiplyByMatrix(Matrices.power(data, 2), Matrices.transpose(Matrices.invertElements(covars)));


        sumLogCov = Matrices.addValue(sumLogCov, n_dim * Math.log(2 * Math.PI)); //n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
        sumDivMeanCov = Matrices.addMatrices(sumDivMeanCov, sumLogCov); // n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1)
        dotXdivOneCovT = Matrices.addMatrices(dotXdivOneCovT, dotXdivMeanCovT); //- 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T)
        dotXdivOneCovT = Matrices.addValue(dotXdivOneCovT, sumDivMeanCov); // (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1) - 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))
        lpr = Matrices.row_mul(dotXdivOneCovT, -0.5);

        return lpr;
    }
}
