package speakerrecognition.impl;

import speakerrecognition.math.Matrices;
import speakerrecognition.math.Statistics;

import java.io.Serializable;

public class Speaker_model implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    private double[][] means = null;
    private double[][] covars = null;
    private double[] weights = null;
    private String name = null;

    public Speaker_model(double[][] means, double[][] covars, double[] weights, String name) {
        this.means = means;
        this.covars = covars;
        this.weights = weights;
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public double getScore(double[][] data) throws MyException {
        double score = 0;
        double[] logprob = null;
        double[][] lpr = log_multivariate_normal_density(data, this.means, this.covars);
        lpr = Matrices.addValue(lpr, Matrices.makeLog(this.weights));
        logprob = Matrices.logsumexp(lpr);
        score = Statistics.getMean(logprob);
        return score;
    }

    private double[][] log_multivariate_normal_density(double[][] data, double[][] means, double[][] covars) throws MyException {
        //diagonal type
        double[][] lpr = new double[data.length][means.length];
        int n_samples = data.length;
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
