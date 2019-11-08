package speakerrecognition.math;

public class MFCCProcessingUtils {
    public static final int MFCC_NUM = 13;
    private static final double PRE_EMPH = 0.95;

    public static double[][] dctmatrix(int n) {
        double[][] x = Matrices.meshgrid_ox(n);
        double[][] y = Matrices.meshgrid_oy(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = (x[i][j] * 2 + 1) * Math.PI / (2 * n);
            }
        }

        double[][] d1 = Matrices.multiplyMatricesElByEl(x, y);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d1[i][j] = Math.sqrt(2 / (double) n) * Math.cos(d1[i][j]);
            }
        }
        for (int i = 0; i < n; i++) {
            d1[0][i] /= Math.sqrt(2);
        }

        double[][] d = new double[MFCC_NUM][n];
        for (int i = 1; i < MFCC_NUM + 1; i++) {
            System.arraycopy(d1[i], 0, d[i - 1], 0, n);
        }

        return d;
    }

    public static double[] arrange(int x1, int x2) {
        double[] temp = null;
        try {
            temp = new double[x2 - x1];
            for (int i = 0; i < temp.length; i++) {
                temp[i] = x1 + i;
            }

        } catch (IndexOutOfBoundsException e) {
            System.err.println("IndexOutOfBoundsException: " + e.getMessage());
        }
        return temp;
    }

    public static double energy(double[] x) {
        double en = 0;
        for (double v : x) en = en + Math.pow(v, 2);
        return en;
    }

    public static double[] preemphasis(double[] x) {
        double[] y = new double[x.length];
        y[0] = x[0];
        for (int i = 1; i < x.length; i++) {
            y[i] = x[i] - PRE_EMPH * x[i - 1];
        }
        return y;
    }
}
