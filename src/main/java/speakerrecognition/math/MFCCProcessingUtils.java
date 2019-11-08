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

    private static double[] arrange(int x1, int x2) {
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

    private static double energy(double[] x) {
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

    public static double[][] melfb(int p, int n, int fs) {
        // p - number of filterbanks
        // n - length of fft
        // fs - sample rate

        double f0 = 700 / (double) fs;
        int fn2 = (int) Math.floor((double) n / 2);
        double lr = Math.log((double) 1 + 0.5 / f0) / (p + 1);
        double[] CF = arrange(1, p + 1);

        for (int i = 0; i < CF.length; i++) {
            CF[i] = fs * f0 * (Math.exp(CF[i] * lr) - 1);
            //CF[i] = (Math.exp(CF[i]*lr));
        }

        double[] bl = {0, 1, p, p + 1};

        for (int i = 0; i < bl.length; i++) {
            bl[i] = n * f0 * (Math.exp(bl[i] * lr) - 1);
        }

        int b1 = (int) Math.floor(bl[0]) + 1;
        int b2 = (int) Math.ceil(bl[1]);
        int b3 = (int) Math.floor(bl[2]);
        int b4 = Math.min(fn2, (int) Math.ceil(bl[3])) - 1;
        double[] pf = arrange(b1, b4 + 1);

        for (int i = 0; i < pf.length; i++) {
            pf[i] = Math.log(1 + pf[i] / f0 / (double) n) / lr;
        }

        double[] fp = new double[pf.length];
        double[] pm = new double[pf.length];

        for (int i = 0; i < fp.length; i++) {
            fp[i] = Math.floor(pf[i]);
            pm[i] = pf[i] - fp[i];
        }

        double[][] m = new double[p][1 + fn2];
        int r = 0;

        for (int i = b2 - 1; i < b4; i++) {
            r = (int) fp[i] - 1;
            m[r][i + 1] += 2 * (1 - pm[i]);
        }

        for (int i = 0; i < b3; i++) {
            r = (int) fp[i];
            m[r][i + 1] += 2 * pm[i];
        }
        double[] temp_row;
        double row_energy = 0;
        for (int i = 0; i < m.length; i++) {
            temp_row = m[i];
            row_energy = energy(temp_row);
            if (row_energy < 0.0001)
                temp_row[i] = i;
            else {
                while (row_energy > 1.01) {
                    temp_row = Matrices.row_mul(temp_row, 0.99);
                    row_energy = energy(temp_row);
                }
                while (row_energy < 0.99) {
                    temp_row = Matrices.row_mul(temp_row, 1.01);
                    row_energy = energy(temp_row);
                }
            }
            m[i] = temp_row;
        }
        return m;
    }
}
