package speakerrecognition.math;


import java.util.Arrays;

public final class Matrices {

    public static double[] row_mul(double[] x, double y) {
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            v[i] = x[i] * y;
        }
        return v;
    }

    public static double[] row_mul(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(String.format("Cannot multiply vectors el by el. Vectors must have same length, while it is [%d] and [%d]", x.length, y.length));
        }

        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            v[i] = x[i] * y[i];
        }
        return v;
    }

    public static double[][] row_mul(double[][] x, double y) {
        double[][] v = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                v[i][j] = x[i][j] * y;
            }
        }
        return v;
    }


    public static double[][] multiplyByMatrix(double[][] m1, double[][] m2) {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if (m1ColLength != m2RowLength) {
            throw new IllegalArgumentException("While multiplying Matrices, number of columns of first array [" + Integer.toString(m1ColLength) + "] must be the same as number of rows in second array [" + Integer.toString(m2RowLength) + "]. Obviously, it is not.");//return null; // matrix multiplication is not possible
        }
        int mRRowLength = m1.length;    // m result rows length
        int mRColLength = m2[0].length; // m result columns length
        double[][] mResult = new double[mRRowLength][mRColLength];
        for (int i = 0; i < mRRowLength; i++) {         // rows from m1
            for (int j = 0; j < mRColLength; j++) {     // columns from m2
                for (int k = 0; k < m1ColLength; k++) { // columns from m1
                    mResult[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return mResult;
    }

    public static double[] multiplyByMatrix(double[][] m1, double[] m2) {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if (m1ColLength != m2RowLength) //return null; // matrix multiplication is not possible
            throw new IllegalArgumentException("While multiplying matrix by vector, number of columns of first array [" + Integer.toString(m1ColLength) + "] must be the same as number of rows (elements) in second vector [" + Integer.toString(m2RowLength) + "]. Obviously, it is not.");
        int mRRowLength = m1.length;    // m result rows length
        int mRColLength = m2RowLength; // m result columns length
        double[] mResult = new double[mRRowLength];
        for (int i = 0; i < mRRowLength; i++) {         // rows from m1
            for (int j = 0; j < mRColLength; j++) {     // columns from m2
                mResult[i] += m1[i][j] * m2[j];
            }
        }
        return mResult;
    }

    public static double[][] multiplyMatricesElByEl(double[][] m1, double[][] m2) {

        if (m1.length != m2.length || m1[0].length != m2[0].length) {
            throw new IllegalArgumentException("While multiplying matrixex element by element, they must have equal dimmensions, while it is [" + Integer.toString(m1.length) + "][" + Integer.toString(m1[0].length) + "] and [" + Integer.toString(m2.length) + "][" + Integer.toString(m2[0].length) + "].");
        }

        double[][] result = new double[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] * m2[i][j];
            }
        }
        return result;

    }

    public static double[][] multiplyByValue(double[][] x, double[] y) {
        double[][] temp = new double[x.length][x[0].length];

        if (x.length != y.length && x[0].length != y.length)
            throw new IllegalArgumentException("Cannot multiply matrix by vecror element by element, neither row-wise nor column-wise. Number of elements in vector [" + Integer.toString(y.length) + "] must be equal to any of dimmension parameters of first array [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "].");

        if (x.length == y.length) {
            for (int i = 0; i < x[0].length; i++) {
                for (int j = 0; j < x.length; j++)
                    temp[j][i] = x[j][i] * y[j];
            }
        } else {
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    temp[i][j] = x[i][j] * y[j];
            }
        }

        return temp;
    }

    public static double[] multiplyByValue(double[] x, double y) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] * y;
        }
        return temp;
    }

    public static double squared_norm(double[][] x) {
        double v = 0;
        for (double[] doubles : x) {
            for (double aDouble : doubles) {
                v += aDouble * aDouble;
            }
        }

        return v;
    }

    public static double[][] meshgrid_ox(int n) {
        double[][] x = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[j][i] = i;
            }
        }
        return x;
    }

    public static double[][] meshgrid_oy(int n) {
        double[][] x = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = i;
            }
        }

        return x;
    }

    public static double[][] transpose(double[][] x) {
        double[][] result = new double[x[0].length][x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[j][i] = x[i][j];
            }
        }

        return result;
    }

    public static double[] fillWith(double[] x, double y) {
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++)
            v[i] = y;

        return v;
    }

    public static double[][] substractValue(double[][] x, double[] y) {

        if (x.length != y.length && x[0].length != y.length)
            throw new IllegalArgumentException("Cannot substract vecror from array element by element, neither row-wise nor column-wise. Number of elements in vector [" + Integer.toString(y.length) + "] must be equal to any of dimmension parameters of first array [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "].");
        double[][] temp = new double[x.length][x[0].length];
        // [n][m] + [n][1], m times
        // [n][m] + [1][m] n times
        if (x.length == y.length) {
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    temp[i][j] = x[i][j] - y[i];
            }
        } else {
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    temp[i][j] = x[i][j] - y[j];
            }
        }

        return temp;
    }

    public static int[] addValue(int[] x, int y) {
        int[] temp = new int[x.length];
        for (int i = 0; i < x.length; i++)
            temp[i] = x[i] + y;
        return temp;
    }

    public static double[] addValue(double[] x, double y) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++)
            temp[i] = x[i] + y;
        return temp;
    }

    public static double[][] addValue(double[][] x, double y) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] + y;
        }

        return temp;
    }

    public static double[][] addValue(double[][] x, double y[]) {
        double[][] temp = new double[x.length][x[0].length];
        if (x.length != y.length && x[0].length != y.length) {
            throw new IllegalArgumentException("Cannot add vector to array element by element, neither row-wise nor column-wise. Number of elements in vector [" + Integer.toString(y.length) + "] must be equal to any of dimmension parameters of first array [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "].");
        }
        if (x.length == y.length) {
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    temp[i][j] = x[i][j] + y[i];
            }
        } else {
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    temp[i][j] = x[i][j] + y[j];
            }
        }


        return temp;
    }

    public static double[] addMatrices(double[] x, double[] y) {
        double[] temp = new double[x.length];

        if (x.length != y.length)
            throw new IllegalArgumentException("Cannot add vectors el by el. Vectors must have same length, while it is [" + Integer.toString(x.length) + "] and [" + Integer.toString(y.length) + "].");

        for (int i = 0; i < x.length; i++)
            temp[i] = x[i] + y[i];
        return temp;
    }

    public static double[][] addMatrices(double[][] x, double[][] y) {

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrices must have equal dimensions");
            //return null;
            throw new IllegalArgumentException("While adding Matrices element by element, they must have equal dimmensions, while it is [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "] and [" + Integer.toString(y.length) + "][" + Integer.toString(y[0].length) + "].");
        }

        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] + y[i][j];
        }
        return temp;
    }

    public static double[][] substractMatrices(double[][] x, double[][] y) {

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrices must have equal dimensions");
            //return null;
            throw new IllegalArgumentException("While substracting Matrices element by element, they must have equal dimmensions, while it is [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "] and [" + Integer.toString(y.length) + "][" + Integer.toString(y[0].length) + "].");
        }

        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] - y[i][j];
        }
        return temp;
    }

    public static double sum(double[] x) {
        return Arrays.stream(x).sum();
    }

    public static double[] sum(double[][] x, int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException(String.format("Wrong axis, should be 0 or 1, and is %d", axis));
        }

        double[] result = null;
        if (axis == 1) {
            result = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    result[i] += x[i][j];
                }
            }
        } else {
            result = new double[x[0].length];
            for (int i = 0; i < x[0].length; i++) {
                for (double[] doubles : x) {
                    result[i] += doubles[i];
                }
            }
        }

        return result;
    }

    public static double[] genRandMatrix(double max, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Size cannot be less orequal to 0.");
        }

        double[] x = new double[size];
        for (int i = 0; i < size; i++) {
            x[i] = Math.random() * max;
        }

        return x;
    }

    public static double[] cumsum(double[] x) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < i + 1; j++) {
                temp[i] += x[j];
            }
        }
        return temp;
    }

    public static int[] searchsorted(double[] x, double[] y) {
        int[] result = new int[y.length];
        //int idx=0;
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < x.length; j++) {
                if (x[j] > y[i]) {
                    result[i] = j;
                    break;
                }
            }
        }
        return result;

    }

    public static double[] minimum(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(String.format("Cannot search minimum value from two vectors of different length - [%d] and [%d]", x.length, y.length));
        }

        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            temp[i] = Math.min(y[i], x[i]);
        }

        return temp;
    }

    public static double[] select_row(double[][] x, int y) {

        if (y > x.length - 1)
            throw new IllegalArgumentException(String.format("Selected row out of range - %d. of %d rows (remember aobout 0th row!).", y, x.length));

        double result[] = new double[x[0].length];
        for (int i = 0; i < x[0].length; i++)
            result[i] = x[y][i];
        return result;
    }

    public static double einsum(double[] x) {
        double temp = 0;
        for (int i = 0; i < x.length; i++) {
            temp = temp + Math.pow(x[i], 2);
        }

        return temp;
    }


    public static double[] einsum(double[][] x) {
        double[] temp = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                temp[j] = temp[j] + Math.pow(x[j][i], 2);
            }
        }

        return temp;
    }

    public static double[] euclidean_distances(double[] x, double[][] y, double[] z) {
        //double[] result = null;

        double[] distances = new double[y.length];//[this.numOfRows];

        double XX = einsum(x);
        distances = Matrices.multiplyByMatrix(y, x);
        distances = Matrices.row_mul(distances, -2);
        distances = Matrices.addValue(distances, XX);
        distances = Matrices.addMatrices(distances, z);
        return distances;
    }

    public static double[][] euclidean_distances(double[][] x, double[][] y, double[] z) {
        //double [][] result = new double[x.length][y.length];
        double[][] distances = null;
        double[] XX = null;
        XX = einsum(x);
        distances = Matrices.multiplyByMatrix(x, Matrices.transpose(y));
        distances = Matrices.row_mul(distances, -2);
        distances = Matrices.addValue(distances, XX);
        distances = Matrices.addValue(distances, z);
        return distances;
    }

    public static double[][] cov(double[][] x) {

        double[][] temp = null;
        double[] X_mean = null;

        temp = Matrices.copy2dArray(x);
        //////////substracting mean //////////////
        X_mean = Statistics.getMean(Matrices.transpose(x));
        for (int j = 0; j < x[0].length; j++) {
            for (int i = 0; i < x.length; i++) {
                temp[i][j] -= X_mean[i];
            }
        }

        temp = Matrices.divideByValue(Matrices.multiplyByMatrix(x, Matrices.transpose(temp)), (double) x[0].length - 1);

        return temp;
    }

    public static double[][] divideByValue(double[][] x, double y) {
        if (y == 0)
            throw new IllegalArgumentException("Cannot divide by 0");
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                temp[i][j] = x[i][j] / y;
            }
        }
        return temp;
    }

    public static double[] chooseDiagonalValues(double[][] x) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++)
            temp[i] = x[i][i];
        return temp;
    }

    public static double[] makeLog(double[] x) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {

            if (x[i] <= 0)
                throw new IllegalArgumentException("Cannot make Log of value below 0 - Log(" + Double.toString(x[i]) + "), (index " + Integer.toString(i) + ").");
            temp[i] = Math.log(x[i]);

        }
        return temp;
    }

    public static double[][] makeLog(double[][] x) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {

                if (x[i][j] <= 0)
                    throw new IllegalArgumentException("Cannot make Log of value below 0 - Log(" + Double.toString(x[i][j]) + "), (index [" + Integer.toString(i) + "," + Integer.toString(j) + "]).");

                temp[i][j] = Math.log(x[i][j]);
            }
        }
        return temp;
    }

    public static double[] invertElements(double[] x) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (x[i] == 0)
                throw new IllegalArgumentException("While inverting values, cannot divide by 0, (index " + Integer.toString(i) + ").");
            temp[i] = 1 / (x[i]);
        }
        return temp;
    }

    public static double[][] invertElements(double[][] x) {
        // 1.0 / a[m][n]
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {

                if (x[i][j] <= 0)
                    throw new IllegalArgumentException("While inverting values, cannot divide by 0 (index [" + Integer.toString(i) + "," + Integer.toString(j) + "]).");
                temp[i][j] = 1 / (x[i][j]);
            }
        }
        return temp;
    }

    public static double[][] divideElements(double[][] x, double[][] y) {
        //a[0][0]/b[0][0] ,  a[m][n]/b[m][n] ...

        if (x.length != y.length || x[0].length != y[0].length) {
            throw new IllegalArgumentException("While dividing element by element, they must have equal dimmensions, now it is [" + Integer.toString(x.length) + "][" + Integer.toString(x[0].length) + "] and [" + Integer.toString(y.length) + "][" + Integer.toString(y[0].length) + "].");
        }

        double[][] result = new double[x.length][x[0].length];

        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                if (y[i][j] <= 0)
                    throw new IllegalArgumentException("While inverting values, cannot divide by 0 (y[" + Integer.toString(i) + "][" + Integer.toString(j) + "]).");
                result[i][j] = x[i][j] / y[i][j];
            }
        }

        return result;
    }

    public static double[][] power(double[][] x, double y) {
        double[][] v = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                v[i][j] = Math.pow(x[i][j], y);
        }
        return v;
    }

    public static double[] logsumexp(double[][] data) {
        double[][] temp = Matrices.transpose(data);
        double[] vmax = Matrices.max(temp, 0);
        double[] out = Matrices.makeLog(Matrices.sum(Matrices.exp(Matrices.substractValue(temp, vmax)), 0));
        out = Matrices.addMatrices(out, vmax);
        return out;
    }

    public static double[][] exp(double[][] x) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = Math.exp(x[i][j]);
        }
        return temp;
    }

    public static double[] max(double[][] x, int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException(String.format("Wrong axis, sholud be 0 or 1, and is %d", axis));
        }

        double[] vmax;
        if (axis == 0) {
            vmax = new double[x[0].length];

            for (int i = 0; i < x[0].length; i++) {
                vmax[i] = Double.NEGATIVE_INFINITY;
            }  //JAK CO TO USUN¥Æ!!!

            for (int i = 0; i < x[0].length; i++) {
                for (int j = 0; j < x.length; j++)
                    if (vmax[i] < x[j][i])
                        vmax[i] = x[j][i];
            }
        } else {
            vmax = new double[x.length];

            for (int i = 0; i < x.length; i++) {
                vmax[i] = Double.NEGATIVE_INFINITY;
            }// JAK CO TO USUN¥Æ!!!


            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    if (vmax[i] < x[i][j])
                        vmax[i] = x[i][j];
            }
        }


        return vmax;
    }

    public static double[][] eye(int n, double coef) {
        double[][] temp = new double[n][n];
        for (int i = 0; i < n; i++)
            temp[i][i] = coef;
        return temp;
    }

    public static double[][] duplicate(double[] data, int n) {
        if (n < 1) {
            throw new IllegalArgumentException("Can not duplicate less than once");
        }

        double[][] v = new double[n][data.length];
        for (int i = 0; i < n; i++)
            v[i] = data;

        return v;
    }

    public static double[][] copy2dArray(double[][] x) {
        double[][] v = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double[] aMatrix = x[i];
            int len = aMatrix.length;
            v[i] = new double[len];
            System.arraycopy(aMatrix, 0, v[i], 0, len);
        }

        return v;
    }
}
