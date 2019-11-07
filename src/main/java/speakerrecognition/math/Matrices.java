package speakerrecognition.math;


import speakerrecognition.impl.MyException;

public final class Matrices {

    public static double[] row_mul(double[] x, double y) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            temp[i] = x[i] * y;
        }
        return temp;
    }

    public static double[] row_mul(double[] x, double[] y) throws MyException {

        if (x.length != y.length)
            throw new MyException("Cannot multiply vectors el by el. Vectors must have same length, while it is [" + x.length + "] and [" + y.length + "].");
        else {
            double[] temp = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                temp[i] = x[i] * y[i];
            }
            return temp;
        }
    }

    public static double[][] row_mul(double[][] x, double y) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] * y;
        }
        return temp;
    }


    public static double[][] multiplyByMatrix(double[][] m1, double[][] m2) throws MyException {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if (m1ColLength != m2RowLength) {
            throw new MyException("While multiplying matrixes, number of columns of first array [" + m1ColLength + "] must be the same as number of rows in second array [" + m2RowLength + "]. Obviously, it is not.");//return null; // matrix multiplication is not possible
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

    public static double[] multiplyByMatrix(double[][] m1, double[] m2) throws MyException {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if (m1ColLength != m2RowLength) //return null; // matrix multiplication is not possible
            throw new MyException("While multiplying matrix by vector, number of columns of first array [" + m1ColLength + "] must be the same as number of rows (elements) in second vector [" + m2RowLength + "]. Obviously, it is not.");
        int mRRowLength = m1.length;    // m result rows length
        double[] mResult = new double[mRRowLength];
        for (int i = 0; i < mRRowLength; i++) {         // rows from m1
            for (int j = 0; j < m2RowLength; j++) {     // columns from m2
                mResult[i] += m1[i][j] * m2[j];
            }
        }
        return mResult;
    }

    public static double[][] multiplyMatrixesElByEl(double[][] m1, double[][] m2) throws MyException {

        if (m1.length != m2.length || m1[0].length != m2[0].length) {
            //System.out.println("Matrixes must have equal dimensions");
            //return null;
            throw new MyException("While multiplying matrixex element by element, they must have equal dimmensions, while it is [" + m1.length + "][" + m1[0].length + "] and [" + m2.length + "][" + m2[0].length + "].");
        }
		
		/*double[][] result = new double[m1.length][m1.length];
			for(int i=0;i<m1.length;i++){
				for(int j=0;j<m1.length;j++){
					result[i][j] = m1[i][j]*m2[i][j];
				}
			}*/

        double[][] result = new double[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++) {
                result[i][j] = m1[i][j] * m2[i][j];
            }
        }
        return result;

    }

    public static double[][] multiplyByValue(double[][] x, double y) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] * y;
        }
        return temp;
    }

    public static double[][] multiplyByValue(double[][] x, double[] y) throws MyException {
        double[][] temp = new double[x.length][x[0].length];

        if (x.length != y.length && x[0].length != y.length)
            throw new MyException("Cannot multiply matrix by vecror element by element, neither row-wise nor column-wise. Number of elements in vector [" + y.length + "] must be equal to any of dimmension parameters of first array [" + x.length + "][" + x[0].length + "].");

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
        double result = 0;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                result += Math.pow(x[i][j], 2);
        }

        return result;
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
        int i = x.length;
        int j = x[0].length;
        double[][] result = new double[j][i];
        for (int ii = 0; ii < j; ii++) {
            for (int jj = 0; jj < i; jj++) {
                result[ii][jj] = x[jj][ii];
            }
        }
        return result;

    }

    public static double[] fillWith(double[] x, double y) {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++)
            temp[i] = y;
        return temp;
    }

    public static double[][] substractValue(double[][] x, double[] y) throws MyException {

        if (x.length != y.length && x[0].length != y.length)
            throw new MyException("Cannot substract vecror from array element by element, neither row-wise nor column-wise. Number of elements in vector [" + y.length + "] must be equal to any of dimmension parameters of first array [" + x.length + "][" + x[0].length + "].");
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

    public static double[][] addValue(double[][] x, double[] y) throws MyException {
        double[][] temp = new double[x.length][x[0].length];

        if (x.length != y.length && x[0].length != y.length)
            throw new MyException("Cannot add vecror to array element by element, neither row-wise nor column-wise. Number of elements in vector [" + y.length + "] must be equal to any of dimmension parameters of first array [" + x.length + "][" + x[0].length + "].");

        // [n][m] + [n][1], m times
        // [n][m] + [1][m] n times
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

    public static double[] addMatrixes(double[] x, double[] y) throws MyException {
        double[] temp = new double[x.length];

        if (x.length != y.length)
            throw new MyException("Cannot add vectors el by el. Vectors must have same length, while it is [" + x.length + "] and [" + y.length + "].");

        for (int i = 0; i < x.length; i++)
            temp[i] = x[i] + y[i];
        return temp;
    }

    public static double[][] addMatrixes(double[][] x, double[][] y) throws MyException {

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrixes must have equal dimensions");
            //return null;
            throw new MyException("While adding matrixes element by element, they must have equal dimmensions, while it is [" + x.length + "][" + x[0].length + "] and [" + y.length + "][" + y[0].length + "].");
        }

        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] + y[i][j];
        }
        return temp;
    }

    public static double[][] substractMatrixes(double[][] x, double[][] y) throws MyException {

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrixes must have equal dimensions");
            //return null;
            throw new MyException("While substracting matrixes element by element, they must have equal dimmensions, while it is [" + x.length + "][" + x[0].length + "] and [" + y.length + "][" + y[0].length + "].");
        }

        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] - y[i][j];
        }
        return temp;
    }

    public static double sum(double[] x) {
        double result = 0;
        for (double v : x) {
            result += v;
        }
        return result;
    }

    public static double sum(double[][] x) {
        double result = 0;
        for (double[] doubles : x) {
            for (int j = 0; j < x[0].length; j++) {
                result += doubles[j];
            }
        }
        return result;
    }

    public static double[] sum(double[][] x, int axis) throws MyException {
        double[] result = null;

        if (axis != 0 && axis != 1)
            throw new MyException("Wrong axis, sholud be 1 or 2, and is " + axis);

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

    public static double[][] sum(double[][] x, double[][] y) throws MyException {

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrixes must have equal dimensions");
            //return null;
            throw new MyException("While adding matrixes element by element, they must have equal dimmensions, while it is [" + x.length + "][" + x[0].length + "] and [" + y.length + "][" + y[0].length + "].");
        }

        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = x[i][j] + y[i][j];
        }
        return temp;
    }


    public static double[] genRandMatrix(double max, int size) throws MyException {

        if (size <= 0)
            throw new MyException("Size cannot be less orequal to 0.");
        double[] x = new double[size];
        for (int i = 0; i < size; i++) {
            x[i] = Math.random() * max;
        }
        return x;
    }

    public static double[][] genRandMatrix(double max, int size_x, int size_y) throws MyException {
        if (size_x <= 0 || size_y <= 0)
            throw new MyException("Size cannot be less orequal to 0.");
        double[][] x = new double[size_x][size_y];
        for (int i = 0; i < size_x; i++) {
            for (int j = 0; j < size_y; j++) {
                x[i][j] = Math.random() * max;
            }
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

    public static double[] minimum(double[] x, double[] y) throws MyException {
        double[] temp = new double[x.length];
        if (x.length != y.length)
            throw new MyException("Cannot search minimum value from two vectors of different length - [" + x.length + "] and [" + y.length + "].");
        for (int i = 0; i < x.length; i++) {
            temp[i] = Math.min(y[i], x[i]);
        }


        return temp;
    }

    public static double[] select_row(double[][] x, int y) throws MyException {

        if (y > x.length - 1)
            throw new MyException("Selected row out of range - " + y + ". of " + x.length + " rows (remember aobout 0th row!).");

        double[] result = new double[x[0].length];
        System.arraycopy(x[y], 0, result, 0, x[0].length);
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

        try {
            double XX = einsum(x);
            distances = Matrices.multiplyByMatrix(y, x);
            distances = Matrices.row_mul(distances, -2);
            distances = Matrices.addValue(distances, XX);
            distances = Matrices.addMatrixes(distances, z);
        } catch (Exception myEx) {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
        return distances;
    }

    public static double[][] euclidean_distances(double[][] x, double[][] y, double[] z) {
        //double [][] result = new double[x.length][y.length];
        double[][] distances = null;
        double[] XX = null;
        try {
            XX = einsum(x);
            distances = Matrices.multiplyByMatrix(x, Matrices.transpose(y));
            distances = Matrices.row_mul(distances, -2);
            distances = Matrices.addValue(distances, XX);
            distances = Matrices.addValue(distances, z);
        } catch (Exception myEx) {
            System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
        return distances;
    }

    public static double[][] cov(double[][] x) {

        double[][] temp = null;
        double[] X_mean = null;

        try {
            temp = Matrices.copy2dArray(x);
            //////////substracting mean //////////////
            X_mean = Statistics.getMean(Matrices.transpose(x));
            for (int j = 0; j < x[0].length; j++) {
                for (int i = 0; i < x.length; i++) {
                    temp[i][j] -= X_mean[i];
                }
            }

            temp = Matrices.divideByValue(Matrices.multiplyByMatrix(x, Matrices.transpose(temp)), (double) x[0].length - 1);
        } catch (Exception myEx) {
            System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }

        return temp;
    }

    public static double[][] divideByValue(double[][] x, double y) throws MyException {
        if (y == 0)
            throw new MyException("Cannot divide by 0");
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

    public static double[] makeLog(double[] x) throws MyException {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {

            if (x[i] <= 0)
                throw new MyException("Cannot make Log of value below 0 - Log(" + x[i] + "), (index " + i + ").");
            temp[i] = Math.log(x[i]);

        }
        return temp;
    }

    public static double[][] makeLog(double[][] x) throws MyException {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {

                if (x[i][j] <= 0)
                    throw new MyException("Cannot make Log of value below 0 - Log(" + x[i][j] + "), (index [" + i + "," + j + "]).");

                temp[i][j] = Math.log(x[i][j]);
            }
        }
        return temp;
    }

    public static double[] invertElements(double[] x) throws MyException {
        double[] temp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (x[i] == 0)
                throw new MyException("While inverting values, cannot divide by 0, (index " + i + ").");
            temp[i] = 1 / (x[i]);
        }
        return temp;
    }

    public static double[][] invertElements(double[][] x) throws MyException {
        // 1.0 / a[m][n]
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {

                if (x[i][j] <= 0)
                    throw new MyException("While inverting values, cannot divide by 0 (index [" + i + "," + j + "]).");
                temp[i][j] = 1 / (x[i][j]);
            }
        }
        return temp;
    }

    public static double[][] divideElements(double[][] x, double[][] y) throws MyException {
        //a[0][0]/b[0][0] ,  a[m][n]/b[m][n] ...

        if (x.length != y.length || x[0].length != y[0].length) {
            //System.out.println("Matrixes must have equal dimensions");
            //return null;
            throw new MyException("While dividing element by element, they must have equal dimmensions, now it is [" + x.length + "][" + x[0].length + "] and [" + y.length + "][" + y[0].length + "].");
        }

        double[][] result = new double[x.length][x[0].length];

        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                if (y[i][j] <= 0)
                    throw new MyException("While inverting values, cannot divide by 0 (y[" + i + "][" + j + "]).");
                result[i][j] = x[i][j] / y[i][j];
            }
        }


        return result;
    }

    public static double[][] power(double[][] x, double y) {
        double[][] temp = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++)
                temp[i][j] = Math.pow(x[i][j], y);
        }
        return temp;
    }

    public static double[] logsumexp(double[][] data) {

        double[] out = null;
        try {
            double[][] temp = Matrices.transpose(data);
            double[] vmax = Matrices.max(temp, 0);
            out = Matrices.makeLog(Matrices.sum(Matrices.exp(Matrices.substractValue(temp, vmax)), 0));
            out = Matrices.addMatrixes(out, vmax);
        } catch (Exception myEx) {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
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

    public static double[] max(double[][] x, int axis) throws MyException {
        double[] vmax = null;

        if (axis != 0 && axis != 1)
            throw new MyException("Wrong axis, sholud be 0 or 1, and is " + axis);

        if (axis == 0) {
            vmax = new double[x[0].length];

            for (int i = 0; i < x[0].length; i++) {
                vmax[i] = Double.NEGATIVE_INFINITY;
            }  //JAK CO TO USUN��!!!

            for (int i = 0; i < x[0].length; i++) {
                for (double[] doubles : x)
                    if (vmax[i] < doubles[i])
                        vmax[i] = doubles[i];
            }
        } else {
            vmax = new double[x.length];

            for (int i = 0; i < x.length; i++) {
                vmax[i] = Double.NEGATIVE_INFINITY;
            }// JAK CO TO USUN��!!!


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

    public static double[][] duplicate(double[] data, int n) throws MyException {
        if (n < 1)
            throw new MyException("Can not duplicate less than once");
        double[][] temp = new double[n][data.length];
        for (int i = 0; i < n; i++)
            temp[i] = data;
        return temp;
    }

    public static double[][] copy2dArray(double[][] x) {
        double[][] temp = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double[] aMatrix = x[i];
            int aLength = aMatrix.length;
            temp[i] = new double[aLength];
            System.arraycopy(aMatrix, 0, temp[i], 0, aLength);
        }
        return temp;
    }

}
