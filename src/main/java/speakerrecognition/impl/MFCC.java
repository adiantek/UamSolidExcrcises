package speakerrecognition.impl;


import org.jtransforms.fft.DoubleFFT_1D;
import speakerrecognition.math.MFCCProcessingUtils;
import speakerrecognition.math.Matrices;


public class MFCC {

    private int frame_len;
    private int frame_shift;
    private double[] window;
    private double[][] melfb_coeffs;
    private double[][] mfcc_coeffs = null;
    private int[] samples;
    private double[][] D1;

    public MFCC(int[] x, int y) {
        this.samples = x;
        this.frame_len = 256;
        int fft_size = this.frame_len;
        this.frame_shift = setFrameShift(y);
        window = hamming(frame_len);
        int melfilter_bands = 40;
        this.melfb_coeffs = melfb(melfilter_bands, fft_size, y);
        this.D1 = MFCCProcessingUtils.dctmatrix(melfilter_bands);
    }

    private int setFrameShift(int sample_rate) {
        return (int) (0.0125 * (double) (sample_rate));
    }

    private double[] hamming(int frame_len) {
        double[] window_temp = new double[frame_len];
        for (int i = 0; i < window_temp.length; i++) {
            window_temp[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI / (double) frame_len * ((double) i + 0.5));
        }
        return window_temp;
    }

    double[][] getMFCC() {
        extract_MFCC();
        return this.mfcc_coeffs;
    }

    private double[][] melfb(int p, int n, int fs) {
        // p - number of filterbanks
        // n - length of fft
        // fs - sample rate

        double f0 = 700 / (double) fs;
        int fn2 = (int) Math.floor((double) n / 2);
        double lr = Math.log((double) 1 + 0.5 / f0) / (p + 1);
        double[] CF = MFCCProcessingUtils.arrange(1, p + 1);

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
        double[] pf = MFCCProcessingUtils.arrange(b1, b4 + 1);

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
            row_energy = MFCCProcessingUtils.energy(temp_row);
            if (row_energy < 0.0001)
                temp_row[i] = i;
            else {
                while (row_energy > 1.01) {
                    temp_row = Matrices.row_mul(temp_row, 0.99);
                    row_energy = MFCCProcessingUtils.energy(temp_row);
                }
                while (row_energy < 0.99) {
                    temp_row = Matrices.row_mul(temp_row, 1.01);
                    row_energy = MFCCProcessingUtils.energy(temp_row);
                }
            }
            m[i] = temp_row;
        }
        return m;
    }

    private void extract_MFCC() {
        if (this.samples != null) {
            DoubleFFT_1D fftDo = new DoubleFFT_1D(this.frame_len);
            double[] fft1 = new double[this.frame_len * 2];
            double[] fft_final = new double[this.frame_len / 2 + 1];
            //int[] x = this.samples;
            int frames_num = (int) ((double) (this.samples.length - this.frame_len) / (double) (this.frame_shift)) + 1;
            this.mfcc_coeffs = new double[frames_num][MFCCProcessingUtils.MFCC_NUM];
            double[] frame = new double[this.frame_len];

            for (int i = 0; i < frames_num; i++) {

                for (int j = 0; j < this.frame_len; j++) {
                    frame[j] = this.samples[i * this.frame_shift + j];
                }

                try {
                    frame = Matrices.row_mul(frame, window);

                    frame = MFCCProcessingUtils.preemphasis(frame);
                    System.arraycopy(frame, 0, fft1, 0, this.frame_len);
                    fftDo.realForwardFull(fft1);
					/*for(double d: fft1) {
			          System.out.println(d);
					}*/

                    for (int k = 0; k < (this.frame_len / 2 + 1); k++) {
                        fft_final[k] = Math.pow(Math.sqrt(Math.pow(fft1[k * 2], 2) + Math.pow(fft1[k * 2 + 1], 2)), 2);

                        double power_spectrum_floor = 0.0001;
                        if (fft_final[k] < power_spectrum_floor) fft_final[k] = power_spectrum_floor;
                    }

                    double[] dot_prod = Matrices.multiplyByMatrix(this.melfb_coeffs, fft_final);
                    for (int j = 0; j < dot_prod.length; j++) {
                        dot_prod[j] = Math.log(dot_prod[j]);
                    }
                    //double[][]D1 = dctmatrix(melfilter_bands);
                    dot_prod = Matrices.multiplyByMatrix(this.D1, dot_prod);
                    this.mfcc_coeffs[i] = dot_prod;
                } catch (Exception myEx) {
                    System.out.println("An exception encourred: " + myEx.getMessage());
                    myEx.printStackTrace();
                    System.exit(1);
                }

            }
            //this.mfcc_coeffs =
        } else {
            System.out.println("Vector of input samples is null");
        }
    }
}
