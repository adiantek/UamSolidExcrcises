package speakerrecognition.impl;


import org.jtransforms.fft.DoubleFFT_1D;
import speakerrecognition.math.MFCCProcessingUtils;
import speakerrecognition.math.Matrices;


class MFCC {
    private int frame_len;
    private int frame_shift;
    private double[] window;
    private double[][] melfb_coeffs;
    private double[][] mfcc_coeffs = null;
    private int[] samples;
    private double[][] D1;

    MFCC(int[] x, int y) {
        this.samples = x;
        this.frame_len = 256;
        int fft_size = this.frame_len;
        this.frame_shift = setFrameShift(y);
        window = hamming(frame_len);
        int melfilterBands = 40;
        this.melfb_coeffs = MFCCProcessingUtils.melfb(melfilterBands, fft_size, y);
        this.D1 = MFCCProcessingUtils.dctmatrix(melfilterBands);
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
        extractMFCC();
        return this.mfcc_coeffs;
    }

    private void extractMFCC() {
        if (this.samples != null) {
            DoubleFFT_1D fftDo = new DoubleFFT_1D(this.frame_len);
            double[] fft1 = new double[this.frame_len * 2];
            double[] fftFinal = new double[this.frame_len / 2 + 1];
            //int[] x = this.samples;
            int frames_num = (int) ((double) (this.samples.length - this.frame_len) / (double) (this.frame_shift)) + 1;
            this.mfcc_coeffs = new double[frames_num][MFCCProcessingUtils.MFCC_NUM];
            double[] frame = new double[this.frame_len];

            for (int i = 0; i < frames_num; i++) {

                for (int j = 0; j < this.frame_len; j++) {
                    frame[j] = this.samples[i * this.frame_shift + j];
                }
                frame = Matrices.row_mul(frame, window);
                frame = MFCCProcessingUtils.preemphasis(frame);
                System.arraycopy(frame, 0, fft1, 0, this.frame_len);
                fftDo.realForwardFull(fft1);
                for (int k = 0; k < (this.frame_len / 2 + 1); k++) {
                    fftFinal[k] = Math.pow(Math.sqrt(Math.pow(fft1[k * 2], 2) + Math.pow(fft1[k * 2 + 1], 2)), 2);

                    double power_spectrum_floor = 0.0001;
                    if (fftFinal[k] < power_spectrum_floor) fftFinal[k] = power_spectrum_floor;
                }

                double[] dotProd = Matrices.multiplyByMatrix(this.melfb_coeffs, fftFinal);
                for (int j = 0; j < dotProd.length; j++) {
                    dotProd[j] = Math.log(dotProd[j]);
                }
                dotProd = Matrices.multiplyByMatrix(this.D1, dotProd);
                this.mfcc_coeffs[i] = dotProd;
            }
        } else {
            System.out.println("Vector of input samples is null");
        }
    }
}
