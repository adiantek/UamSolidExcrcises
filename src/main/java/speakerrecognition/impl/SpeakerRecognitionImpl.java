package speakerrecognition.impl;

import java.io.IOException;
import java.util.List;

import speakerrecognition.SpeakerRecognition;
import speakerrecognition.impl.gmm.GMM;
import speakerrecognition.impl.kmeans.KMeans;
import speakerrecognition.io.WavFile;

public class SpeakerRecognitionImpl implements SpeakerRecognition {

    public int[] openWavFile(String resourcePath) throws IOException {
        WavFile wavFile = new WavFile(resourcePath);
        wavFile.open();
        return wavFile.getSamples();
    }

    public int getSamplingFrequency(String resourceSOundFilePath) throws IOException {
        WavFile wavFile = new WavFile(resourceSOundFilePath);
        wavFile.open();
        return wavFile.getFs();
    }

    public double[][] getMeansOfClustersFor2DdataByGMM(double[][] data, int numOfClusters) {
        GMM gmm = new GMM(data, numOfClusters);
        gmm.fit();
        return gmm.get_means();
    }

    public double[][] getMeansOfClustersFor2DdataByKMeans(double[][] data, int numOfClusters) {
        KMeans kMeans = new KMeans(data, numOfClusters);
        kMeans.fit();
        return kMeans.get_centers();
    }

    public double getLogProbabilityOfDataUnderModel(SpeakerModel model, double[][] dataToBeTested) {
        return model.getScore(dataToBeTested);
    }

    public double[][] computeMFCC(int[] soundSamples, int fs) {
        MFCC mfcc = new MFCC(soundSamples, fs);
        double[][] speaker_mfcc = mfcc.getMFCC();
        return speaker_mfcc;
    }

    public String recognize(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath) throws IOException {
        double finalScore = Long.MIN_VALUE;
        String bestFittingModelName = "";
        for (SpeakerModel model : speakerModels) {
            WavFile wavFile1 = new WavFile(resourceSoundSpeechFilePath);
            wavFile1.open();
            int[] x3 = wavFile1.getSamples();
            int fs3 = wavFile1.getFs();
            MFCC mfcc3 = new MFCC(x3, fs3);
            double[][] speaker_mfcc3 = mfcc3.getMFCC();
            double scoreForTest1 = model.getScore(speaker_mfcc3);
            if (scoreForTest1 > finalScore) {
                finalScore = scoreForTest1;
                bestFittingModelName = model.getName();
            }

        }
        return "Test speech from file " + resourceSoundSpeechFilePath + " is most similar to model " + bestFittingModelName;
    }

    public void printLogProbsForRecognition(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath)
            throws IOException {
        for (SpeakerModel model : speakerModels) {
            WavFile wavFile1 = new WavFile(resourceSoundSpeechFilePath);
            wavFile1.open();
            int[] x3 = wavFile1.getSamples();
            int fs3 = wavFile1.getFs();
            MFCC mfcc3 = new MFCC(x3, fs3);
            double[][] speaker_mfcc3 = mfcc3.getMFCC();
            double scoreForTest1 = model.getScore(speaker_mfcc3);
            System.out.println("Test speech from file " + resourceSoundSpeechFilePath + " is similar to model " + model.getName() + " with log probability " + scoreForTest1);

        }

    }

}
