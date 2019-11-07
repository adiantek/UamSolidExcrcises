package speakerrecognition;

import java.io.IOException;
import java.util.List;

import speakerrecognition.impl.MyException;
import speakerrecognition.impl.Speaker_model;

public interface SpeakerRecognition {

    public int[] openWavFile(String resourcePath) throws IOException, MyException;

    public int getSamplingFrequency(String resourceSOundFilePath) throws IOException, MyException;

    double[][] getMeansOfClustersFor2DdataByGMM(double[][] data, int numOfClusters);

    double[][] getMeansOfClustersFor2DdataByKMeans(double[][] data, int numOfClusters);

    double getLogProbabilityOfDataUnderModel(Speaker_model model, double[][] dataToBeTested) throws MyException;

    double[][] computeMFCC(int[] soundSamples, int fs);

    String recognize(List<Speaker_model> speakerModels, String resourceSoundSpeechFilePath) throws IOException, MyException;

    void printLogProbsForRecognition(List<Speaker_model> speakerModels, String resourceSoundSpeechFilePath) throws IOException, MyException;


}
