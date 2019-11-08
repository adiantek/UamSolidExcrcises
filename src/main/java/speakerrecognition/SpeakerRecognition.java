package speakerrecognition;

import java.io.IOException;
import java.util.List;

import speakerrecognition.impl.SpeakerModel;

public interface SpeakerRecognition {

    public int[] openWavFile(String resourcePath) throws IOException;

    public int getSamplingFrequency(String resourceSOundFilePath) throws IOException;

    double[][] getMeansOfClustersFor2DdataByGMM(double[][] data, int numOfClusters);

    double[][] getMeansOfClustersFor2DdataByKMeans(double[][] data, int numOfClusters);

    double getLogProbabilityOfDataUnderModel(SpeakerModel model, double[][] dataToBeTested);

    double[][] computeMFCC(int[] soundSamples, int fs);

    String recognize(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath) throws IOException;

    void printLogProbsForRecognition(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath) throws IOException;


}
