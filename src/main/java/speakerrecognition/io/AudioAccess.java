package speakerrecognition.io;

public interface AudioAccess {
    int[] getSamples();

    int getFs();

    int getNumOfChannels();
}
