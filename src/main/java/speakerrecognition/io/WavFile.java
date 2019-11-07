package speakerrecognition.io;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class WavFile implements AudioAccess {
    private int[] samples;
    private int fs;
    private Path filePath;
    private int channelsNum;

    public WavFile(String filePath) {
        this(Paths.get(filePath));
    }

    public WavFile(Path filePath) {
        this.filePath = filePath;
    }

    @Override
    public int[] getSamples() {
        return this.samples;
    }

    @Override
    public int getFs() {
        return this.fs;
    }

    @Override
    public int getNumOfChannels() {
        return channelsNum;
    }

    public void open() throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(filePath)).order(ByteOrder.LITTLE_ENDIAN);
        int samplesNum = bb.getInt(40);
        this.channelsNum = Short.toUnsignedInt(bb.getShort(22));
        samples = new int[samplesNum / 2 / this.channelsNum];
        this.fs = bb.getInt(24);
        bb.position(44);
        for (int i = 0; i < samples.length; i++) {
            samples[i] = 0;
            for (int j = 0; j < this.channelsNum; j++) {
                samples[i] += bb.getShort();
            }
            samples[i] /= this.channelsNum;
        }
        if (bb.hasRemaining()) {
            throw new IOException("Extra data in WAV file");
        }
    }
}
