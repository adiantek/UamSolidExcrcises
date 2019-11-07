package speakerrecognition.impl;

import java.io.File;
import java.util.Arrays;


public class TimitFileNames {
    private String timitPath = null;
    private String[] names = new String[630];

    //names[0]="aaa";


    TimitFileNames(String name) {
        this.timitPath = name;

        File folder = new File(this.timitPath);
        File[] listOfFiles = folder.listFiles();
        int j = 0;
        for (int i = 0; i < listOfFiles.length; i++) {
            String extension = listOfFiles[i].getName().substring(listOfFiles[i].getName().length() - 3);
            if (listOfFiles[i].isFile() && extension.equals("wav")) {
                String fileName = listOfFiles[i].getName().substring(0, listOfFiles[i].getName().length() - 5);
                if (!Arrays.asList(this.names).contains(fileName)) {
                    names[j] = fileName;
                    j++;
                    //System.out.println(Integer.toString(j));
                }
            }
        }
        //System.out.println("ok");

    }

    public String[] getTimitNames() {
        return this.names;
    }


}
