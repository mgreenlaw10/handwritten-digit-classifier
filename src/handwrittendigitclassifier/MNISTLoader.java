package handwrittendigitclassifier;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTLoader 
{
    public static List<TrainingObject> load(String imagePath, String labelPath) throws IOException 
    {
        try (
            DataInputStream imageStream =
                new DataInputStream(
                    new BufferedInputStream(
                        new FileInputStream(imagePath)));

            DataInputStream labelStream =
                new DataInputStream(
                    new BufferedInputStream(
                        new FileInputStream(labelPath)))
        ) {
            int imageMagic = imageStream.readInt();
            int numImages = imageStream.readInt();
            int rows = imageStream.readInt();
            int cols = imageStream.readInt();

            int labelMagic = labelStream.readInt();
            int numLabels = labelStream.readInt();

            if (imageMagic != 2051) 
            {
                throw new IOException("Invalid image file");
            }

            if (labelMagic != 2049) 
            {
                throw new IOException("Invalid label file");
            }

            if (numImages != numLabels) 
            {
                throw new IOException("Image count does not match label count");
            }

            List<TrainingObject> examples = new ArrayList<>(numImages);

            for (int i = 0; i < numImages; i++) 
            {
                float[] pixels = new float[rows * cols];

                for (int j = 0; j < pixels.length; j++) 
                {
                    int pixel = imageStream.readUnsignedByte();
                    pixels[j] = pixel / 255.0f;
                }

                int label = labelStream.readUnsignedByte();

                examples.add(new TrainingObject(pixels, label));
            }

            return examples;
        }
    }
}
