package handwrittendigitclassifier;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) 
    {
        try 
        {
            List<TrainingObject> trainingSet = MNISTLoader.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
            float[] pixels = trainingSet.get(0).pixels;
            System.out.println(trainingSet.size()); 
            NeuralNetwork nn = new NeuralNetwork(pixels.length, 10, 2, new int[] {16, 16}, ActivationFunction.SIGMOID);
            float[] result = nn.apply(pixels);
            
            System.out.println("Result:");
            for (float n : result)
            {
                System.out.print(n + ", ");
            }
        }
        catch (IOException e) 
        {
            System.out.println(e);
        }
    }
}
