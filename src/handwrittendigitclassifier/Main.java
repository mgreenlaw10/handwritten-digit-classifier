package handwrittendigitclassifier;

import java.io.IOException;
import java.util.List;

public class Main {
    
    static final int TRAINING_ITERATIONS = 10;
    static final float TRAINING_FACTOR = 0.1f;
    
    public static void main(String[] args) 
    {
        try 
        {
            List<TrainingObject> trainingSet = MNISTLoader.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

            NeuralNetwork nn = new NeuralNetwork(trainingSet.get(0).pixels.length, 10, 2, new int[] {16, 16}, ActivationFunction.SIGMOID);
            
            for (int i = 0; i < TRAINING_ITERATIONS; i++)
            {
                nn.train(trainingSet, TRAINING_FACTOR);
                System.out.println("Finished training iteration #" + (i+1));
            }
            
            
            int objectNumber = 1;
            int totalClassifications = 0;
            int correctClassifications = 0;
            
            for (TrainingObject object : trainingSet)
            {
                float[] result = nn.apply(object.pixels);
                int predictedClass = LinearAlgebra.vectorMaxIndex(result);
                float predictedClassConfidence = LinearAlgebra.vectorMax(result);
                
                System.out.println("Object " + objectNumber++ + ":");
                System.out.println("True class: " + object.label);
                System.out.println("Classification: " + predictedClass + " (probability: " + predictedClassConfidence + ")");

                totalClassifications++;
                if (predictedClass == object.label) correctClassifications++;
            }
            
            System.out.println("\nTotal accuracy: " + (float)correctClassifications / (float)totalClassifications);
            
        }
        catch (IOException e) 
        {
            System.out.println(e);
        }
    }
}
