package handwrittendigitclassifier;

import java.util.Arrays;
import java.util.Random;

public class LayerFunction 
{
    float[][] weightMatrix;
    float[] biasVector;
    ActivationFunction activationFunction;
    
    public float[] lastPreActivationResult;
    
    int inputDimension;
    int outputDimension;
    
    // For initializing weights
    private static final Random rng = new Random();
    
    public LayerFunction(int inputDimension, int outputDimension, ActivationFunction activationFunction)
    {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.activationFunction = activationFunction;
        
        initializeParameters();
    }
     
    private void initializeParameters()
    {
        weightMatrix = new float[outputDimension][inputDimension];
        
        // Initialize weights randomly
        for (int i = 0; i < outputDimension; i++)
        {
            for (int j = 0; j < inputDimension; j++)
            {
                weightMatrix[i][j] = (float)rng.nextGaussian(0, 0.01);
            }
        }
        biasVector = new float[outputDimension];
        Arrays.fill(biasVector, 0f);
    }
    
    public float[] apply(float[] inputVector)
    {
        float[] affineResult = LinearAlgebra.vectorSum(LinearAlgebra.matrixVectorProduct(weightMatrix, inputVector), biasVector);
        this.lastPreActivationResult = affineResult;
        return activationFunction.apply(affineResult);
    }
}
