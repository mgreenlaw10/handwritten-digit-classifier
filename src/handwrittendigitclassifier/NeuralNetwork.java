package handwrittendigitclassifier;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork 
{
    public final int inputDimension;
    public final int outputDimension;
    public final int numHiddenLayers;
    public final int[] hiddenLayerWidths;
    
    public final LayerFunction[] layerFunctions;
    
    public List<float[]> layerOutputCache;
    
    public NeuralNetwork(int inputDimension, int outputDimension, int numHiddenLayers, int[] hiddenLayerWidths, ActivationFunction activationFunction)
    {
        assert(inputDimension > 0);
        assert(outputDimension > 0);
        assert(numHiddenLayers > 0);
        assert(hiddenLayerWidths.length == numHiddenLayers);
        
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.numHiddenLayers = numHiddenLayers;
        this.hiddenLayerWidths = hiddenLayerWidths;
        
        this.layerFunctions = new LayerFunction[numHiddenLayers + 1];
        this.layerOutputCache = new ArrayList<>();
        initializeLayerFunctions(activationFunction);
    }
    
    public float[] apply(float[] inputVector)
    {
        assert(inputVector.length == inputDimension);
        
        float[] result = inputVector;
        for (int i = 0; i < layerFunctions.length; i++)
        {
            result = layerFunctions[i].apply(result);
            layerOutputCache.set(i, result);
        }
        return result;
    }
    
    public void train(List<TrainingObject> trainingObjects)
    {
        
        LayerFunction h = layerFunctions[layerFunctions.length - 1];
        float[][] gradientMatrix = new float[h.outputDimension][h.inputDimension];
        
        for (TrainingObject object : trainingObjects)
        {
            float[] approx = apply(object.pixels);
            float[] actual = new float[outputDimension];
            actual[object.label] = 1.0f;
            
            float[] dLdz = LinearAlgebra.vectorDifference(approx, actual);
            float[] dzdA = layerOutputCache.get(layerFunctions.length - 2);
            
            float[][] dLdA = LinearAlgebra.outerProduct(dLdz, dzdA);
            // this is just the gradient matrix for the final layer function weight matrix
            gradientMatrix = LinearAlgebra.matrixSum(gradientMatrix, dLdA);
        } 
    }
    
    private void initializeLayerFunctions(ActivationFunction activationFunction)
    {
        for (int i = 0; i < layerFunctions.length; i++)
        {
            int lfInputDimension;
            int lfOutputDimension;
            
            if (i == 0)
                lfInputDimension = inputDimension;
            else
                lfInputDimension = hiddenLayerWidths[i - 1];
            
            if (i == layerFunctions.length - 1)
                lfOutputDimension = outputDimension;
            else
                lfOutputDimension = hiddenLayerWidths[i];
            
            // Final layer activation function is always softmax
            layerFunctions[i] = new LayerFunction (
                lfInputDimension, 
                lfOutputDimension, 
                (i == layerFunctions.length - 1)? ActivationFunction.SOFTMAX : activationFunction
            );
        }
    }
    
    
    
    
    private float calculateLoss()
    {
        return 0;
    }
}
