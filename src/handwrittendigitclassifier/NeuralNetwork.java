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
    
    public float[][] layerOutputCache;
    
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
        this.layerOutputCache = new float[numHiddenLayers + 1][];
        initializeLayerFunctions(activationFunction);
    }
    
    public float[] apply(float[] inputVector)
    {
        assert(inputVector.length == inputDimension);
        
        float[] result = inputVector;
        for (int i = 0; i < layerFunctions.length; i++)
        {
            result = layerFunctions[i].apply(result);
            layerOutputCache[i] = result;
        }
        return result;
    }
    
    public void train(List<TrainingObject> trainingObjects, float learningRate)
    {
        List<float[][]> weightGradients = new ArrayList<>();
        List<float[]> biasGradients = new ArrayList<>();
        
        for (LayerFunction layer : layerFunctions)
        {
            weightGradients.add(new float[layer.outputDimension][layer.inputDimension]);
            biasGradients.add(new float[layer.outputDimension]);
        }
        
        int lastLayer = layerFunctions.length - 1;
        float[][] errorVectors = new float[layerFunctions.length][];
        
        for (TrainingObject object : trainingObjects)
        {
            float[] approx = apply(object.pixels);
            float[] actual = new float[outputDimension];
            actual[object.label] = 1.0f;
            
            // Calculate error vector for final layer.
            // Softmax + cross entropy derivative shortcut: dL/dz = approx - actual
            errorVectors[lastLayer] = LinearAlgebra.vectorDifference(approx, actual);
            
            // Propagate error vector calculation to previous layers
            for (int i = lastLayer - 1; i >= 0; i--)
            {
                float[][] wT = LinearAlgebra.matrixTranspose(layerFunctions[i + 1].weightMatrix);
                float[] dActivation = layerFunctions[i].activationFunction.derivativeAt(layerFunctions[i].lastPreActivationResult);
                float[] wTerr = LinearAlgebra.matrixVectorProduct(wT, errorVectors[i + 1]);
                
                errorVectors[i] = new float[wTerr.length];
                for (int j = 0; j < errorVectors[i].length; j++)
                {
                    errorVectors[i][j] = wTerr[j] * dActivation[j];
                }
            }
            
            // Calculate the gradient for each layer:
            //
            // dL/dA[i] = errorVectors[i] * input[i]^T
            // dL/dB[i] = errorVectors[i]
            //
            // Accumulate the sum of gradients over the training set for each layer.
            for (int i = 0; i < layerFunctions.length; i++)
            {
                float[] layerInput = (i > 0) ? layerOutputCache[i - 1] : object.pixels;
                
                weightGradients.set(i, LinearAlgebra.matrixSum(weightGradients.get(i), LinearAlgebra.outerProduct(errorVectors[i], layerInput)));
                biasGradients.set(i, LinearAlgebra.vectorSum(biasGradients.get(i), errorVectors[i]));
            }
        }
        
        // Apply the gradient descent.
        // Normalization against size of training set is baked into stepScale.
        float stepScale = -learningRate / trainingObjects.size();
        for (int i = 0; i < layerFunctions.length; i++)
        {
            LayerFunction layer = layerFunctions[i];
            layer.weightMatrix = LinearAlgebra.matrixSum(layer.weightMatrix, LinearAlgebra.matrixScalarProduct(weightGradients.get(i), stepScale));
            layer.biasVector = LinearAlgebra.vectorSum(layer.biasVector, LinearAlgebra.vectorScalarProduct(biasGradients.get(i), stepScale));
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
}
