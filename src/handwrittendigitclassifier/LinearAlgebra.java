package handwrittendigitclassifier;

import java.util.function.BiFunction;
import java.util.function.Function;

public class LinearAlgebra 
{
    public static float[] matrixVectorProduct(float[][] matrix, float[] vector)
    {
        assert(matrix[0].length == vector.length);
        
        float[] product = new float[matrix.length];
        for (int i = 0; i < matrix.length; i++)
        {
            float[] matrixRow = matrix[i];
            product[i] = dotProduct(matrixRow, vector);
        }
        return product;
    }
    
    public static float dotProduct(float[] vectorA, float[] vectorB)
    {
        assert(vectorA.length == vectorB.length);
        
        float dotProduct = 0f;
        for (int i = 0; i < vectorA.length; i++)
        {
            dotProduct += (vectorA[i] * vectorB[i]);
        }
        return dotProduct;
    }
    
    public static float[] vectorSum(float[] vectorA, float[] vectorB)
    {
        assert(vectorA.length == vectorB.length);
        
        float[] vectorSum = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++)
        {
            vectorSum[i] = vectorA[i] + vectorB[i];
        }
        return vectorSum;
    }
    
    public static float[][] matrixSum(float[][] matrixA, float[][] matrixB)
    {
        assert(matrixA.length == matrixB.length);
        assert(matrixA[0].length == matrixB[0].length);
        
        float[][] result = new float[matrixA.length][matrixA[0].length];
        
        for (int i = 0; i < matrixA.length; i++)
        {
            result[i] = vectorSum(matrixA[i], matrixB[i]);
        }
        return result;
    }
    
    public static float[] vectorDifference(float[] vectorA, float[] vectorB)
    {
        assert(vectorA.length == vectorB.length);
        
        float[] vectorSum = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++)
        {
            vectorSum[i] = vectorA[i] - vectorB[i];
        }
        return vectorSum;
    }
    
    public static float[] componentwiseApplication(float[] vector, Function<Float, Float> operator)
    {   
        float[] result = new float[vector.length];
        for (int i = 0; i < vector.length; i++)
        {
            result[i] = operator.apply(vector[i]);
        }
        return result;
    }
    
    public static float sumOfComponentwiseApplication(float[] vector, Function<Float, Float> operator)
    {
        float sum = 0f;
        for (int i = 0; i < vector.length; i++)
        {
            sum += operator.apply(vector[i]);
        }
        return sum;
    }
    
    public static float sumOfComponentwiseApplication(float[] vectorA, float[] vectorB, BiFunction<Float, Float, Float> operator)
    {
        assert(vectorA.length == vectorB.length);
        
        float sum = 0f;
        for (int i = 0; i < vectorA.length; i++)
        {
            sum += operator.apply(vectorA[i], vectorB[i]);
        }
        return sum;
    }
    
    public static float maxValue(float[] vector)
    {
        float max = vector[0];
        for (float n : vector)
        {
            if (n > max) max = n;
        }
        return max;
    }
    
    public static float[][] outerProduct(float[] vectorA, float[] vectorB)
    {
        float[][] result = new float[vectorA.length][vectorB.length];
        
        for (int i = 0; i < vectorA.length; i++)
        {
            for (int j = 0; j < vectorB.length; j++)
            {
                result[i][j] = vectorA[i] * vectorB[j];
            }
        }
        return result;
    }
}
