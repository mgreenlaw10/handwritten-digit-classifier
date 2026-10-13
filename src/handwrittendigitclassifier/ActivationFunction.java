
package handwrittendigitclassifier;

import java.util.function.Function;

public class ActivationFunction 
{
    private final Function<float[], float[]> func;
    private final Function<float[], float[]> derivative;
    
    public static final ActivationFunction RELU;
    public static final ActivationFunction SIGMOID;
    public static final ActivationFunction SOFTMAX;
    
    static 
    {
        RELU = new ActivationFunction((vector) -> 
        {
            return LinearAlgebra.componentwiseApplication(vector, (n) -> Math.max(0, n));
        },
        (vector) -> {
            return null;
        });
        
        SIGMOID = new ActivationFunction((vector) -> 
        {
            return LinearAlgebra.componentwiseApplication(vector, (n) -> 1f / (1f + (float)Math.exp(-n)));
        },
        (vector) -> 
        {
            float[] sigmaX = LinearAlgebra.componentwiseApplication(vector, (n) -> 1f / (1f + (float)Math.exp(-n)));
            return LinearAlgebra.componentwiseApplication(sigmaX, (n) -> n * (1-n));
        });
        
        SOFTMAX = new ActivationFunction((vector) -> 
        {
            float[] result = new float[vector.length];

            float alpha = LinearAlgebra.maxValue(vector);
            float expSum = LinearAlgebra.sumOfComponentwiseApplication(vector, (n) -> (float)Math.exp(n - alpha));

            for (int i = 0; i < vector.length; i++)
            {
                result[i] = (float)Math.exp(vector[i] - alpha) / expSum;
            }
            return result;
        },
        (vector) -> {
            return null;
        });
    }
    
    private ActivationFunction(Function<float[], float[]> func, Function<float[], float[]> derivative) 
    {
        this.func = func;
        this.derivative = derivative;
    }
    
    public float[] apply(float[] vector)
    {
        return func.apply(vector);
    }
    
    public float[] derivativeAt(float[] vector)
    {
        return derivative.apply(vector);
    }
}
