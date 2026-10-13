
package handwrittendigitclassifier;

import java.util.function.Function;

public class ActivationFunction 
{
    private final Function<float[], float[]> func;
    
    public static final ActivationFunction RELU;
    public static final ActivationFunction SIGMOID;
    public static final ActivationFunction SOFTMAX;
    
    static 
    {
        RELU = new ActivationFunction((vector) -> 
        {
            return LinearAlgebra.componentwiseApplication(vector, (n) -> Math.max(0, n));
        });
        
        SIGMOID = new ActivationFunction((vector) -> 
        {
            return LinearAlgebra.componentwiseApplication(vector, (n) -> 1f / (1f + (float)Math.exp(-n)));
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
        });
    }
    
    private ActivationFunction(Function<float[], float[]> func) 
    {
        this.func = func;
    }
    
    public float[] apply(float[] vector)
    {
        return func.apply(vector);
    }
}
