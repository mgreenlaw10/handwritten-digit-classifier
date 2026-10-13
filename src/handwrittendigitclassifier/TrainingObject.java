package handwrittendigitclassifier;

public class TrainingObject 
{
    public final float[] pixels;
    public final int label;

    public TrainingObject(float[] pixels, int label) 
    {
        this.pixels = pixels;
        this.label = label;
    }
}
