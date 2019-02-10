using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    /*
     * Input dataset class
     * 
     * float - value for sentiment
     * either positive or negative
     * string - for the comment
     */
    public class SentimentData
    {
        [Column(ordinal: "0", name: "Label")]
        public float Sentiment;

        [Column(ordinal: "1")]
        public string SentimentText;
    }

    /*
     * Prediction after the model has been trained
     * 
     * The boolean Sentiment and PredictedLabel column
     * The label is used to create and train the model
     * Also used with a second dataset to evaluate the model
     * The PredictedLabel is used during evaluation and prediction
     * For evaluation, an input with triaing data, the 
     * predicted values, and the model are used
     */
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }

}
