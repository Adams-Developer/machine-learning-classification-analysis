using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    class Program
    {

        /*Fields to hold paths to downloaded files
         _trainDataPath - path to the dataset used to train the model
         _testDataPath - path to the dataset used to evaluate the model
         _modelPath - path where the trained model is saved
         _textLoader - used to load and transform the datasets
        */
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        //Initialize variables in Main
        static void Main(string[] args)
        {
            //create an MLContext - for exception tracking and logging
            MLContext mlContext = new MLContext(seed: 0);

            //initialize the _textLoader in order to reuse
            //it for the needed datasets
            _textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            });

            var model = Train(mlContext, _trainDataPath);
        }

        /*
         * Train method excutes the following:
         * Loads the data
         * Extracts and transforms the data
         * Trains the model
         * Predicts sentiment based on test data
         * Returns the model
         */
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            //load the data via _textLoader
            //return IDataView(the input and output of Transforms
            IDataView dataView = _textLoader.Read(dataPath);

            //Extract and transform the data
            /* into a format that ML algorithms recognize
             * transform pipelines purpose is data featurization
             * which involves defining a set of features
             * The pipeline featurized the text column(SentimentTest)
             * into a numeric vector called Features
            */

            /*
             * Choosing the learning algorithm
             * Add the trainer by calling(mlContext...FeaturizeText)
             * which will return the decision tree learner used in the pipeline.
             * The tree is appended to the pipeline and accepts the featurzed data
             * and the Label input params to learn from the historic data.
             */
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            //Train the model
            /*
             * Train the model based on the dataset that has been 
             * loaded and transformed. Once the estimator has been defined,
             * you train your model using Fit while providing the already
             * loaded training data. This returns a model to use for predictions.
             * pipeline.Fit() trains the pipeline and returns a transformer 
             * based on the dataView passed in
             */
            Console.WriteLine("=============== Create and Train the Model ===============");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");

            Console.WriteLine();

            //return the model trained to use for evaluation
            return model;
        }

        //Evaluate the model
        /* The model has been created and trained
         * Now, we have to evaluate it with a different
         * dataset for quality assurance and validation
         * 
         * The model created in Train is passed in to be evaluated
         * 
         * Execute the following:
         * Load the test dataset
         * Creates the binary evaluator
         * Evaluates the model and create metrics
         * Displays the metrics
         */
        public static void Evaluate(MLContext mLContext, ITransformer model)
        {
            Evaluate(mLContext, model);

            //load the test dataset via _textloader
            IDataView dataView = _textLoader.Read(_testDataPath);

            //Use the ML model param(a transformer)
            //to input the features and return predictions
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

            var predictions = model.Transform(dataView);

            //The Evaluate method computes the quality metrics for
            //the predictionModel using the specified dataset

            /* Getting the metrics
             * The binaryClassificationEvaluator.CalibratedResult objec
             * contains metrics computed by binary classification evaluators
             * We want to display these to determine the quality of the model
             */
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");

            //Displaying the metrics for model validation
            Console.WriteLine();

            Console.WriteLine("Model quality metrics evaluation");

            Console.WriteLine("--------------------------------");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

            Console.WriteLine($"Auc: {metrics.Auc:P2}");

            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

            Console.WriteLine("=============== End of model evaluation ===============");

            //Save the model to a .zip file
            SaveModelAsFile(mLContext, model);
        }

        //Predict the test data outcome with the saved model
        /*
         * This method will execute the following tasks:
         * Creates batch test data
         * Predicts sentiment based on test data
         * Combines test data and predictions for reporting
         * Displays the predicted results
         */
        public static void PredictWithModelLoadedFromFile(MLContext mlContext)
        {
            PredictWithModelLoadedFromFile(mlContext);

            //Add some comments to test the trained model's predictions
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This is a very rude movie"
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            //Load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine
            var sentimentStreamingDataView = mlContext.CreateStreamingDataView(sentiments);
            var predictions = loadedModel.Transform(sentimentStreamingDataView);

            // Use the model to predict whether comment data is toxic (1) or nice (0).
            var predictedResults = predictions.AsEnumerable<SentimentPrediction>(mlContext, reuseRowObject: false);

            //Model operationalization: prediction
            //Display SentimentText and corresponding sentiment prediction
            //in order to share the results and act accordingly
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

            //Befor displaying the predicted result,
            //combine the sentiment and prediction together
            //to see the original comment with its predicted sentiment
            var sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));

            //Display the results of the combined SentimentTest and Sentiment
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {item.prediction.Probability} ");
            }

            Console.WriteLine("=============== End of predictions ===============");

        }

        //Create save method
        private static void SaveModelAsFile(MLContext mLContext, ITransformer model)
        {
            //saving the model so that it can be 
            //reused and consumed in other applications
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mLContext.Model.Save(model, fs);

            //Display where the file was written
            Console.WriteLine("the model is saved to {0}", _modelPath);
        }

        //Predict the test data outcome with the model
        //and a single comment
        /*
         * The Predict method executes the following:
         * Creates a single comment of test data
         * Predicts sentiment based on test data
         * Combines test data and predictions for reporting
         * Displays the predicted results
         */
        private static void Predict(MLContext mLContext, ITransformer model)
        {
            Predict(mLContext, model);

            //The model is a  transformer that operates 
            //on many rows of data, a very common production
            //scenario is a need for predictions on individual examples
            var predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mLContext);

            //Adding a comment to test the trained model's prediction
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This is a very rude movie"
            };

            //Predictng  the Toxic or Non Toxic sentiment of 
            //a single instance of the comment data.
            //The pipeline is in sync during training and prediction
            var resultprediction = predictionFunction.Predict(sampleStatement);

            //Prediction
            //Displaying Sentimenttext and corresponding sentiment prediction
            //in order to share the results and act on them accordingly
            //This is called operationalization(using the returned data as part
            // of the operational policies)
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Not Toxic")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }


    }
}
