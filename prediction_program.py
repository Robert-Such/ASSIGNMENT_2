import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creating a Spark session
spark = SparkSession.builder.appName("wine_prediction").getOrCreate()

# Loading saved model
model_path = "rf_model.model"
rf_model = RandomForestClassificationModel.load(model_path)

# Getting pathname of the test file from command-line arguments
if len(sys.argv) != 2:
    print("Usage: python program_name.py <test_file_path>")
    sys.exit(1)

test_file_path = sys.argv[1]

# Loading the test dataset
test_df = spark.read.format("csv").load(test_file_path, header=True, inferSchema=True, sep=";")

# Removing double quotes from column names
test_df = test_df.toDF(*[name.strip('""') for name in test_df.columns])

# Creating feature vector using VectorAssembler
vector_assembler = VectorAssembler(inputCols=test_df.columns[1:-1], outputCol='features')
test_df = vector_assembler.transform(test_df)

# Making predictions on the test dataset using the loaded model
predictions = rf_model.transform(test_df)

# Showing the predictions (quality and prediction columns)
predictions.select("quality", "prediction").show()

# Saving the predictions to "test_results.csv" file
predictions.select("quality", "prediction").write.csv("prediction_test_results.csv", header=True)

# Calculating the F1-score
evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(predictions)
print("F1 Score:", f1_score)