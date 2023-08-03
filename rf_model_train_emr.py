from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creation a Spark session
spark = SparkSession.builder.appName("wine").config('spark.ui.port', '4050').getOrCreate()

# Loading dataset
df = spark.read.format("csv").load("s3://rs228-bucket/TrainingDataset.csv", header=True, inferSchema=True, sep=";")

# Removing double quotes from column names
df = df.toDF(*[name.strip('""') for name in df.columns])

# Creation of feature vector using VectorAssembler
vector_assembler = VectorAssembler(inputCols=df.columns[1:-1], outputCol='features')
df = vector_assembler.transform(df)

# Splitting the dataset into train and test with seed 99
(training, test) = df.randomSplit([0.7, 0.3], seed=99)

# Creation of random forest model
rf = RandomForestClassifier(featuresCol='features', labelCol='quality', numTrees=21, maxDepth=30, seed=42)
rf_model = rf.fit(training)

# Making predictions on the test dataset
predictions = rf_model.transform(test)

# Show the predictions (quality and prediction columns)
predictions.select("quality", "prediction").show()

# Save the predictions to "test_results.csv" file
predictions.select("quality", "prediction").write.csv("s3://rs228-bucket/prediction_results/rf_test_results.csv", header=True)

# Calculating the F1-score
evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(predictions)
print("F1 Score:", f1_score)

# Save the training model
rf_model.save("s3://rs228-bucket/models/rf_model.model")
