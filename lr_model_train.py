from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creation of Spark session
spark = SparkSession.builder.appName("wine").config('spark.ui.port', '4050').getOrCreate()

# Loading dataset
df = spark.read.format("csv").load("TrainingDataset.csv", header=True, inferSchema=True, sep=";")

# Removing double quotes from column names
df = df.toDF(*[name.strip('""') for name in df.columns])

# Creation of feature vector using VectorAssembler
vector_assembler = VectorAssembler(inputCols=df.columns[1:-1], outputCol='features')
df = vector_assembler.transform(df)

# Splitting the dataset into train and test with seed 99
(training, test) = df.randomSplit([0.7, 0.3], seed=99)

# Creation of Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='quality', maxIter=100, regParam=0.0, elasticNetParam=0.0)
lr_model = lr.fit(training)

# Making predictions on the test dataset
predictions = lr_model.transform(test)

# Show the predictions (quality and prediction columns)
predictions.select("quality", "prediction").show()


# Save the predictions to "test_results.csv" file
predictions.select("quality", "prediction").write.csv("lr_test_results.csv", header=True)

# Calculate F1-score
evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(predictions)
print("F1 Score:", f1_score)

# Save the trained logistic regression model
lr_model.save("lr_model.model")
