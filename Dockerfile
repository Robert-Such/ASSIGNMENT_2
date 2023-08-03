# Use an official openjdk image as a base
FROM openjdk:8-jdk-slim

# Set environment variable for Spark and PySpark
ENV SPARK_VERSION=3.1.2
ENV PYSPARK_VERSION=3.1.2

# Install necessary Python packages and PySpark
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && \
    pip3 install pyspark==${PYSPARK_VERSION} numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Spark
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -qO- "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz" | tar -xz -C /opt && \
    ln -s "/opt/spark-${SPARK_VERSION}-bin-hadoop3.2" /opt/spark && \
    apt-get remove -y wget && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Spark home environment variable
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Copy the prediction script and the trained model to the container
COPY prediction_program.py /app/prediction_program.py
COPY rf_model.model /app/rf_model.model

# Set the working directory
WORKDIR /app

# Run the prediction script when the container starts
CMD ["python3", "prediction_program.py", "/app/test_file.csv"]



