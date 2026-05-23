from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import from_json


spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("WeatherSpeedLayer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,com.datastax.spark:spark-cassandra-connector_2.13:3.5.0") \
    .config("spark.cassandra.connection.host", "cassandra1") \
    .getOrCreate()

sdf_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker-1:9092,broker-2:9092,broker-3:9092")\
    .option("subscribe", "weather-data") \
    .option("startingOffsets", "earliest") \
    .load()

# Conversion de la valeur binaire en chaîne de caractères
sdf_string = sdf_raw.selectExpr("CAST(value AS STRING)")

# Schéma adapté aux données OpenWeatherMap
weather_schema = StructType([
    StructField("name", StringType()),
    StructField("main", StructType([
        StructField("temp", DoubleType()),
        StructField("pressure", IntegerType()),
        StructField("humidity", IntegerType())
    ])),
    StructField("wind", StructType([
        StructField("speed", DoubleType())
    ])),
    StructField("weather", ArrayType(StructType([
        StructField("main", StringType())
    ]))),
    StructField("visibility", IntegerType()),
    StructField("dt", LongType())
])

# Extraction des colonnes
sdf_weather = sdf_string.withColumn("data", from_json(col("value"), weather_schema)) \
    .select("data.*") \
    .withColumn("weather_main", col("weather").getItem(0).getField("main"))

# 1. Alertes Température
alerts_temp = sdf_weather.filter("main.temp < 0 OR main.temp > 30") \
    .select(col("name"), 
            when(col("main.temp") < 0, "GELEE").otherwise("CANICULE").alias("type"))

# 2. Alertes Vent
alerts_wind = sdf_weather.filter("(wind.speed * 3.6) > 60") \
    .select(col("name"), 
            lit("VENT FORT").alias("type"))

# 3. Alertes Conditions 
alerts_cond = sdf_weather.filter("weather_main IN ('Rain', 'Snow', 'Fog')") \
    .select(col("name"), 
            when(col("weather_main") == "Rain", "PLUIE")
            .when(col("weather_main") == "Snow", "NEIGE")
            .otherwise("BROUILLARD").alias("type"))


all_alerts = alerts_temp.union(alerts_wind).union(alerts_cond) \
    .withColumn("alert_time", current_timestamp())

# On transforme le timestamp 'dt' en format Timestamp Spark
sdf_with_time = sdf_weather.withColumn("timestamp", expr("CAST(from_unixtime(dt) AS TIMESTAMP)"))

# On définit un watermark de 10 minutes pour les calculs de fenêtre
sdf_press = sdf_with_time.withWatermark("timestamp", "10 minutes") \
    .groupBy("name", window("timestamp", "10 minutes", "2 minutes")) \
    .agg({"main.pressure": "avg"})

# Ecriture des alertes dans Cassandra
query = all_alerts.writeStream \
    .outputMode("append") \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "weather_data") \
    .option("table", "weather_alerts") \
    .option("checkpointLocation", "/opt/spark-apps/batch_data/checkpoint_alerts") \
    .start()

# Sauvegarde des données brutes dans le Data Lake (Parquet)
query_batch = sdf_weather.writeStream \
    .format("parquet") \
    .option("path", "/opt/spark-apps/batch_data/weather") \
    .option("checkpointLocation", "/opt/spark-apps/batch_data/checkpoint_parquet") \
    .trigger(processingTime='10 seconds') \
    .start()

spark.streams.awaitAnyTermination()