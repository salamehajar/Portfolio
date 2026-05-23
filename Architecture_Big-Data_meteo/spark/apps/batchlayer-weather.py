from pyspark.sql import SparkSession
from pyspark.sql import functions as F

print("Démarrage de la Batch Layer...")

# 1. Configuration
spark = SparkSession.builder \
    .appName("WeatherBatchToCluster") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.13:3.5.0") \
    .config("spark.cassandra.connection.host", "cassandra1") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Lecture du Data Lake (Parquet)
input_path = "/opt/spark-apps/batch_data/weather"
df_history = spark.read.parquet(input_path)

# Heure actuelle en secondes
current_epoch = F.unix_timestamp(F.current_timestamp())
# Minuit aujourd'hui en secondes : pour pouvoir filtrer les données d'aujourd'hui
midnight_epoch = F.unix_timestamp(F.current_date())

# 3. Calcul des agrégations conditionnelles (1h et Aujourd'hui)
stats_df = df_history.groupBy("name").agg(
    
    # Moyenne sur la dernière heure
    F.avg(F.when(F.col("dt") >= (current_epoch - 3600), F.col("main.temp"))).alias("avg_temp_1h"),
    
    # Moyenne et Max d'aujourd'hui
    F.avg(F.when(F.col("dt") >= midnight_epoch, F.col("main.temp"))).alias("avg_temp_today"),
    F.max(F.when(F.col("dt") >= midnight_epoch, F.col("main.temp"))).alias("max_temp_today"),
    
    # Heure de la dernière donnée reçue
    F.max("dt").alias("last_update_ts")
)

# 4. Formatage final pour correspondre à Cassandra
final_df = stats_df.withColumn("last_update", F.from_unixtime("last_update_ts").cast("timestamp")) \
                   .withColumnRenamed("name", "city") \
                   .select("city", 
                           "avg_temp_1h", 
                           "avg_temp_today", 
                           "max_temp_today", 
                           "last_update")

# 5. Écriture vers Cassandra
print("Envoi des moyennes (1h et Jour) vers Cassandra...")
final_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="city_stats", keyspace="weather_data") \
    .mode("append") \
    .save()

print("Batch Layer terminée avec succès !")
spark.stop()