import findspark

findspark.init("/opt/manual/spark")
from pyspark.sql.types import *
from pyspark.sql import SparkSession, functions as F
import pandas as pd

pd.options.display.max_columns = None
####### Starting Spark Session  ########
"""
spark = SparkSession.builder \
    .appName("sensor") \
    .master("local[2]") \
    .config("spark.executer.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1,org.elasticsearch:elasticsearch-spark-30_2.12:7.12.1") \
    .enableHiveSupport() \
    .getOrCreate()
"""

spark = (
    SparkSession.builder
        .appName("Spark-Cassandra-Without-Catalog")
        .master("local[2]")
        .config("spark.driver.memory", "2048m")
        .config("spark.sql.shuffle.partitions", 4)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:7.12.1")
        .getOrCreate()
)

######## Reading all datas, defining schemas ###########

sensor_schema = "  BID integer  ,\
  SENSOR_ID  integer  , \
  INSERT_DT  timestamp  ,\
  COUNT  integer  ,\
  INTERSECTION_ID  integer  ,\
  ID  string  "

sensor_df = spark.read.format("csv") \
    .option("header", False) \
    .option("sep", ";") \
    .schema(sensor_schema) \
    .load("file:///home/train/final_project/sensordata/*")

sensor_df.limit(5).toPandas()
sensor_df.printSchema()

isc_sn_schema = "  ID integer  ,\
  LANE_ID  integer  , \
  SENSOR_ID  integer  ,\
  SENSOR_TYPE  integer  ,\
  DIRECTION  integer  ,\
  SENSOR_GROUP  string ,\
INTERSECTION_ID string "

isc_sn = spark.read.format("csv"). \
    options(header=False). \
    option("delimiter", ";"). \
    schema(isc_sn_schema). \
    load("file:///home/train/final_project/intsens/*")

isc_sn.limit(5).toPandas()

lane_schema = "  ID integer  ,\
  EDGE_ID  integer  , \
  LANE_NO  string  ,\
  INTERSECTION_ID  integer  ,\
  LANE_ID  string  "

isc_ln = spark.read.format("csv"). \
    option("header", False). \
    option("delimiter", ";"). \
    schema(lane_schema). \
    load("file:///home/train/final_project/intlane/*")

isc_ln.limit(5).toPandas()

isc_edge_schema = "  ID integer  ,\
  EDGE_NO  string  , \
  EDGE_ID  integer  ,\
  INTERSECTION_ID  string "

isc_edge = spark.read.format("csv"). \
    options(header=False). \
    schema(isc_edge_schema). \
    option("delimiter", ";"). \
    load("file:///home/train/final_project/intedge_2/*")

isc_edge.limit(5).toPandas()

# Dropping unnecessary columns

isc_edge = isc_edge.drop(*["ID", "INTERSECTION_ID"])

isc_edge.limit(5).toPandas()

# Dropping unnecessary columns
isc_ln = isc_ln.drop(*["ID", "INTERSECTION_ID"])

isc_ln.limit(5).toPandas()

# Dropping unnecessary columns
isc_sn = isc_sn.drop(*["ID", "INTERSECTION_ID", "SENSOR_TYPE", "DIRECTION", "SENSOR_GROUP"])

isc_sn.limit(5).toPandas()
isc_edge.limit(5).toPandas()

# splitting timestamp
split_col = F.split(isc_ln['LANE_ID'], ',')
isc_ln = isc_ln.withColumn('LANE_ID', split_col.getItem(0))

isc_ln.limit(5).toPandas()

# Merging two tables, intersectionlane to intersection edge
df_edge_ln = isc_ln.join(isc_edge, on=["EDGE_ID"], how="inner")

df_edge_ln.limit(20).toPandas()
isc_sn.limit(40).toPandas()
sensor_df.limit(50).toPandas()

from pyspark.sql.functions import when
from pyspark.sql.functions import regexp_replace

# Replacing sensorids = 121,122,1233 to 1201,1202,1203
isc_sn = isc_sn.withColumn('SENSOR_ID',
                           when(isc_sn.SENSOR_ID.endswith("121"), regexp_replace(isc_sn.SENSOR_ID, "121", "1201")) \
                           .when(isc_sn.SENSOR_ID.endswith("122"), regexp_replace(isc_sn.SENSOR_ID, "122", "1202")) \
                           .when(isc_sn.SENSOR_ID.endswith("123"), regexp_replace(isc_sn.SENSOR_ID, "123", "1203")) \
                           .otherwise(isc_sn.SENSOR_ID))

# Changing column type to integer
isc_sn = isc_sn.withColumn("SENSOR_ID", F.col("SENSOR_ID").cast(IntegerType()))

isc_ln.printSchema()
isc_sn.printSchema()

# Join two table intersectionlane_edge to intersectionsensor
df_edge_ln_sn = df_edge_ln.join(isc_sn, on=["LANE_ID"], how="inner")
df_edge_ln_sn.limit(10).toPandas()

# Reducing sensor_id to sensor_df
df_edge_ln_sn = df_edge_ln_sn.where(df_edge_ln_sn.SENSOR_ID > 100)
df_final = sensor_df.join(df_edge_ln_sn, on=["SENSOR_ID"], how="inner")
df_final.limit(5).toPandas()

# Dropping unnecessary columns
df_final = df_final.drop(*["BID", "INTERSECTION_ID", "ID"])
df_final.limit(5).toPandas()

# Shape of data
print((df_final.count(), len(df_final.columns)))

df_final.count()

# writing to local as parquet
df_final.write.format("parquet").save("file:///home/train/final_project/df_sensors")


# adding timeseries columns

# defining udf timeseries function
def split_to_datetime(df, col_name):
    df = df.withColumn("year", F.year(F.col(col_name))) \
        .withColumn("month", F.month(F.col(col_name))) \
        .withColumn("day", F.dayofmonth(F.col(col_name))) \
        .withColumn("dayofyear", F.dayofyear(F.col(col_name))) \
        .withColumn("dayofweek", F.dayofweek(F.col(col_name))) \
        .withColumn("weekofyear", F.weekofyear(F.col(col_name))) \
        .withColumn("hour", F.hour(F.col(col_name))) \
        .withColumn("minute", F.minute(F.col(col_name))) \
        .withColumn("second", F.second(F.col(col_name)))
    return df


# registering udf
spark.udf.register("split_to_datetime", split_to_datetime)

# applying udf
df_final_2 = split_to_datetime(df_final, "INSERT_DT")

df_final_2.limit(5).toPandas()

# rounding hours
df_final_3 = df_final_2.withColumn("hourly_timestamp", F.date_trunc("minute", df_final_2.INSERT_DT))

df_final_3.limit(5).toPandas()

# saving data as parquet with snappy compression
df_final_3.write.format("parquet").option("compression", "snappy").save("file:///home/train/elastic/df_sns")
df_final_3 = spark.read.format("parquet").option("compression", "snappy").load("file:///home/train/elastic/df_sns")
df_final_3.limit(5).toPandas()

isc_ln.limit(5).toPandas()
isc_sn.limit(5).toPandas()
isc_edge.limit(5).toPandas()
sensor_df.limit(5).toPandas()
df_final.limit(5).toPandas()
df_final_3.limit(5).toPandas()
pd.set_option("expand_frame_repr", True)

pd.set_option("max_colwidth", 100)
pd.set_option('display.width', 190)

import warnings

warnings.filterwarnings('ignore')

from elasticsearch import Elasticsearch, helpers
import time

es = Elasticsearch("localhost:9200")
# es.indices.create("sensor_dt")


sensor_index = {
    "settings": {
        "index": {
            "analysis": {
                "analyzer": {
                    "custom_analyzer":
                        {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase", "custom_edge_ngram", "asciifolding"
                            ]
                        }
                },
                "filter": {
                    "custom_edge_ngram": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 10
                    }
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "INSERT_DT": {"type": "date"},
            "COUNT": {"type": "integer"},
            "EDGE_ID": {"type": "integer"},
            "EDGE_NO": {"type": "keyword"},
            "LANE_NO": {"type": "keyword"},
            "SENSOR_ID": {"type": "integer"},
            "day": {"type": "integer"},
            "dayofweek": {"type": "integer"},
            "dayofyear": {"type": "integer"},
            "hour": {"type": "integer"},
            "hourly_timestamp": {"type": "keyword"},
            "minute": {"type": "integer"},
            "month": {"type": "integer"},
            "second": {"type": "integer"},
            "weekofyear": {"type": "integer"},
            "year": {"type": "integer"}
        }
    }
}

try:
    es.indices.delete("sensor_data")
    print("sensor  index deleted.")
except:
    print("No index")

es.indices.create("sensor_data", body=sensor_index)
# write to elasticsearch
df_final_3.write \
    .format("org.elasticsearch.spark.sql") \
    .mode("overwrite") \
    .option("es.nodes", "localhost") \
    .option("es.port", "9200") \
    .save("sensor_data")

# read from elasticsearch
df_final_3 = spark.read \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "localhost") \
    .option("es.port", "9200") \
    .load("sensor_data")
