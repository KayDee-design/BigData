from pyspark.sql import SparkSession

# === 🔹 Start Spark Session ===
spark = SparkSession.builder.appName("Combine Posts and Comments").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# === 🔹 Load CSVs ===
comments_df = spark.read.csv("hdfs://sentiment-namenode-1:8020/input/comments.csv", header=True, inferSchema=True)
posts_df = spark.read.csv("hdfs://sentiment-namenode-1:8020/input/posts.csv", header=True, inferSchema=True)

# === 🔹 Rename Columns for Clarity ===
comments_df = comments_df.withColumnRenamed("Comment", "Comments")
posts_df = posts_df.withColumnRenamed("Content", "Posts")

# === 🔹 Join on Post_ID ===
combined_df = comments_df.join(posts_df, on="Post_ID", how="inner")

# === 🔹 Select Only Required Columns ===
final_df = combined_df.select("Post_ID", "Posts", "Date", "Comments", "Sentiment")

# === 🔹 Show Preview with Truncation
print("\n Combined Preview (Truncated):")
final_df.show(5)

# === 🔹 Save Combined Raw Data
# final_df.coalesce(1).write.csv("/opt/spark/combineddata_raw.csv", header=True, mode="overwrite")

print("\n Combined raw data saved to: /opt/spark/combineddata_raw.csv")
