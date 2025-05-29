import pandas as pd
import re
import urllib.request
from functools import reduce
import emoji
from pyspark.sql.functions import length, trim
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, udf, sum as _sum, when, avg, lit, count
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_date


# === 🔹 Start Spark Session
spark = SparkSession.builder.appName("Combined CSV Cleaning and Sentiment").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# === 🔹 Load CSV
df = spark.read.csv(
    "/opt/spark/combineddata_raw.csv/part-00000-3d188904-13b8-46f5-9a7b-0fd395f333d7-c000.csv",
    header=True,
    inferSchema=True
)

# === 🔹 Show original data
df.show(5)



from pyspark.sql.functions import col
from dateutil import parser
from pyspark.sql.types import StringType

# Step 1: Collect distinct non-null date strings from the DataFrame
unique_dates = df.select("Date") \
    .filter(col("Date").isNotNull()) \
    .distinct() \
    .rdd.flatMap(lambda x: x) \
    .collect()

# Step 2: Use dateutil.parser to auto-detect and parse dates
from datetime import datetime

def get_detected_format(date_str):
    try:
        dt = parser.parse(date_str)
        # Check for common output format after parsing
        if date_str.endswith("Z"):
            return "ISO 8601 with Z (UTC): yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        elif "T" in date_str:
            return "ISO 8601: yyyy-MM-dd'T'HH:mm:ss"
        elif "-" in date_str:
            return "Dash-separated: yyyy-MM-dd"
        elif "/" in date_str:
            return "Slash-separated: dd/MM/yyyy or MM/dd/yyyy"
        else:
            return "Other recognizable format"
    except:
        return "Unrecognized format"

# Step 3: Get unique formats
detected_formats = set(get_detected_format(date) for date in unique_dates)

# Step 4: Print results
print(" Unique date formats detected:")
for fmt in detected_formats:
    print(f" • {fmt}")


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from dateutil import parser

# UDF to parse and format date
def normalize_date(date_str):
    try:
        dt = parser.parse(date_str)
        return dt.strftime("%Y-%m-%d")
    except:
        return None  # Invalid format

# Register UDF
normalize_date_udf = udf(normalize_date, StringType())

# Apply the UDF to normalize all date strings
df = df.withColumn("Date", normalize_date_udf(col("Date")))


# === 🔹 Print Null Count Per Column BEFORE Removal
print("\n Null Count Per Column (Before Cleaning):")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# === 🔹 Print Rows with Any Null Value BEFORE Removal
print("\n Rows with Null Values (Before Cleaning):")
df.filter(reduce(lambda x, y: x | y, (col(c).isNull() for c in df.columns))).show()

# === 🔹 Remove rows with any null values
df = df.dropna()

# === 🔹 Print Null Count Per Column AFTER Removal
print("\n Null Count Per Column (After dropna):")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# === 🔹 Count empty strings in 'Comments'
empty_comments_count = df.filter(trim(col("Comments")) == "").count()
print(f" Empty string count in 'Comments': {empty_comments_count}")

# === 🔹 Count empty strings in 'Posts'
empty_posts_count = df.filter(trim(col("Posts")) == "").count()
print(f" Empty string count in 'Posts': {empty_posts_count}")

# === 🔹 Text Cleaning Function
def clean_text(text):
    if text:
        text = emoji.demojize(text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s:]", "", text)
        text = re.sub(r":([a-z_]+):", r"\1", text)

        return text
    return ""

# === 🔹 Register UDF
clean_udf = udf(clean_text, StringType())

# === 🔹 Apply UDF to 'Comments' Column
df_cleaned = df.withColumn("Cleaned_Comment", clean_udf(col("Comments")))

# === 🔹 Show Original and Cleaned Comments
print("\n Original vs Cleaned Text (Structured View):\n")
rows = df_cleaned.select("Comments", "Cleaned_Comment").limit(10).collect()

for i, row in enumerate(rows, 1):
    original = row["Comments"]
    cleaned = row["Cleaned_Comment"]
    print(f"#{i}")
    print(f" Original: {original}")
    print(f" Cleaned : {cleaned}")
    print("-" * 80)
     
# === 🔹 Encode Sentiment Labels
df = df.withColumn("label", 
    when(col("Sentiment") == "neg", 0.0)
    .when(col("Sentiment") == "neu", 1.0)
    .when(col("Sentiment") == "pos", 2.0)
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Common preprocessing stages
tokenizer = Tokenizer(inputCol="Comments", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
word2vec = Word2Vec(inputCol="filtered", outputCol="features", vectorSize=100, minCount=1)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Models to train (Naive Bayes removed)
models = {
    "Logistic Regression": LogisticRegression(maxIter=10),
    "Random Forest": RandomForestClassifier(numTrees=50)
}

# Evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

import shutil
import os

#  Initialize before loop
best_model = None
best_accuracy = 0.0
best_model_name = ""

#  Train & Evaluate Models
for name, classifier in models.items():
    print(f"\n🔹 Training: {name}")
    pipeline = Pipeline(stages=[tokenizer, remover, word2vec, classifier])
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)

    print(f" {name} Results:")
    print(f"   Accuracy : {accuracy:.2f}")
    print(f"   Precision: {precision:.2f}")
    print(f"   Recall   : {recall:.2f}")

    #  Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# #  Save best model
# if best_model is not None:
#     model_path = f"/opt/spark/best_sentiment_model_{best_model_name.replace(' ', '_').lower()}"
#     if os.path.exists(model_path):
#         shutil.rmtree(model_path)
#     best_model.save(model_path)
#     print(f"\n Best model '{best_model_name}' saved with accuracy {best_accuracy:.2f}")
# else:
#     
# print("⚠ No model was trained or evaluated.")
print(f"\n Best model '{best_model_name}' saved with accuracy {best_accuracy:.2f}")

from pyspark.sql.functions import count, col, when

from pyspark.sql.functions import col, when, count

# === 🔹 Step 1: Group by Post_ID and count sentiment types ===
sentiment_counts = df.groupBy("Post_ID").agg(
    count(when(col("Sentiment") == "pos", 1)).alias("pos_count"),
    count(when(col("Sentiment") == "neg", 1)).alias("neg_count"),
    count(when(col("Sentiment") == "neu", 1)).alias("neu_count")
)

# === 🔹 Step 2: Decide Post Sentiment based on Majority ===
final_post_sentiment = sentiment_counts.withColumn(
    "Post_Sentiment",
    when((col("pos_count") > col("neg_count")) & (col("pos_count") > col("neu_count")), "pos")
    .when((col("neg_count") > col("pos_count")) & (col("neg_count") > col("neu_count")), "neg")
    .when((col("neu_count") > col("pos_count")) & (col("neu_count") > col("neg_count")), "neu")
    .otherwise("tie")
)

# === 🔹 Step 3: Join with Original Post Content ===
# Include Post_ID and Posts
unique_posts = df.select("Post_ID", "Posts").dropDuplicates(["Post_ID"])

# Join with sentiment results
final_result = unique_posts.join(
    final_post_sentiment.select("Post_ID", "Post_Sentiment"),
    on="Post_ID",
    how="inner"
)

# === 🔹 Step 4: Display Final Output ===
final_result.orderBy("Post_ID").show(10, truncate=False)


# # Optionally collect to driver and print cleanly
# results = final_result.select("Post_ID", "Posts", "Post_Sentiment").collect()
# Collect top 10 rows ordered by Post_ID
top10 = final_result.orderBy("Post_ID").limit(10).collect()

# Print in vertical format
for row in top10:
    print(f"Post ID   : {row['Post_ID']}")
    print(f"Post      : {row['Posts']}")
    print(f"Sentiment : {row['Post_Sentiment']}")
    print("-"*40)


    # === 🔹 Save final cleaned post sentiment result as CSV ===
final_result.select("Post_ID", "Posts", "Post_Sentiment") \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("/opt/spark/cleaned_post_sentiment")

# === 🔹 Optionally save full cleaned comments dataset as well ===
df_cleaned.write.option("header", True).mode("overwrite").csv("/opt/spark/cleaned_comments_full")
print(" Cleaned datasets saved successfully.")
