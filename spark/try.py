import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob
from wordcloud import WordCloud
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
import re
import emoji

# === 🔹 Page Config ===
st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")
st.title("📊 Social Media Sentiment Dashboard")

# === 🔹 Spark Session and Model ===
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("SentimentDashboard").getOrCreate()

@st.cache_resource
def load_model():
    return PipelineModel.load("/opt/spark/best_sentiment_model_random_forest")

spark = get_spark()
model = load_model()

# === 🔹 Load Data ===
@st.cache_data
def load_data():
    files = glob.glob("/opt/spark/cleaned_comments_full/part-*.csv")
    if not files:
        return pd.DataFrame()
    df = pd.concat(
        (pd.read_csv(f, quotechar='"', escapechar='\\', engine='python', on_bad_lines='skip') for f in files),
        ignore_index=True
    )
    return df

df = load_data()

# === 🔹 Validate & Preprocess ===
if df.empty:
    st.warning("⚠ No data found. Please check the path or regenerate the CSV from Spark.")
    st.stop()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

if "Comments" in df.columns:
    df["comment_length"] = df["Comments"].astype(str).apply(len)

# === 🔹 Summary Metrics ===
st.subheader("📌 Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Comments", len(df))
col2.metric("Unique Posts", df["Post_ID"].nunique())
col3.metric("Date Range", f"{df['Date'].min().date()} - {df['Date'].max().date()}")
col4.metric("Avg. Comment Length", f"{df['comment_length'].mean():.1f} chars")

# === 🔹 Sidebar Filters ===
st.sidebar.title("🔎 Filters")
sentiments = sorted(df["Sentiment"].dropna().unique()) if "Sentiment" in df.columns else []
selected_sentiment = st.sidebar.selectbox("Sentiment", ["All"] + sentiments)

if selected_sentiment != "All":
    df = df[df["Sentiment"] == selected_sentiment]

# === 🔹 Overview Stats ===
st.subheader("📈 Comment Sentiment Distribution")
if "Sentiment" in df.columns:
    st.bar_chart(df["Sentiment"].value_counts())

if "Post_Sentiment" in df.columns:
    st.subheader("🧠 Post Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    df["Post_Sentiment"].value_counts().plot(kind="bar", ax=ax1, color=["green", "red", "gray", "blue"])
    ax1.set_title("Post-Level Sentiment")
    st.pyplot(fig1)



# === 🔹 Top Posts by Comment Volume ===
st.subheader("🔥 Top Posts by Comment Volume")
top_posts = df["Post_ID"].value_counts().head(10)
st.bar_chart(top_posts)

# === 🔹 Word Cloud ===
if "Cleaned_Comment" in df.columns and "Sentiment" in df.columns:
    st.subheader("☁️ Word Cloud by Sentiment")
    wc_sentiment = st.selectbox("Select Sentiment for Word Cloud", df["Sentiment"].dropna().unique())
    cloud_text = " ".join(df[df["Sentiment"] == wc_sentiment]["Cleaned_Comment"].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(cloud_text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

# === 🔹 Top 5 Days with Most Comments ===
if "Date" in df.columns:
    st.subheader(" Top 5 Days with Most Comments")
    top_dates = df["Date"].value_counts().nlargest(5).sort_index()
    st.bar_chart(top_dates)

   

# === 🔹 Prescriptive Insights ===
st.subheader(" Prescriptive Insights")
most_neg_date = df[df['Sentiment'] == "neg"]["Date"].value_counts().idxmax()
top_post = df["Post_ID"].value_counts().idxmax()
top_words = df[df['Sentiment'] == "pos"]["Cleaned_Comment"].str.cat(sep=" ").split()


# === 🔹 Top 10 Comments ===
st.subheader("📝 Top 10 Comments")
st.dataframe(df[["Post_ID", "Comments", "Sentiment"]].head(10), use_container_width=True)

# === 🔹 Keyword Search ===
st.subheader("🔍 Search Comments")
search_term = st.text_input("Enter keyword to search in comments")
if search_term:
    matches = df[df["Comments"].str.contains(search_term, case=False, na=False)]
    st.write(f"🔎 {len(matches)} results found for `{search_term}`")
    st.dataframe(matches[["Posts", "Post_ID", "Comments", "Sentiment"]])

# === 🔹 Download Filtered Data ===
st.subheader("📤 Download Filtered Comments")
csv = df.to_csv(index=False)
st.download_button("Download as CSV", csv, "filtered_comments.csv", "text/csv")

# === 🔹 Live Sentiment Prediction ===
st.markdown("---")
st.subheader("🧠 Predict Sentiment for Your Comment")

def clean_text(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s:]", "", text)
    text = re.sub(r":([a-z_]+):", r"\1", text)
    return text

user_input = st.text_area("Enter a comment for sentiment prediction", "")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a comment.")
    else:
        cleaned_input = clean_text(user_input)
        input_df = spark.createDataFrame([(cleaned_input,)], ["Comments"])
        prediction = model.transform(input_df).select("prediction").collect()[0][0]
        label_map = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
        st.success(f"🧾 Predicted Sentiment: **{label_map.get(prediction, 'Unknown')}**")

# === 🔚 Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Spark, Streamlit, and Python · © Sonam Choden")
