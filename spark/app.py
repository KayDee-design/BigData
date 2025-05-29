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
    files = glob.glob("/opt/spark/cleaned_comments_full/part-00001-bdbd0bba-db8c-4898-b79c-51cb3ebf7c4b-c000.csv")
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

# === 🔹 Sidebar Filters ===
st.sidebar.title("🔎 Filters")
sentiments = sorted(df["Sentiment"].dropna().unique()) if "Sentiment" in df.columns else []
selected_sentiment = st.sidebar.selectbox("Sentiment", ["All"] + sentiments)

# content_types = sorted(df["content_type"].dropna().unique()) if "content_type" in df.columns else []
# selected_content = st.sidebar.selectbox("Content Type", ["All"] + content_types)

if selected_sentiment != "All":
    df = df[df["Sentiment"] == selected_sentiment]

# if selected_content != "All":
#     df = df[df["content_type"] == selected_content]

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

# === 🔹 Time Series ===
if "Date" in df.columns and "Sentiment" in df.columns:
    st.subheader("📅 Sentiment Over Time")
    time_sentiment = df.groupby(["Date", "Sentiment"]).size().unstack(fill_value=0)
    st.line_chart(time_sentiment)

# # === 🔹 Content Type Pie Chart ===
# if "content_type" in df.columns:
#     st.subheader("📰 Content Type Distribution")
#     fig2, ax2 = plt.subplots()
#     df["content_type"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax2)
#     ax2.set_ylabel("")
#     st.pyplot(fig2)

# # === 🔹 Sentiment by Content Type (Grouped Bar) ===
# if "content_type" in df.columns and "Sentiment" in df.columns:
#     st.subheader("💬 Sentiment by Content Type")
#     grouped = df.groupby(["content_type", "Sentiment"]).size().unstack(fill_value=0)
#     st.bar_chart(grouped)

# === 🔹 Top Posts by Comment Volume ===
st.subheader("🔥 Top Posts by Comment Volume")
top_posts = df["Post_ID"].value_counts().head(10)
st.bar_chart(top_posts)

# === 🔹 Top 5 Dates with Most Comments ===
if "Date" in df.columns:
    st.subheader("📆 Top 5 Days with Most Comments")
    top_dates = df["Date"].value_counts().head(5).sort_index()
    st.bar_chart(top_dates)

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

# === 🔹 Comment Length by Sentiment ===
if "comment_length" in df.columns and "Sentiment" in df.columns:
    st.subheader("📏 Comment Length by Sentiment")
    fig_len, ax_len = plt.subplots()
    df.boxplot(column="comment_length", by="Sentiment", ax=ax_len)
    plt.suptitle("")
    ax_len.set_title("Comment Length per Sentiment")
    st.pyplot(fig_len)

# === 🔹 Top 10 Comments ===
st.subheader("📝 Top 10 Comments")
st.dataframe(df[["Post_ID", "Comments", "Sentiment"]].head(10), use_container_width=True)

# === 🔹 Keyword Search ===
st.subheader("🔍 Search Comments")
search_term = st.text_input("Enter keyword to search in comments")
if search_term:
    matches = df[df["Comments"].str.contains(search_term, case=False, na=False)]
    st.write(f"🔎 {len(matches)} results found for `{search_term}`")
    st.dataframe(matches[["Post_ID", "Comments", "Sentiment"]])

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



