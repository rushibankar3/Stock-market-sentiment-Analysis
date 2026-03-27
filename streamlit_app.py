import re
from functools import lru_cache

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Sentiment Predictor", page_icon="📈", layout="wide")


@st.cache_data
@lru_cache(maxsize=1)
def load_sample_tweets(path: str = "stock_tweets.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in ["Date", "Stock Name", "Tweet"] if c in df.columns]
    return df[cols].copy() if cols else df.copy()


def ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


@st.cache_resource
@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    ensure_vader()
    return SentimentIntensityAnalyzer()


def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def predict_sentiment(text: str) -> dict:
    analyzer = get_analyzer()
    scores = analyzer.polarity_scores(clean_tweet(text))
    scores["label"] = label_from_compound(scores["compound"])
    return scores


st.title("📈 Stock Tweet Sentiment Predictor")
st.caption("Predict sentiment for a tweet and explore aggregate sentiment by stock.")

left, right = st.columns([1.6, 1])

with left:
    input_text = st.text_area(
        "Enter tweet text",
        value="Tesla stock looks strong this quarter and guidance is impressive.",
        height=140,
    )

    if st.button("Predict Sentiment", type="primary"):
        if not input_text.strip():
            st.warning("Please enter tweet text first.")
        else:
            result = predict_sentiment(input_text)
            st.subheader(f"Prediction: {result['label']}")
            st.metric("Compound Score", f"{result['compound']:.3f}")

            score_df = pd.DataFrame(
                {
                    "component": ["positive", "neutral", "negative"],
                    "score": [result["pos"], result["neu"], result["neg"]],
                }
            )
            fig = px.bar(
                score_df,
                x="component",
                y="score",
                title="Sentiment Components",
                color="component",
                color_discrete_sequence=["#16a34a", "#64748b", "#dc2626"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Quick Rules")
    st.markdown(
        """
- `compound >= 0.05` → Positive
- `compound <= -0.05` → Negative
- otherwise → Neutral
        """
    )

    st.subheader("Tips")
    st.markdown(
        """
- Enter full tweet context for better polarity.
- Emojis and negations can shift sentiment strongly.
- This is lexicon-based sentiment, not price prediction.
        """
    )

st.divider()
st.subheader("Dataset Explorer")

try:
    tweets_df = load_sample_tweets()
    if "Stock Name" in tweets_df.columns and "Tweet" in tweets_df.columns:
        stock_names = sorted(tweets_df["Stock Name"].dropna().unique().tolist())
        selected_stock = st.selectbox("Filter by stock", ["All"] + stock_names)
        preview_df = tweets_df if selected_stock == "All" else tweets_df[tweets_df["Stock Name"] == selected_stock]

        if not preview_df.empty:
            sample_n = st.slider("Rows to preview", min_value=5, max_value=50, value=10)
            st.dataframe(preview_df.head(sample_n), use_container_width=True)

            sample_texts = preview_df["Tweet"].dropna().astype(str).head(200)
            if not sample_texts.empty:
                preds = sample_texts.apply(predict_sentiment).apply(pd.Series)
                counts = preds["label"].value_counts().rename_axis("label").reset_index(name="count")
                pie = px.pie(counts, names="label", values="count", title="Sample Sentiment Distribution")
                st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("Expected columns (`Stock Name`, `Tweet`) not found in CSV.")
except FileNotFoundError:
    st.info("`stock_tweets.csv` not found. Place the dataset in the same directory as this app.")
