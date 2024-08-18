import streamlit as st
import pandas as pd
import plotly.express as px
import openai

openai.api_key = st.secrets['api_key']


# App title
st.title("GCash App Reviews Sentiment Analysis Dashboard")


    # Load the CSV file into a DataFrame
df = pd.read_json('./16_sentiments.json')

df = df[['content',
    'score',
    'thumbsUpCount',
    'appVersion',
    'spacy_lemmatized_tokens',
    'positives',
    'negatives',
    'neutral',
    'compound']]

df = df[df['thumbsUpCount']>=20]



# Define a function to analyze sentiment using GPT-3.5-turbo
def analyze_sentiment(review_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for analyzing customer reviews."},
            {"role": "user", "content": f"Analyze the sentiment of this review: {review_text}"}
        ]
    )
    sentiment = response.choices[0].message['content'].strip()
    return sentiment

# Define a function to perform topic extraction using GPT-3.5-turbo
def extract_topics(review_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for analyzing customer reviews."},
            {"role": "user", "content": f"Identify the main topics in this review: {review_text}"}
        ]
    )
    topics = response.choices[0].message['content'].strip()
    return topics

# Streamlit app interface
st.title("GCash App Reviews Analysis with GPT-3.5-turbo")
st.write("Upload your GCash app reviews CSV file to analyze sentiment and extract topics using GPT-3.5-turbo.")

# Check if 'content' column exists
if 'content' not in df.columns:
    st.error("CSV file must contain a 'content' column.")
else:
    st.write("Analyzing reviews...")

    # Analyze sentiment and extract topics for each review
    df['Sentiment'] = df['content'].apply(analyze_sentiment)
    df['Topics'] = df['content'].apply(extract_topics)

    # Display the DataFrame with sentiment and topics
    st.write("Analysis complete! Here are the results:")
    st.dataframe(df)

    # Option to download the analysis results
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download analysis as CSV",
        data=csv,
        file_name="gcash_reviews_analysis.csv",
        mime="text/csv"
    )

# Sidebar options for exploration
st.sidebar.title("Explore Reviews")
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])

if sentiment_filter != "All":
    filtered_df = df[df['Sentiment'].str.contains(sentiment_filter, case=False)]
    st.sidebar.write(f"Displaying {len(filtered_df)} reviews with {sentiment_filter} sentiment:")
    st.sidebar.dataframe(filtered_df[['content', 'Sentiment', 'Topics']])

st.sidebar.title("Explore Topics")
keyword = st.sidebar.text_input("Search reviews by topic or keyword")

if keyword:
    keyword_filtered_df = df[df['Topics'].str.contains(keyword, case=False)]
    st.sidebar.write(f"Displaying {len(keyword_filtered_df)} reviews containing '{keyword}':")
    st.sidebar.dataframe(keyword_filtered_df[['content', 'Sentiment', 'Topics']])