import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import openai
import re

openai.api_key = st.secrets['api_key']

# Function to remove special characters
def remove_special_characters(text):
    # Use re.sub() to replace any character that is not alphanumeric with an empty string
    cleaned_text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return cleaned_text

def app():

    # App title
    st.title("Top N-gram Keywords with Sentiment Analysis")

    # Load the JSON file into a DataFrame
    df = pd.read_json('./final_data.json')

    # Select relevant columns
    df = df[['spacy_lemmatized_tokens', 'thumbsUpCount', 'knn_sentiment']]

    # Filter DataFrame
    df = df[df['thumbsUpCount'] >= 20]

    df['spacy_lemmatized_tokens'] = df['spacy_lemmatized_tokens'].apply(remove_special_characters)

    n_gram = st.number_input('Select n for ngrams', min_value=2, max_value=5, value=2, step=1)

    if n_gram:
        st.subheader("Words from the keyword(s)")

        # Combine all tokens into a single list
        all_tokens = df['spacy_lemmatized_tokens'].tolist()

        # Generate ngrams
        vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram))  # ngrams only
        X = vectorizer.fit_transform(all_tokens)

        # Get the ngram counts
        ngram_counts = X.sum(axis=0)
        ngram_frequencies = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        ngram_frequencies = sorted(ngram_frequencies, key=lambda x: x[1], reverse=True)

        # Convert to DataFrame for easy plotting
        ngram_df = pd.DataFrame(ngram_frequencies, columns=['ngram', 'count'])

        # Merge the ngram counts with the original DataFrame
        df['ngram'] = df['spacy_lemmatized_tokens'].apply(lambda x: [ng for ng in ngram_df['ngram'] if ng in x])
        df = df.explode('ngram').dropna(subset=['ngram'])

        # Group by ngram and sentiment
        sentiment_counts = df.groupby(['ngram', 'knn_sentiment']).size().unstack(fill_value=0)

        # Plot the top 20 most common ngrams with sentiment stacked
        top_ngrams = ngram_df.head(20)
        top_sentiment_counts = sentiment_counts.loc[top_ngrams['ngram']]

        top_sentiment_counts.plot(kind='barh', stacked=True, figsize=(10, 8), color=['red', 'blue', 'green'])
        plt.xlabel('Frequency')
        plt.ylabel('N-grams')
        plt.title('Top 20 Most Frequent N-grams with Sentiment Analysis')
        plt.gca().invert_yaxis()  # Highest values on top
        st.pyplot(plt)

        top_sentiment_counts['frequency'] = top_sentiment_counts['negative'] + top_sentiment_counts['neutral'] + top_sentiment_counts['positive']
        top_sentiment_counts = top_sentiment_counts.reset_index()
        top_sentiment_counts = top_sentiment_counts[['ngram','frequency','negative','neutral','positive']]
        top_sentiment_counts.rename(columns={'ngram': 'word'})

        # Display the n-gram frequency DataFrame
        st.dataframe(top_sentiment_counts.iloc[:20]  )
    else:
        st.write("No reviews match the keyword(s).")


