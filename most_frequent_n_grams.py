import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import openai
import re
import seaborn as sns

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

    rows_n = st.number_input('How many rows to show?', min_value=2, max_value=100, value=10, step=1)

    if n_gram:
        st.subheader("Words from the keyword(s)")

        def ngram_analysis(df, text_column, n=2):
            # Initialize the CountVectorizer for n-gram analysis
            vectorizer = CountVectorizer(ngram_range=(n, n))

            # Fit the vectorizer to the text data and transform it into a matrix of token counts
            ngram_matrix = vectorizer.fit_transform(df[text_column])

            # Sum the counts of each n-gram and convert them into a DataFrame
            ngram_counts = pd.DataFrame(ngram_matrix.sum(axis=0), columns=vectorizer.get_feature_names_out()).T
            ngram_counts.columns = ['Frequency']

            # Sort the n-grams by frequency in descending order
            ngram_counts = ngram_counts.sort_values(by='Frequency', ascending=False)

            return ngram_counts

        result_n_gram = ngram_analysis(df,'spacy_lemmatized_tokens',n_gram)

        top_ngrams = result_n_gram.head(rows_n)

        # Plot the results
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_ngrams['Frequency'], y=top_ngrams.index, palette='viridis')

        plt.title(f'Top {rows_n} n-grams')
        plt.xlabel('Frequency')
        plt.ylabel('n-gram')
        plt.show()

        st.pyplot(plt)

        st.dataframe(top_ngrams.iloc[:rows_n])

    else:
        st.write("No reviews match the keyword(s).")


