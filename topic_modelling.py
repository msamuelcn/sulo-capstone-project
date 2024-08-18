import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
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

def identify_topic(weights):
    # Convert weights to string
    weights_str = str(weights)

    # Define the prompt
    prompt = f"""
    I have the following list of terms and their associated weights from a topic modeling analysis:

    {weights_str}

    Please identify the topic these terms are related to and provide a brief description of what the topic might involve.
    """

    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant skilled in interpreting text data."},
            {"role": "user", "content": prompt}
        ]
    )

    # Get the response content
    response_content = response.choices[0].message['content']

    return response_content

def app():

    # App title
    st.title("Topic Modelling")

    # Load the JSON file into a DataFrame
    df = pd.read_json('./final_data.json')

    # Select relevant columns
    df = df[['spacy_lemmatized_tokens','thumbsUpCount']]

    # Filter DataFrame
    df = df[df['thumbsUpCount'] >= 20]

    df['spacy_lemmatized_tokens'] = df['spacy_lemmatized_tokens'].apply(remove_special_characters)

    topic_count = st.number_input('Select number of topics', min_value=2, max_value=30, value=5, step=1)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['spacy_lemmatized_tokens'])

    # Apply NMF
    n_components = topic_count  # Number of topics
    nmf_model = NMF(n_components=n_components, init='random', random_state=2)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    # Convert terms to words
    terms = vectorizer.get_feature_names_out()

    # Set the number of words to display in each word cloud and bar chart
    wordcloud_words = 30
    barchart_words = 8

    top_terms_per_topic = []

    # Generate and display word clouds and bar charts for each topic
    for topic_idx, topic in enumerate(H):
        # Get the top 30 words for the word cloud
        wordcloud_indices = topic.argsort()[-wordcloud_words:]
        word_freq = {terms[i]: topic[i] for i in wordcloud_indices}

        # Create and display the WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        st.subheader(f"Topic {topic_idx + 1}")

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for Topic {topic_idx + 1}")
        plt.show()

        st.pyplot(plt)

        # Get the top 8 words for the bar chart
        barchart_indices = topic.argsort()[-barchart_words:]
        top_terms = [terms[i] for i in barchart_indices]
        top_weights = [topic[i] for i in barchart_indices]

        top_terms_per_topic.append({
            'topic_count': topic_idx + 1,
            'top_terms':top_terms,
            'top_weights': top_weights
        })

        # Sort the terms and weights in descending order
        sorted_terms_weights = sorted(zip(top_weights, top_terms), reverse=True)
        top_weights, top_terms = zip(*sorted_terms_weights)

        # print(sorted_terms_weights)

        # Create and display the bar chart
        plt.figure(figsize=(10, 5))
        plt.barh(top_terms, top_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.title(f"Top {barchart_words} Words in Topic {topic_idx + 1}")
        plt.gca().invert_yaxis()  # Invert y-axis to display the highest weighted word on top
        plt.show()
        st.pyplot(plt)

        what_topic = identify_topic(sorted_terms_weights)

        st.write(what_topic)





