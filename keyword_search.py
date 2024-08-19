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

def business_action(texts):
    # Convert weights to string

    selected_texts = str(texts)

    # Define the prompt
    prompt = f"""
    I have the following list of reviews from the app and highlights the keyword '{selected_texts}':

    {selected_texts}

    Please identify its actionable insight and make a business decisions for this.
    """

    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a business analyst or manager skilled in interpreting text data."},
            {"role": "user", "content": prompt}
        ]
    )

    # Get the response content
    response_content = response.choices[0].message['content']

    return response_content

def app():

    # App title
    st.title("Keyword Search Reviews")

    # Load the JSON file into a DataFrame
    df = pd.read_json('./final_data.json')

    # Select relevant columns
    df = df[['content',
            'at',
            'score',
            'thumbsUpCount',
            'appVersion','spacy_lemmatized_tokens','knn_sentiment']].sort_values('at',ascending=False)

    # Filter DataFrame
    df = df[df['thumbsUpCount'] >= 20]

    df['at'] = pd.to_datetime(df['at'], unit='ms')

    df['spacy_lemmatized_tokens'] = df['spacy_lemmatized_tokens'].apply(remove_special_characters)

    keywords = st.text_input(
        "Enter a keywords and make analysis",
        placeholder='Enter a keyword(s)',
    )

    filtered_result_df = df[df['spacy_lemmatized_tokens'].str.contains(remove_special_characters(keywords))]

    if(keywords):
        st.subheader("Word Cloud of Filtered Reviews")

        all_words = ' '.join(filtered_result_df['spacy_lemmatized_tokens'].tolist())

        # Generate the word cloud with a maximum of 30 words
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=30).generate(all_words)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Time Series Plot based on keyword
    if(keywords):
        st.subheader("Time Series Plot of Review Volume")

        # Extract year and month from the 'at' column
        filtered_result_df['year_month'] = filtered_result_df['at'].dt.to_period('M')

        # Group by year_month and count the number of reviews
        review_counts = filtered_result_df.groupby('year_month').size().reset_index(name='count')

        # Convert 'year_month' to a shorter format
        review_counts['year_month'] = review_counts['year_month'].dt.strftime('%Y-%m')

        # Plotting the time series
        plt.figure(figsize=(10, 5))
        plt.plot(review_counts['year_month'], review_counts['count'], marker='o')

        # Adjusting x-axis labels to avoid overlap
        plt.xticks(rotation=45, ha='right')  # 'ha' stands for horizontal alignment
        plt.title('Volume of Reviews Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Reviews')
        plt.grid(True)
        st.pyplot(plt)

    if(keywords):
        st.subheader("Words associated with keyword")

        n_gram = st.number_input('Select n for ngrams', min_value=2, max_value=5, value=2, step=1)

        # Combine all tokens into a single list
        all_tokens = filtered_result_df['spacy_lemmatized_tokens'].tolist()

        # Generate bigrams
        vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram))  # bigrams only
        X = vectorizer.fit_transform(all_tokens)

        # Get the bigram counts
        bigram_counts = X.sum(axis=0)
        bigram_frequencies = [(word, bigram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        bigram_frequencies = sorted(bigram_frequencies, key=lambda x: x[1], reverse=True)

        # Convert to DataFrame for easy plotting
        bigram_df = pd.DataFrame(bigram_frequencies, columns=['bigram', 'count'])

        bigram_df = bigram_df[bigram_df['bigram'].str.contains(keywords)]

        # Plot the top 10 most common bigrams
        top_bigrams = bigram_df.head(10)
        plt.figure(figsize=(10, 8))
        plt.barh(top_bigrams['bigram'], top_bigrams['count'], color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.title('Most Frequent Associated word')
        plt.gca().invert_yaxis()  # Highest values on top
        st.pyplot(plt)
    if(keywords):
        st.subheader("Sentiment distribution")
        df_grouped = filtered_result_df.groupby('knn_sentiment').size()

        df_grouped = pd.DataFrame(df_grouped).reset_index().rename(columns={'knn_sentiment':'sentiment',0:'frequency'})

        # Calculate percentages
        df_grouped['percentage'] = (df_grouped['frequency'] / df_grouped['frequency'].sum()) * 100

        colors = ['red', 'blue','green']

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df_grouped['sentiment'], df_grouped['percentage'], color=colors)

        # Adding labels
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.title('Sentiment Percentages')
        plt.ylim(0, 100)  # Ensure y-axis ranges from 0 to 100%
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adding percentage labels on bars
        for i, value in enumerate(df_grouped['percentage']):
            plt.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom')

        plt.show()
        st.pyplot(plt)
    if keywords:
        st.subheader("Business decisions")
        result = business_action(filtered_result_df['content'].iloc[:20])
        st.write(result)
    else:
        st.write("No reviews match the keyword(s).")


