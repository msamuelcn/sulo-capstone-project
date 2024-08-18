import streamlit as st
import pandas as pd
import openai

openai.api_key = st.secrets['api_key']

def app():

    # App title
    st.title("Analyse GCash App Reviews")

    # Load the JSON file into a DataFrame
    df = pd.read_json('./final_data.json')

    # Select relevant columns
    df = df[['content',
            'at',
            'score',
            'thumbsUpCount',
            'appVersion']].sort_values('at',ascending=False)

    # Filter DataFrame
    df = df[df['thumbsUpCount'] >= 20]

    df['at'] =pd.to_datetime(df['at'], unit='ms')

    # Define the number of rows per page
    rows_per_page = 50

    # Calculate total number of pages
    total_pages = (len(df) - 1) // rows_per_page + 1

    # Initialize session state for page number if it doesn't exist
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1

    # Sidebar for selecting page number
    # st.sidebar.title("Navigation")
    page_number = st.number_input('Page number', min_value=1, max_value=total_pages, value=st.session_state.page_number, step=1)

    # Update the session state with the new page number
    st.session_state.page_number = page_number

    # Calculate the start and end indices for the current page
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    # Display the table with the selected rows
    st.write(f"Displaying page {page_number} of {total_pages}")
    st.dataframe(df.iloc[start_idx:end_idx])

    # Optional: Display page navigation at the bottom
    col1, col2, col3 = st.columns([1, 2, 1])

    # Previous button
    if col1.button("Previous") and page_number > 1:
        st.session_state.page_number -= 1

    # Display current page number
    col2.write(f"Page {page_number} of {total_pages}")

    # Next button
    if col3.button("Next") and page_number < total_pages:
        st.session_state.page_number += 1

    def display_star_rating(star_count):
        full_star = ":star:"
        empty_star = "☆"
        return full_star * star_count + empty_star * (5 - star_count)

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

    # Dropdown selection based on the given table for comprehensive analysis
    if not df.iloc[start_idx:end_idx].empty:
        selected_index = st.selectbox("Select an index from the table for comprehensive analysis", df.iloc[start_idx:end_idx].index,
                                        index=None,
                                        placeholder="Select the number from the name...",
                                    )

        if(selected_index):
            # Display the selected row
            st.write(f"You selected index: {selected_index}")
            get_data = df.loc[selected_index]

            st.subheader("Content")
            st.write('App version: ' + str(get_data['appVersion']))
            st.write('Date posted: '+str(get_data['at']))
            st.write('Stars: '+display_star_rating(get_data['score']))
            st.write(str(get_data['content']))
            st.write(':thumbsup:' + str(get_data['thumbsUpCount']))

            sentiment_explanation = analyze_sentiment(str(get_data['content']))
            st.write(sentiment_explanation)


    else:
        st.write("No data available to select from.")