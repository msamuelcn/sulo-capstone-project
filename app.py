import streamlit as st
from analyze_a_review import app as analyze_review
from keyword_search import app as keyword
from most_frequent_n_grams import app as n_grams
from topic_modelling import app as topic
# Sidebar for navigation
st.sidebar.title("Navigation")
pages = {
    "Analyze a review": analyze_review,
    "Keyword search":keyword,
    "Most Frequent N-grams": n_grams,
    "Topic Modelling": topic
}

# Select page
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Run the selected page's app() function
pages[selection]()
