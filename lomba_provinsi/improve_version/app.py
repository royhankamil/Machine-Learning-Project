import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from algorithm.Algorithm import KNN
from algorithm.Algorithm import Preprocessing
import os
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load pre-trained model and supporting files
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model/knn_model.pkl')
word_set_path = os.path.join(BASE_DIR, 'model/word_set.pkl')
indexdict_path = os.path.join(BASE_DIR, 'model/index_dict.pkl')
word_count_path = os.path.join(BASE_DIR, 'model/word_count.pkl')
preprocessing_path = os.path.join(BASE_DIR, 'model/preprocessing.pkl')
alldata_path = os.path.join(BASE_DIR, 'all_word_data.csv')

model = load_pickle(model_path)
word_set = load_pickle(word_set_path)
index_dict = load_pickle(indexdict_path)
word_count = load_pickle(word_count_path)
preprocessing = load_pickle(preprocessing_path)

# Function to predict sentiment based on input text
def predict_sentiment(input_text):
    # Preprocess and transform input to TF-IDF
    tfidf_vector = preprocessing.transform_to_tfidf(input_text)
    prediction = model.predict([tfidf_vector])
    return prediction

# Function to display WordCloud using Streamlit
def display_wordcloud(column_data, title, colormap="Blues_r"):
    st.write(title)
    wordcloud = WordCloud(
        height=800,
        width=1200,
        collocations=False,
        colormap=colormap,
        random_state=123
    ).generate(' '.join(column_data.dropna().to_list()))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Title and subheader
st.title("Pendeteksi Komentar Negatif")

st.subheader("Komentar Sentimen")
# Input area for user comment
user_input = st.text_area("Masukkan komentar Anda di sini:")

# Prediction button and result
if st.button("Prediksi"):
    if user_input:
        sentiment_code = predict_sentiment(user_input)
        sentiment_result = "Netral" if sentiment_code == 0 else "Ras" if sentiment_code == 1 else "Agama"
        st.write(f"Sentimen: {sentiment_result}")
    else:
        st.write("Tidak ada input")
st.subheader("Visualisasi Data Sentimen")

# Load data for WordCloud visualization
all_wordo = pd.read_csv(alldata_path)

# Display WordClouds for different categories
display_wordcloud(all_wordo["Ras"], "Distribusi Sentimen (WordCloud - Ras)")
display_wordcloud(all_wordo["Agama"], "Distribusi Sentimen (WordCloud - Agama)")
display_wordcloud(all_wordo["Netral"], "Distribusi Sentimen (WordCloud - Netral)")

