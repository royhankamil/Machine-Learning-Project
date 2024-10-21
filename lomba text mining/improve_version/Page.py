import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud  # Perbaikan impor yang benar
import matplotlib.pyplot as plt
import pickle
from Algorithm import KNN

with open('model/knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/word_set.pkl', 'rb') as file:
    word_set = pickle.load(file)

with open('model/index_dict.pkl', 'rb') as file:
    index_dict = pickle.load(file)

with open('model/word_count.pkl', 'rb') as file:
    word_count = pickle.load(file)

def predict_sentiment(input_text):
    # Preprocess dan transformasi input menjadi TF-IDF
    tfidf_vector = transform_to_tfidf(input_text)
    prediction = model.predict([tfidf_vector])
    return prediction

# Membaca data CSV
all_wordo = pd.read_csv("all_word_data.csv")

# Fungsi untuk menampilkan WordCloud menggunakan Streamlit
def showing(wc):
    fig, ax = plt.subplots(figsize=(12, 8))  # Membuat figure dan axes
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")  # Menghilangkan axis
    st.pyplot(fig)  # Menampilkan figure di Streamlit

# Set the title of the app
st.title("Komentar Sentimen")

# Adding some charts
st.subheader("Visualisasi Data Sentimen")

# Membuat dan menampilkan WordCloud untuk kolom 'Ras'
st.write("Distribusi Sentimen (WordCloud - Ras)")
word_cloud = WordCloud(
    height=800,
    width=1200,
    collocations=False,
    colormap="Blues_r"
).generate(' '.join(all_wordo["Ras"].dropna().to_list()))
showing(word_cloud)

# Membuat dan menampilkan WordCloud untuk kolom 'Agama'
st.write("Distribusi Sentimen (WordCloud - Agama)")
word_cloud1 = WordCloud(
    height=800,
    width=1200,
    collocations=False,
    colormap="Blues_r"
).generate(' '.join(all_wordo["Agama"].dropna().to_list()))
showing(word_cloud1)

# Membuat dan menampilkan WordCloud untuk kolom 'Netral'
st.write("Distribusi Sentimen (WordCloud - Netral)")
word_cloud2 = WordCloud(
    height=800,
    width=1200,
    collocations=False,
    colormap="Blues_r"
).generate(' '.join(all_wordo["Netral"].dropna().to_list()))
showing(word_cloud2)

# Create input text area for the comment
st.subheader("Komentar Sentimen")
user_input = st.text_area("Masukkan komentar Anda di sini:")

# Button for prediction
if st.button("Prediksi"):
    # Placeholder for the sentiment result
    if user_input:
        # You can replace this logic with a sentiment analysis model later
        sentiment_result = predict_sentiment(user_input)
    else:
        sentiment_result = "Tidak ada input"

    # Display the sentiment result below the text area
    st.write(f"Sentiment: {sentiment_result}")
