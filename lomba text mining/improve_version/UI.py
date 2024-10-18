import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("Komentar Sentimen")

# Adding some charts
st.subheader("Visualisasi Data Sentimen")

# Example data for charts
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Positif', 'Netral', 'Negatif']
)

# 1. Bar chart
st.write("Distribusi Sentimen (Bar Chart)")
st.bar_chart(chart_data)

# 2. Line chart
st.write("Tren Sentimen dari Waktu ke Waktu (Line Chart)")
st.line_chart(chart_data)

# 3. Area chart
st.write("Distribusi Sentimen dalam Area (Area Chart)")
st.area_chart(chart_data)

# Create input text area for the comment
st.subheader("Komentar Sentimen")
user_input = st.text_area("Masukkan komentar Anda di sini:")

# Placeholder for the sentiment result
if user_input:
    # You can replace this logic with a sentiment analysis model later
    sentiment_result = "sentimen positif"  # Contoh hasil sementara
else:
    sentiment_result = "Tidak ada input"

# Display the sentiment result below the text area
st.write(f"Sentiment: {sentiment_result}")
