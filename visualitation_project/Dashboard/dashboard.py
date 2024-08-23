# import library
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# pengambilan data
df = pd.read_csv("day.csv")

# pengubahan format
df["dteday"] = pd.to_datetime(df['dteday'])

# Menentukan kolor paletnya
dark_palette = sns.color_palette(["#000", "#f71707", "#8e1ef7", "#0008ff", "#22e322"])

st.title('Analisis Data Rental Sepeda')
tab1, tab2 = st.tabs(["Pertanyaan 1", "Pertanyaan 2"])

with tab1:
    st.header("Pertanyaan 1: ")
    st.write("Bagaimana tren peningkatan rental sepeda di tahun 2011?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualisasi Grafik 1: Tren Rental Sepeda Setiap Bulan di Tahun 2011
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.lineplot(x=df["mnth"], y=df[df["dteday"].dt.year == 2011]["cnt"], marker='o', ax=ax1)
        ax1.set_title("Tren Rental Sepeda Setiap Bulan di Tahun 2011")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Total Rentals")
        ax1.set_xticks(range(1, 12 + 1))
        ax1.grid(True)
        st.pyplot(fig1)
    
    with col2:
        # Visualisasi Grafik 2: Tren Rental Sepeda Setiap Minggu di Tahun 2011
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.barplot(x=df["weekday"], y=df[df["dteday"].dt.year == 2011]["cnt"], ax=ax2)
        ax2.set_title("Tren Rental Sepeda Setiap Minggu di Tahun 2011")
        ax2.set_xlabel("Weekday")
        ax2.set_ylabel("Total Rentals")
        ax2.grid(True)
        st.pyplot(fig2)
    st.write("Trend pada **bulan** dengan tingkat **rental sepeda terbanyak** yaitu pada bulan **Juni**. Untuk **Hari** dengan **tingkat rental terbanyak** yaitu di hari **Sabtu**. Dalam hal ini dapat disimpulkan keterpengaruhan hari libur dengan banyaknya perental sepeda")

with tab2:
    st.header("Pertanyaan 2: ")
    
    st.write("Apa dampak dari hawa / suasana dengan banyaknya penyewa sepeda?")
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualisasi Grafik 3: Pengaruh Suasana dengan Banyaknya Penyewa Sepeda
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x='temp', y='cnt', hue='weathersit', palette=dark_palette, alpha=0.7, ax=ax3)
        ax3.set_title("Pengaruh Suasana dengan Banyaknya Penyewa Sepeda")
        ax3.set_xlabel("Normalized Temperature")
        ax3.set_ylabel("Total Rentals")
        ax3.grid(True)
        st.pyplot(fig3)
    
    with col2:
        # Visualisasi Grafik 4: Korelasi Antar Variabel
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[["cnt", "weathersit", "temp"]].corr(), annot=True, ax=ax4)
        ax4.set_title("Korelasi antara Suasana dengan Total Penyewa")
        st.pyplot(fig4)
    st.write("Kita dapat melihat kalau **hawa / suasana** dapat **berdampak** dengan hasil **banyaknya penyewa sepeda**. Jika hasil **temperatur tinggi** akan menghasilkan jauh **lebih sedikit** **penyewa** daripada ketika **temperaturnya tinggi**. **Penyewa terbanyak** ketika temperatur nya menengah (0.6-0.7) atau pada **tingkatan hangat**. Hal ini juga berlaku dengan cuaca, jika **cuacanya cerah** maka tingkat **penyewa jauh lebih tinggi.")
