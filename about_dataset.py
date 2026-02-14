import streamlit as st
import pandas as pd
from PIL import Image

def about_dataset():
    # Deskripsi dataset
    st.write('**Tentang Dataset**')
    st.write('Dataset Bank Customer Churn berisi informasi profil dan perilaku nasabah bank yang digunakan untuk menganalisis kemungkinan pelanggan berhenti menggunakan layanan (churn). Data ini umumnya mencakup variabel seperti usia, jenis kelamin, '\
             'wilayah, lama menjadi nasabah (tenure), saldo rekening, jumlah produk yang digunakan, status kartu kredit, aktivitas transaksi, serta estimasi pendapatan.')
    
    # Menampilkan Dataset
    st.write('**Data Churn for Bank Customers**')
    df_visualisasi = pd.read_csv('churn(Visualisasi).csv')
    st.dataframe(df_visualisasi.head(10))

    # Dashnboard Overview
    st.write('**1. Dashboard Overview (Desain with Tableau)**')
    st.write('Dashboard ini memberikan gambaran umum kondisi churn nasabah secara menyeluruh dan ditujukan untuk manajemen atau pengambil keputusan.')
    # Memuat gambar dashboard obverview
    image = Image.open("Dashboard_overview.png")
    #Tampilkan gambar
    st.image(image, caption="Dashboard Overview", use_container_width=True)

    # Dashboard Factor Churn
    st.write('**2. Dashboard Factor Churn (Desain withTableau)**')
    st.write('Dashboard ini lebih mendalam dan fokus pada analisis faktor-faktor yang memengaruhi churn.')
    # Memuat gambar dashboard factor churn
    image = Image.open("Dashboard_factor_churn.png")
    #Tampilkan gambar
    st.image(image, caption="Dashboard Factor Churn", use_container_width=True)

    # Link To Tableau Public
    st.write('Link to Tableau Public')
    st.write('🔗 Tableau : [Public.Tableau.com/app/aria.kumara.tungga](https://public.tableau.com/shared/BW7TZGF8T?:display_count=n&:origin=viz_share_link)')