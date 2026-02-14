import streamlit as st

st.header('Customer Churn Analytics for Banking Industry')
st.write('Dalam menghadapi persaingan industri perbankan yang semakin ketat, kemampuan untuk mempertahankan nasabah menjadi faktor penting dalam menjaga stabilitas dan pertumbuhan bisnis. '\
         'Oleh karena itu, diperlukan pendekatan berbasis data untuk mengidentifikasi nasabah yang berpotensi churn sejak dini.')

# Membuat tab navigasi
tab1, tab2, tab3, tab4 = st.tabs(['About Dataset', 'Machine Learning', 'Prediction Customer', 'Contact Me'])

# Mengakses masing-masing halaman
with tab1:
    import about_dataset
    about_dataset.about_dataset()
with tab2:
    import ml_churn_bank
    ml_churn_bank.ml_model()
with tab3:
    import prediction_churn
    prediction_churn.prediction_app()
with tab4:
    import kontak
    kontak.contact_me()
