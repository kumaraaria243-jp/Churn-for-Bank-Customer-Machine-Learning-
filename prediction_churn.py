import streamlit as st
import pandas as pd
import joblib

def prediction_app():
    st.subheader("Prediksi Status Nasabah (Loyal / Churn)")
    st.write("Masukkan data nasabah untuk melihat apakah nasabah akan tetap setia atau berpotensi berhenti menggunakan layanan bank.")

    # =============================
    # LOAD MODEL, FITUR, DAN SCALER
    # =============================
    try:
        model = joblib.load("model_random_forest_churn.pkl")
        fitur_model = joblib.load("fitur_model.pkl")
        scalers = joblib.load("scalers_minmax.pkl")
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return

    # =========================
    # AMBIL BATAS MIN-MAX DARI SCALER
    # =========================
    limits = {}
    for col, scaler in scalers.items():
        try:
            limits[col] = {
                "min": float(scaler.data_min_[0]),
                "max": float(scaler.data_max_[0])
            }
        except:
            pass

    st.write("**📝 Data Nasabah**")

    # =========================
    # INPUT ANGKA (INTEGER)
    # =========================
    def int_input(label, col_name):
        return st.number_input(
            label,
            min_value=int(limits[col_name]["min"]),
            max_value=int(limits[col_name]["max"]),
            value=None,
            step=1,
            format="%d",
            help=f"Batas data training: {int(limits[col_name]['min'])} - {int(limits[col_name]['max'])}"
        )

    credit_score = int_input("Skor Kredit", "CreditScore")
    age = int_input("Umur Nasabah", "Age")
    tenure = int_input("Lama Menjadi Nasabah (Tahun)", "Tenure")
    num_products = int_input("Jumlah Produk Bank yang Digunakan", "NumOfProducts")

    # =========================
    # INPUT UANG (EURO)
    # =========================
    def euro_input(label, col_name):
        value = st.number_input(
            label,
            min_value=limits[col_name]["min"],
            max_value=limits[col_name]["max"],
            value=None,
            step=100.0,
            format="%.2f",
            help=f"Batas training: €{limits[col_name]['min']:,.2f} - €{limits[col_name]['max']:,.2f}"
        )

        if value is not None:
            st.caption(f"💶 Format: €{value:,.2f}")

        return value

    balance = euro_input("Total Saldo di Rekening (€)", "Balance")
    estimated_salary = euro_input("Perkiraan Gaji Tahunan (€)", "EstimatedSalary")

    # =========================
    # INPUT KATEGORI
    # =========================
    geography = st.selectbox("Negara Tempat Tinggal", ["", "France", "Spain", "Germany"])
    gender = st.selectbox("Jenis Kelamin", ["", "Male", "Female"])
    has_cr_card = st.selectbox("Memiliki Kartu Kredit?", ["", 0, 1])
    is_active_member = st.selectbox("Nasabah Aktif?", ["", 0, 1])

    # =========================
    # PREPROCESSING
    # =========================
    def preprocess_input(data_dict):
        df = pd.DataFrame([data_dict])

        for col, scaler in scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[col].values.reshape(-1, 1)).ravel()

        df = pd.get_dummies(df)

        for col in fitur_model:
            if col not in df.columns:
                df[col] = 0

        df = df[fitur_model]
        return df

    # =========================
    # VALIDASI & PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Sekarang"):

        input_data = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "NumOfProducts": num_products,
            "Balance": balance,
            "EstimatedSalary": estimated_salary,
            "Geography": geography,
            "Gender": gender,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member
        }

        if any(v is None or v == "" for v in input_data.values()):
            st.warning("⚠️ Semua data harus diisi sebelum melakukan prediksi.")
            return

        try:
            X_input = preprocess_input(input_data)
            pred = int(model.predict(X_input)[0])
            prob = float(model.predict_proba(X_input)[0][1])

            st.subheader("📊 Hasil Prediksi")

            # Jika 1 = Excited (Loyal), 0 = Churn
            if pred == 1:
                st.success("✅ Nasabah Diprediksi Excited (Tetap Setia)")
                st.write("Nasabah kemungkinan besar akan tetap menggunakan layanan bank.")
            else:
                st.error("⚠️ Nasabah Berpotensi Berhenti (Churn)")
                st.write("Nasabah memiliki risiko berhenti menggunakan layanan bank.")
            # Tampilkan probabilitas Excited dan Churn (urut sesuai hasil prediksi)
            if pred == 1:
                st.info(f"Probabilitas Nasabah Excited: {prob:.2%}")
                st.caption(f"Probabilitas Nasabah Churn: {(1 - prob):.2%}")
            else:
                st.info(f"Probabilitas Nasabah Churn: {(1 - prob):.2%}")
                st.caption(f"Probabilitas Nasabah Excited: {prob:.2%}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
