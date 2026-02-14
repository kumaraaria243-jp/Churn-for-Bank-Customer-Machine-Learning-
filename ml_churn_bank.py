from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import plotly.express as px
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # untuk menyimpan model yang sudah dilatih


def ml_model():
    df = pd.read_csv('churn(Cleaned).csv')
    # Menghapus Kolom yang Tidak Diperlukan
    st.write("Sebelum Pemrosesan Data Kolom **RowNumber**, **CustomerId**, dan **Surname** **dihapus** karena bersifat identifikasi unik dan tidak memiliki relevansi prediktif terhadap churn. "\
             "Sementara itu, **Kategori_Umur**, **Kategori_Skor_Kredit**, **Status_Saldo**, **Kategori_Gaji**, dan **Customer_Loyalty_Level** dieliminasi karena merupakan variabel turunan yang berpotensi "\
             "menimbulkan redundansi informasi dan multikolinearitas dalam pemodelan.")
    # Code Penghapusan Kolom
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Kategori_Umur', 'Kategori_Skor_Kredit', 'Status_Saldo', 'Kategori_Gaji', 'Customer_Loyalty_Level'], errors = "ignore")
    
    # Memisahkan Kolom Numbers dan Kategori
    numbers = df.select_dtypes(include = ['number']).columns
    categories = df.select_dtypes(exclude = ['number']).columns


    # 1.Data Preparation
    st.subheader("1. Data Preparation")
    # Membaca Dataset Churn for Bank Customers
    st.write('**a. Dataset yang digunakan**')
    st.write("Dataset yang digunakan untuk pemodelan adalah **Bank Customer Churn**, yaitu kumpulan data yang berisi informasi nasabah bank dengan tujuan"\
             "untuk memprediksi kemungkinan pelanggan berhenti menggunakan layanan (churn).")
    st.dataframe(df.head())

    # Deteksi dan penanganan outlier dengan IQR Method
    st.write("**b. Deteksi Outlier (Data Numerik)**")
    st.write("Deteksi outlier adalah proses mengidentifikasi data yang memiliki nilai jauh berbeda atau menyimpang dari sebagian besar data lainnya dalam suatu kumpulan data")
    # Code Deteksi Outlier | Ambil kolom numerik, tapi exclude kolom biner (0/1)
    numeric_columns = df.select_dtypes(include="number").columns
    numeric_columns = [c for c in numeric_columns if df[c].nunique(dropna=True) > 2]
    outlier_mask_df = pd.DataFrame(False, index=df.index, columns=numeric_columns)
    outlier_summary = {}
    for col in numeric_columns:
        s = df[col].dropna()
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask_df[col] = col_mask
        outlier_summary[col] = int(col_mask.sum())
    # Tabel outlier per kolom
    outlier_df = (
        pd.DataFrame(outlier_summary.items(), columns=["Kolom", "Jumlah Outlier"])
        .sort_values("Jumlah Outlier", ascending=False)
        .reset_index(drop=True))
    st.dataframe(outlier_df, use_container_width=True)
    # Total baris unik yang mengandung outlier
    total_outlier_rows = int(outlier_mask_df.any(axis=1).sum())
    # (Opsional tapi sering dibutuhkan) total kejadian outlier (cell)
    total_outlier_cells = int(outlier_mask_df.sum().sum())
    st.write(f"Total Outlier (akumulasi semua kolom): **{total_outlier_cells} Data**")


    # 2.Pemilihan Feature untuk Pemodelan
    st.subheader("2. Pemilihan Feature untuk Pemodelan")
    st.write("Kolom target **Excited** telah **berbentuk numerik**, dengan nilai 1 menunjukkan pelanggan yang excited dan 0 menunjukkan pelanggan yang churn. Oleh karena itu, tidak "\
             "diperlukan proses konversi atau pengkodean ulang, sehingga data dapat langsung digunakan dalam tahap pemodelan.")
    
    # Unvariat Analysis dengan Density Plot
    st.write("**a. Unvariat Analysis dengan Density Plot**")
    st.write("Analisis univariat dengan density plot digunakan untuk melihat distribusi satu variabel secara visual. Grafik ini menampilkan pola sebaran data sehingga memudahkan dalam "\
             "memahami karakteristik, kecenderungan, dan kemungkinan adanya nilai ekstrem sebelum dilakukan analisis lanjutan.")
    # Code Density Plot
    numbers = df.select_dtypes(include="number").columns.tolist()
    cols_per_row = 6
    n_cols = len(numbers)
    n_rows = math.ceil(n_cols / cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 3 * n_rows))
    # Jika cuma 1 baris, axes bukan array 2D → amankan
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    # Background transparan
    fig.patch.set_alpha(0)
    for i, col in enumerate(numbers):
        axes[i].set_facecolor("none")
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="black")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", labelrotation=45)
    # Hapus subplot kosong
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    
    # Korelasi Antar Fitur Numerik
    st.write("**b. Correlation Heatmap (Data Numerik)**")
    st.write("Correlation heatmap pada dataset Bank Customer Churn digunakan untuk menggambarkan hubungan antar variabel secara visual melalui tingkat korelasi. Visualisasi ini membantu "\
             "mengidentifikasi faktor-faktor yang memiliki hubungan kuat dengan churn serta mendeteksi potensi multikolinearitas antar fitur.")
    # Code Correlation Heatmap
    corr = df[numbers].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


    # Normalisasi data & encoding
    st.subheader("3. Normalisasi Data")
    # Menyalin data yang telah dibersihkan
    df_select = df.copy()

    # Normalisasi kolom numerik dengan MinMax Scaler
    st.write("Normalisasi data dilakukan menggunakan **Metode Min-Max Scaler**, yaitu teknik transformasi yang mengubah nilai setiap variabel ke dalam rentang tertentu, umumnya 0 hingga 1. "\
             "Metode ini bertujuan untuk menyamakan skala antar fitur sehingga meningkatkan stabilitas dan kinerja model dalam proses pemodelan.")
    # Code Normalisasi dengan MinMax Scaler | Simpan scaler untuk setiap kolom
    from sklearn.preprocessing import MinMaxScaler
    scalers = {}
    for col in numbers:
        scaler = MinMaxScaler()
        df_select[col] = scaler.fit_transform(df_select[col].values.reshape(len(df_select), 1))
        scalers[col] = scaler
    
    # Visualisasi Hasil Normalisasi dengan Density Plot
    st.write("**a. Density Plot Setelah Normalisasi (MinMax Scaler)**")
    st.write("Hasil normalisasi dengan density plot menunjukkan distribusi variabel setelah proses penskalaan menggunakan Min-Max Scaler. Visualisasi ini digunakan untuk memastikan bahwa pola "\
             "distribusi data tetap terjaga meskipun rentang nilainya telah disesuaikan. Dengan demikian, normalisasi tidak mengubah bentuk distribusi, tetapi hanya menyelaraskan skala data.")
    cols_per_row = 6
    n_cols = len(numbers)
    n_rows = math.ceil(n_cols / cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 3 * n_rows))
    # Jika cuma 1 baris, axes bukan array 2D → amankan
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    # Background transparan
    fig.patch.set_alpha(0)
    for i, col in enumerate(numbers):
        axes[i].set_facecolor("none")
        sns.histplot(df_select[col].dropna(), kde=True, ax=axes[i], color="black")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", labelrotation=45)
    # Hapus subplot kosong
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig, transparent=True)

    # One-Hot Encoding
    st.write("**b. One Hot Encoding (Data Kategori)**")
    st.write("One-Hot Encoding digunakan untuk mengubah data kategorikal seperti jenis kelamin, wilayah, atau tipe produk menjadi format numerik agar dapat dianalisis oleh model machine learning, "\
             "sehingga perusahaan dapat menghasilkan prediksi dan insight yang lebih akurat untuk pengambilan keputusan berbasis data.")
    # Code One-Hot Encoding
    df_select = pd.get_dummies(df_select, columns=categories)
    st.dataframe(df_select.head())


    # Pra-Modeling
    st.subheader("4. Pra-Modeling")
    st.write("Pra-modeling adalah tahap persiapan data sebelum pembuatan model analitik, seperti membersihkan dan mengolah data agar hasil prediksi lebih akurat, sehingga "\
             "keputusan yang diambil perusahaan menjadi lebih tepat dan berbasis data. Disini dilakukan **pemisahan variabel target (Excited)** dengan **variabel fitur**.")
    # Memisahkan variabel bebas (x) dan terikat (y)
    X = df_select.drop("Exited", axis=1)
    Y = df_select["Exited"]

    # Class Imbalance dan Class Balance
    st.write("**a. Handle Class**")
    st.write("Handle class adalah upaya mengelola ketidakseimbangan data antar kategori (misalnya churn vs tidak churn) agar model tidak bias dan mampu mendeteksi kasus penting secara lebih akurat untuk mendukung pengambilan keputusan yang tepat.")
    tab1, tab2 = st.tabs(["**Class Imbalance**", "**Class Balance**"]) 
    with tab1:
        # Handle Class Imbalance
        st.write("Class imbalance terjadi ketika satu **kategori data jauh lebih dominan daripada yang lain**, sehingga model bisa bias dan kurang efektif mendeteksi kasus penting seperti churn atau fraud.")
        st.dataframe(Y.value_counts())
    with tab2:
        # Class Balance dengan SMOTE
        st.write("**SMOTE digunakan untuk menyeimbangkan data yang timpang** dengan menambah sampel sintetis pada kelas minoritas agar model lebih akurat dan tidak bias dalam mendeteksi kasus penting seperti churn atau fraud.")
        # Code SMOTE
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state = 42)
        X, Y= sm.fit_resample(X, Y)
        # Dipanggil
        st.dataframe(Y.value_counts())  

    # Membagi data menjadi 80% untuk training dan 20% untuk testing
    st.write("**b. Split Data Train dan Test**")
    st.write("Data dibagi menjadi 80% untuk pelatihan (training) dan 20% untuk pengujian (testing) guna memastikan model dapat belajar dari data yang cukup dan diuji secara efektif untuk mengukur kinerjanya.")
    # Code Split Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Menampilkan hasil split data
    tab1, tab2 = st.tabs(["**Variabel Latih (Train)**", "**Variabel Uji (Test)**"]) 
    with tab1:
        st.write('Variabel X (train)')
        st.dataframe(X_train.head())
        st.write('Variabel Y (train)')
        st.dataframe(Y_train.head())
    with tab2:
        st.write('Variabel X (Test)')
        st.dataframe(X_test.head())
        st.write('Variabel Y (Test)')
        st.dataframe(Y_test.head())
    

    # Modelling
    st.subheader("5. Modelling")
    st.write("Modelling adalah proses membangun dan melatih model machine learning untuk mengidentifikasi nasabah yang berpotensi berhenti menggunakan layanan berdasarkan "\
             "pola data historis, sehingga bank dapat menyusun strategi retensi yang lebih tepat sasaran, mengurangi kehilangan pelanggan, dan meningkatkan profitabilitas.")

    # a.Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    # Inisiasi model Logistic Regression|
    model_logreg = LogisticRegression()
    # Train model
    model_logreg.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_logreg = model_logreg.predict(X_test)
    train_accuracy_logreg = model_logreg.score(X_train, Y_train)

    # b.Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    # Inisiasi model Decision Tree
    model_dt = DecisionTreeClassifier()
    # Train model
    model_dt.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_dt = model_dt.predict(X_test)
    train_accuracy_dt = model_dt.score(X_train, Y_train)

    # c.Random Forest
    from sklearn.ensemble import RandomForestClassifier
    # Inisiasi model Random Forest
    model_rf = RandomForestClassifier()
    # Train model
    model_rf.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_rf = model_rf.predict(X_test)
    train_accuracy_rf = model_rf.score(X_train, Y_train)

    # d.Support Vector Machine
    from sklearn.svm import SVC
    # Inisiasi model SVM
    model_svm = SVC(probability=True)
    # Train model
    model_svm.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_svm = model_svm.predict(X_test)
    train_accuracy_svm = model_svm.score(X_train, Y_train)

    # e.K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier
    # Inisiasi model KNN
    model_knn = KNeighborsClassifier(n_neighbors=47)
    # Train model
    model_knn.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_knn = model_knn.predict(X_test)
    train_accuracy_knn = model_knn.score(X_train, Y_train)

    # f.Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    # Inisiasi model Naive Bayes
    model_nb = GaussianNB()
    # Train model
    model_nb.fit(X_train, Y_train)
    # Prediksi menggunakan data test
    Y_pred_nb = model_nb.predict(X_test)
    # Akurasi data train
    train_accuracy_nb = model_nb.score(X_train, Y_train)

    # Menampilkan akurasi data train dalam bentuk tabel
    accuracy_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "Naive Bayes"
        ],
        "Training Accuracy": [
            train_accuracy_logreg,
            train_accuracy_dt,
            train_accuracy_rf,
            train_accuracy_svm,
            train_accuracy_knn,
            train_accuracy_nb
        ]
    })
    # Murutkan dari yang tertinggi
    accuracy_df = accuracy_df.sort_values("Training Accuracy", ascending=False).reset_index(drop=True)
    # Format persen 2 desimal
    accuracy_df["Training Accuracy"] = accuracy_df["Training Accuracy"].apply(lambda x: f"{x:.2%}")
    # Dipanggil
    st.write("**Perbandingan Training Accuracy Model**")
    st.dataframe(accuracy_df, use_container_width=True)


    # Evaluasi Model (Membangun Metrik Model Tp,Tn,Fp,Fn)
    st.subheader("6. Evaluasi Model")
    st.write("evaluasi model adalah proses mengukur kinerja model prediksi untuk memastikan kemampuannya dalam mengidentifikasi nasabah yang berpotensi churn secara akurat, sehingga strategi retensi yang dijalankan benar-benar efektif dan tepat sasaran.")

    # Import metrik evaluasi
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

    # a.Logistic Regression | Prediksi pada data testing
    Y_pred_logreg = model_logreg.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
    precision_logreg = precision_score(Y_test, Y_pred_logreg)
    recall_logreg = recall_score(Y_test, Y_pred_logreg)
    f1_logreg = f1_score(Y_test, Y_pred_logreg)
    roc_auc_logreg = roc_auc_score(Y_test, model_logreg.predict_proba(X_test)[:,1])

    # b.Decision Tree | Prediksi pada data testing
    Y_pred_dt = model_dt.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
    precision_dt = precision_score(Y_test, Y_pred_dt)
    recall_dt = recall_score(Y_test, Y_pred_dt)
    f1_dt = f1_score(Y_test, Y_pred_dt)
    roc_auc_dt = roc_auc_score(Y_test, model_dt.predict_proba(X_test)[:,1])

    # c.Random Forest | Prediksi pada data testing
    Y_pred_rf = model_rf.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
    precision_rf = precision_score(Y_test, Y_pred_rf)
    recall_rf = recall_score(Y_test, Y_pred_rf)
    f1_rf = f1_score(Y_test, Y_pred_rf)
    roc_auc_rf = roc_auc_score(Y_test, model_rf.predict_proba(X_test)[:,1])

    # d.Support Vector Machine | Prediksi pada data testing
    Y_pred_svm = model_svm.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
    precision_svm = precision_score(Y_test, Y_pred_svm)
    recall_svm = recall_score(Y_test, Y_pred_svm)
    f1_svm = f1_score(Y_test, Y_pred_svm)

    # e.K-Nearest Neighbors | Prediksi pada data testing
    Y_pred_knn = model_knn.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
    precision_knn = precision_score(Y_test, Y_pred_knn)
    recall_knn = recall_score(Y_test, Y_pred_knn)
    f1_knn = f1_score(Y_test, Y_pred_knn)
    roc_auc_knn = roc_auc_score(Y_test, model_knn.predict_proba(X_test)[:,1])

    # f.Naive Bayes | Prediksi pada data testing
    Y_pred_nb = model_nb.predict(X_test)
    # Hitung metrik evaluasi
    accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
    precision_nb = precision_score(Y_test, Y_pred_nb)
    recall_nb = recall_score(Y_test, Y_pred_nb)
    f1_nb = f1_score(Y_test, Y_pred_nb)
    cm_nb = confusion_matrix(Y_test, Y_pred_nb)
    roc_auc_nb = roc_auc_score(Y_test, model_nb.predict_proba(X_test)[:, 1])


    # Confusion Matrix
    st.subheader("7. Confusion Matrix")
    st.write("Alat evaluasi yang digunakan untuk melihat seberapa baik model dalam mengklasifikasikan suatu kasus (misalnya churn atau tidak churn) dengan membandingkan prediksi model terhadap kondisi sebenarnya, "\
             "sehingga perusahaan dapat memahami tingkat kesalahan prediksi dan dampaknya terhadap keputusan bisnis.")
    
    # Membangun Confusion Matrix untuk setiap model
    cm_logreg = confusion_matrix(Y_test, Y_pred_logreg)
    cm_dt = confusion_matrix(Y_test, Y_pred_dt)
    cm_rf = confusion_matrix(Y_test, Y_pred_rf)
    cm_svm = confusion_matrix(Y_test, Y_pred_svm)
    cm_knn = confusion_matrix(Y_test, Y_pred_knn)
    cm_nb = confusion_matrix(Y_test, Y_pred_nb)    
    
    # Simpan semua confusion matrix dalam dictionary
    conf_matrices = {
        "Logistic Regression": cm_logreg,
        "Decision Tree": cm_dt,
        "Random Forest": cm_rf,
        "Support Vector Machine": cm_svm,
        "K-Nearest Neighbors": cm_knn,
        "Naive Bayes": cm_nb}
    cols_per_row = 3
    n_models = len(conf_matrices)
    n_rows = math.ceil(n_models / cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
    axes = np.array(axes).reshape(-1)
    # Background transparan
    fig.patch.set_alpha(0)
    for i, (model_name, cm) in enumerate(conf_matrices.items()):
        axes[i].set_facecolor("none")
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i])
        axes[i].set_title(model_name, fontsize=11)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    # Hapus subplot kosong
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
    # jarak manual biar rapi
    plt.subplots_adjust(
        hspace=0.4,   # jarak vertikal antar baris
        wspace=0.3    # jarak horizontal antar kolom
    )
    # Dipanggil
    st.pyplot(fig, transparent=True)


    # Menampilkan metrik evaluasi dalam bentuk tabel
    st.subheader("8. Perbandingan Metrik Evaluasi Model")
    st.write("perbandingan metrik evaluasi model dilakukan untuk menentukan model terbaik dengan melihat keseimbangan antara akurasi, kemampuan mendeteksi kasus penting (recall), "\
             "ketepatan prediksi (precision), serta stabilitas performa (misalnya ROC-AUC), sehingga keputusan yang diambil lebih tepat dan berdampak optimal bagi perusahaan.")
    # Code Tabel Perbandingan Metrik Evaluasi Model
    df_evaluasi = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'Random Forest',
            'Support Vector Machine',
            'K-Nearest Neighbors',
            'Naive Bayes'
        ],
        'Accuracy': [
            accuracy_logreg, accuracy_dt, accuracy_rf,
            accuracy_svm, accuracy_knn, accuracy_nb
        ],
        'Precision': [
            precision_logreg, precision_dt, precision_rf,
            precision_svm, precision_knn, precision_nb
        ],
        'Recall': [
            recall_logreg, recall_dt, recall_rf,
            recall_svm, recall_knn, recall_nb
        ],
        'F1 Score': [
            f1_logreg, f1_dt, f1_rf,
            f1_svm, f1_knn, f1_nb
        ]
    })
    # Mengurutkan berdasarkan Accuracy tertinggi
    df_evaluasi = df_evaluasi.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    # Format ke persen 2 desimal
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    df_evaluasi[metric_cols] = df_evaluasi[metric_cols].applymap(lambda x: f"{x:.2%}")
    # Dipanggil
    st.dataframe(df_evaluasi, use_container_width=True)

    # Kesimpulan
    st.write("**Random Forest merupakan model terbaik** dengan akurasi 89.23% serta nilai precision, recall, dan F1-score tertinggi dibanding model lain. recall yang tinggi berarti model mampu mendeteksi "\
             "sebagian besar nasabah yang berpotensi berhenti, sehingga bank dapat melakukan tindakan retensi lebih cepat dan tepat sasaran. Precision yang tinggi juga memastikan strategi retensi tidak salah target. "\
             "Dengan performa paling stabil dan seimbang, Random Forest menjadi pilihan paling optimal untuk memprediksi dan mengelola risiko churn nasabah.")

    # ==========================================================
    # 9. MENYIMPAN MODEL TERBAIK (RANDOM FOREST) - TANPA UBAH KODE
    # ==========================================================
    st.subheader("9. Simpan Model Terbaik")
    st.write("Model terbaik (**Random Forest**) disimpan menggunakan **joblib** agar dapat digunakan kembali tanpa training ulang.")

    # Simpan model Random Forest
    joblib.dump(model_rf, "model_random_forest_churn.pkl", compress=3)

    # Opsional tapi disarankan untuk kebutuhan prediksi/deployment:
    # simpan urutan kolom fitur setelah get_dummies & preprocessing
    joblib.dump(X_train.columns.tolist(), "fitur_model.pkl")

    # simpan scaler MinMax agar input prediksi bisa dinormalisasi konsisten
    joblib.dump(scalers, "scalers_minmax.pkl")

    st.success("Berhasil menyimpan: model_random_forest_churn.pkl, fitur_model.pkl, scalers_minmax.pkl")
