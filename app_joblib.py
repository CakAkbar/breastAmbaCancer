import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.stats import spearmanr
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Load Model dan Scaler menggunakan joblib
@st.cache_resource
def load_model():
    model = joblib.load('mlp_model.joblib')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.joblib')
    return scaler

model = load_model()
scaler = load_scaler()

# Sidebar navigation
st.sidebar.title("Breast Cancer Prediction")
navigation = st.sidebar.radio("Navigation", ["Data Understanding", "Data Preprocessing", "Modeling", "Prediction"])

# Section: Data Understanding
if navigation == "Data Understanding":
    st.title("Data Understanding")
    st.write("Kanker payudara merupakan salah satu jenis kanker yang paling umum terjadi pada wanita di seluruh dunia. Meskipun prevalensinya lebih tinggi pada wanita, kanker payudara juga dapat memengaruhi pria meskipun dalam jumlah yang jauh lebih kecil. Kanker payudara dapat dibagi menjadi dua jenis utama: kanker payudara jinak dan kanker payudara ganas. Kanker payudara ganas, yang juga dikenal sebagai kanker payudara invasif, merupakan jenis yang paling serius dan memerlukan perhatian medis yang mendalam.")
    st.write("Kami akan menganalisis data kanker payudara dengan fokus pada pemahaman perbedaan antara kanker payudara jinak dan ganas, sehingga dapat menentukan apakah suatu kanker payudara tersebut merupakan kategori jinak atau ganas dengan menggunakan model yang akan kami buat. Diharapkan dengan model yang kami buat dapat mempermudah paramedis untuk mengidentifikasi kanker payudara. ")
    st.write("Di bagian ini, Anda dapat memahami fitur-fitur yang digunakan dalam model prediksi kanker payudara.")

    # Sumber Data
    st.subheader("Sumber Data")
    st.write("""
    Dataset ini adalah Breast Cancer Wisconsin (Original) yang diperoleh dari UCI Machine Learning Repository ([link dataset](https://archive.ics.uci.edu/dataset/15/+breast+cancer+wisconsin+original)). Dataset ini dikumpulkan oleh Dr. William H. Wolberg di University of Wisconsin Hospitals, Madison. 
    
    Referensi penelitian yang mendasari pengumpulan dataset ini meliputi:
    1. O. L. Mangasarian dan W. H. Wolberg: "Cancer diagnosis via linear programming".
    2. William H. Wolberg dan O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology".
    3. O. L. Mangasarian, R. Setiono, dan W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis".
    4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets".
    """)

    # Integrasi Data
    st.subheader("Integrasi Data")
    st.write("""
    Dataset ini dapat diambil dengan menggunakan package `ucimlrepo` yang memungkinkan pengambilan dataset dari UCI Machine Learning Repository secara langsung. Berikut adalah contoh kode untuk mengambil data dan menyimpannya ke dalam CSV:
    """)

    st.code("""
from ucimlrepo import fetch_ucirepo
  
# Fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
data = breast_cancer_wisconsin_original.data.original
data.to_csv("breast_cancer_wisconsin_original.csv", index=True)
print(data.info())
print(data.head())
    """, language='python')

    # Eksplorasi Data
    st.subheader("Eksplorasi Data")
    st.write("Untuk memudahkan pemahaman dataset, kita melakukan eksplorasi dan visualisasi untuk melihat distribusi dari masing-masing fitur dan label.")

    st.subheader("Distribusi Kelas")
    st.write("Pada dataset ini, kolom `Class` menunjukkan apakah tumor adalah kanker jinak (2) atau ganas (4). Berikut adalah visualisasi distribusi jumlah sampel untuk setiap kelas.")
    data = pd.read_csv('breast_cancer_wisconsin_original.csv')

    # Distribusi kelas
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=data, palette='viridis')
    plt.title('Jumlah Sampel per Kelas')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Sampel')
    st.pyplot(plt)
    class_counts = Counter(data['Class'])
    st.write("Distribusi kelas dalam dataset:", class_counts)

    st.subheader("Distribusi Fitur")
    plt.figure(figsize=(12, 10))
    data.hist(bins=30, figsize=(15, 10), color='blue')
    plt.suptitle('Distribusi Fitur')
    st.pyplot(plt)

    # Struktur Dataset
    st.subheader("Struktur Dataset")
    st.write("""
    Dataset ini terdiri dari 11 kolom, dengan 9 kolom fitur dan 1 kolom label yang menunjukkan apakah tumor adalah kanker jinak (2) atau ganas (4). Berikut adalah daftar fitur:
    - `Clump_thickness`: Ketebalan kelompok sel.
    - `Uniformity_of_cell_size`: Keseragaman ukuran sel.
    - `Uniformity_of_cell_shape`: Keseragaman bentuk sel.
    - `Marginal_adhesion`: Adhesi sel di tepi.
    - `Single_epithelial_cell_size`: Ukuran sel epitel tunggal.
    - `Bare_nuclei`: Inti sel tanpa sitoplasma.
    - `Bland_chromatin`: Sifat kromatin.
    - `Normal_nucleoli`: Nukleolus normal.
    - `Mitoses`: Jumlah pembelahan sel.
    
    Label pada dataset adalah:
    - `Class`: Mengindikasikan jenis kanker - `2` (jinak) dan `4` (ganas).
    """)

    # Menampilkan struktur dataset
    st.write("Struktur dataset:", data.shape)
    st.write("Kolom dataset:", data.columns.tolist())
    st.write("Tipe data dari masing-masing kolom:")
    st.write(data.dtypes)

    # Analisis Korelasi Spearman
    st.subheader("Analisis Korelasi Spearman")
    st.write("""
    Korelasi Spearman digunakan untuk mengevaluasi kekuatan dan arah hubungan antara fitur dalam dataset ini.
    
    Berikut ini adalah hasil korelasi Spearman antara beberapa fitur penting. Nilai korelasi dan p-value ditampilkan bersama dengan interpretasi kekuatan hubungan dan arah hubungan.
    """)

    # Daftar kolom yang akan diuji korelasinya
    columns = [
        "Uniformity_of_cell_size",
        "Uniformity_of_cell_shape",
        "Marginal_adhesion",
        "Single_epithelial_cell_size",
        "Bare_nuclei",
        "Bland_chromatin",
        "Normal_nucleoli",
        "Mitoses"
    ]

    # Menyimpan hasil korelasi dalam dictionary
    correlation_results = []

    # Loop untuk menghitung korelasi Spearman untuk setiap pasangan kolom
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            
            # Hitung koefisien korelasi Spearman dan p-value
            corr, p_value = spearmanr(data[col1], data[col2])
            
            # Tentukan kategori kekuatan hubungan berdasarkan koefisien korelasi
            if abs(corr) >= 0 and abs(corr) <= 0.25:
                strength = "Hubungan sangat lemah"
            elif abs(corr) > 0.25 and abs(corr) <= 0.50:
                strength = "Hubungan cukup"
            elif abs(corr) > 0.50 and abs(corr) <= 0.75:
                strength = "Hubungan kuat"
            elif abs(corr) > 0.75 and abs(corr) < 1:
                strength = "Hubungan sangat kuat"
            elif abs(corr) == 1:
                strength = "Hubungan sempurna"
            
            # Tentukan arah hubungan
            direction = "Berbanding lurus (positif)" if corr > 0 else "Berbanding terbalik (negatif)"
            
            # Tentukan signifikansi hubungan
            significance = "Signifikan" if p_value < 0.05 else "Tidak signifikan"
            
            # Menyimpan hasil dalam dictionary
            correlation_results.append({
                "Variabel 1": col1,
                "Variabel 2": col2,
                "Spearman Correlation": corr,
                "P-Value": p_value,
                "Signifikansi": significance,
                "Kekuatan Hubungan": strength,
                "Arah Hubungan": direction
            })

    # Membuat DataFrame dari hasil
    results_df = pd.DataFrame(correlation_results)
    st.write("Hasil Korelasi Spearman:")
    st.dataframe(results_df)

    st.write("""
    **Interpretasi Kekuatan Hubungan:**
    - **Hubungan sangat lemah**: Korelasi antara dua variabel hampir tidak ada.
    - **Hubungan cukup**: Korelasi lemah, tetapi ada keterkaitan.
    - **Hubungan kuat**: Korelasi yang cukup kuat.
    - **Hubungan sangat kuat**: Korelasi yang sangat erat.
    - **Hubungan sempurna**: Korelasi sempurna, nilai 1 atau -1.
    
    **Arah Hubungan:**
    - **Berbanding lurus (positif)**: Kedua variabel bergerak ke arah yang sama.
    - **Berbanding terbalik (negatif)**: Kedua variabel bergerak ke arah berlawanan.
    """)

    # Summary Statistik
    st.subheader("Summary Statistik")
    st.write("Berikut ini adalah statistik ringkasan dari fitur dalam dataset, termasuk nilai rata-rata, standar deviasi, nilai minimum, dan maksimum.")
    st.write(data.iloc[:, 1:10].describe())
    
elif navigation == "Modeling":
    st.title("Modeling")
    
    # Pengantar
    st.write("""
    Pada halaman ini, kita akan mempelajari tiga model jaringan saraf: 
    1. **Single Layer Perceptron (SLP)**
    2. **Multi-Layer Perceptron (MLP)**
    3. **Recurrent Neural Network (RNN)**

    Setiap model memiliki karakteristik unik dan kemampuan yang berbeda dalam memproses data.
    """)

    # Single Layer Perceptron
    st.header("1. Single Layer Perceptron (SLP)")
    st.write("""
    Single Layer Perceptron adalah jaringan saraf sederhana dengan satu lapisan output yang cocok untuk klasifikasi linier.
    Berikut langkah-langkah dalam SLP:
    1. **Input**: Data input dimasukkan sebagai vektor fitur.
    2. **Bobot**: Setiap fitur memiliki bobot tersendiri.
    3. **Persamaan Linear**: Nilai \( z = W \cdot X + b \) dihitung.
    4. **Aktivasi**: Fungsi aktivasi (Step Function) digunakan untuk mengubah nilai menjadi kelas output.
    5. **Pembaharuan Bobot**: Jika prediksi salah, bobot diperbarui berdasarkan error.

    Rumus Pembaruan Bobot:
    """)
    st.latex(r"w_i = w_i + \alpha (y_{\text{actual}} - y_{\text{predicted}}) x_i")
    st.write("di mana \( \alpha \) adalah learning rate.")

    # Multi-Layer Perceptron
    st.header("2. Multi-Layer Perceptron (MLP)")
    st.write("""
    Multi-Layer Perceptron memiliki lapisan tersembunyi yang memungkinkan model ini menangani masalah non-linier.
    Langkah-langkah dalam MLP:
    1. **Input Layer**: Data input dimasukkan sebagai vektor fitur.
    2. **Lapisan Tersembunyi**: Neuron dalam lapisan tersembunyi menerima input dari lapisan input.
    3. **Lapisan Output**: Keluaran dihitung sebagai hasil dari lapisan terakhir.
    4. **Loss Function**: Loss function digunakan untuk mengukur kesalahan.
    5. **Backpropagation**: Algoritma backpropagation digunakan untuk memperbarui bobot berdasarkan error.

    Rumus Backpropagation:
    """)
    st.latex(r"w_{\text{new}} = w_{\text{old}} - \alpha \frac{\partial \text{Loss}}{\partial w}")

    # Recurrent Neural Network
    st.header("3. Recurrent Neural Network (RNN)")
    st.write("""
    Recurrent Neural Network memiliki kemampuan untuk mengingat urutan informasi sebelumnya, cocok untuk data berurutan.
    Berikut adalah rumus yang digunakan dalam RNN:
    """)

    # Menampilkan Rumus Hidden State dan Output menggunakan LaTeX
    st.write("Rumus Hidden State:")
    st.latex(r"h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)")
    st.write("""
    di mana:
    - \( h_t \): Hidden state pada waktu t
    - \( W_{xh} \): Bobot antara input dan hidden state
    - \( W_{hh} \): Bobot antara hidden state sebelumnya dan hidden state saat ini
    - \( b_h \): Bias untuk hidden state
    """)

    st.write("Rumus Output:")
    st.latex(r"y_t = \text{softmax}(W_{hy} h_t + b_y)")
    st.write("""
    di mana:
    - \( y_t \): Output pada waktu t
    - \( W_{hy} \): Bobot antara hidden state dan output
    - \( b_y \): Bias untuk output
    """)

    # Kesimpulan
    st.header("Kesimpulan")
    st.write("""
    Ketiga model ini memiliki kelebihan masing-masing:
    - **Single Layer Perceptron (SLP)**: Sederhana, cocok untuk masalah linier.
    - **Multi-Layer Perceptron (MLP)**: Lebih kompleks, dapat menangani masalah non-linier.
    - **Recurrent Neural Network (RNN)**: Mampu menangani data berurutan dan mempertimbangkan konteks waktu.

    Dengan menggunakan notasi LaTeX, kita dapat menampilkan rumus dengan lebih rapi dan mudah dibaca.
    """)



# Section: Data Preprocessing
if navigation == "Data Preprocessing":
    st.title("Data Preprocessing")
    st.write("""
    Data preprocessing adalah tahap penting dalam proses pengolahan data yang melibatkan berbagai teknik dan langkah-langkah untuk memastikan bahwa data yang digunakan berkualitas tinggi, bebas dari noise, dan sesuai dengan kebutuhan analisis. 
    Pada tahapan ini, beberapa langkah utama yang akan dibahas meliputi Missing Values, Data Imbalance, dan Data Cleaning.
    """)

    # Missing Values
    st.header("1. Missing Values")
    st.write("""
    Missing values adalah data yang hilang atau kosong dalam dataset yang dapat menyebabkan bias atau menurunkan performa model. Beberapa metode yang umum digunakan untuk menangani missing values antara lain:
    """)

    st.subheader("Metode untuk Menangani Missing Values")
    st.write("""
    - **Deletion (Penghapusan)**:
      - **Listwise Deletion**: Menghapus seluruh baris yang memiliki missing values.
      - **Pairwise Deletion**: Menghapus data hanya pada analisis tertentu jika kolom yang dibutuhkan memiliki missing values.
      
    - **Mean, Median, atau Mode Imputation**:
      - **Mean Imputation**: Mengisi nilai yang hilang dengan rata-rata kolom terkait, cocok untuk data numerik.
      - **Median Imputation**: Mengisi dengan median, cocok jika ada outlier.
      - **Mode Imputation**: Mengisi dengan nilai yang paling sering muncul, lebih tepat untuk data kategorikal.
      
    - **Advanced Imputation Techniques**:
      - **K-Nearest Neighbors (KNN)**: Mengisi nilai hilang dengan melihat nilai tetangga terdekat.
      - **Multiple Imputation**: Menggunakan beberapa metode imputasi dan memilih yang paling relevan.
      - **Iterative Imputer**: Pendekatan berbasis regresi untuk memprediksi nilai yang hilang secara iteratif.
    """)

    st.subheader("Contoh Implementasi")
    st.code("""
# Menampilkan jumlah missing values
missing_values = data.isnull().sum()
st.write("Missing Values per Kolom:", missing_values)

# Mengisi missing values menggunakan median
data = data.apply(lambda x: x.fillna(x.median()) if x.dtype in ['float64', 'int64'] else x.fillna(x.mode()[0]))
    """, language='python')

    # Data Imbalance
    st.header("2. Data Imbalance")
    st.write("""
    Data imbalance terjadi ketika distribusi kelas dalam dataset tidak merata, sehingga model lebih mudah mengklasifikasikan kelas mayoritas dan kurang akurat pada kelas minoritas.
    """)

    st.subheader("Metode untuk Mengatasi Data Imbalance")
    st.write("""
    - **Oversampling**:
      - **Random Oversampling**: Duplikasi sampel kelas minoritas secara acak untuk meningkatkan jumlahnya.
      - **SMOTE (Synthetic Minority Oversampling Technique)**: Membuat sampel sintetis dari kelas minoritas berdasarkan interpolasi antara dua sampel minoritas yang berdekatan.
      
    - **Undersampling**:
      - Mengurangi jumlah sampel dari kelas mayoritas secara acak, efektif namun bisa menghilangkan informasi penting.
      
    - **Advanced Sampling Techniques**:
      - **ADASYN (Adaptive Synthetic Sampling)**: Mirip SMOTE, tetapi lebih fokus pada data yang sulit diklasifikasi.
      - **Cluster-Based Sampling**: Mengelompokkan data terlebih dahulu, lalu melakukan oversampling atau undersampling dalam cluster.
    """)

    st.subheader("Contoh Implementasi")
    st.code("""
from imblearn.over_sampling import SMOTE
from collections import Counter

# Menampilkan distribusi kelas sebelum SMOTE
st.write("Distribusi Kelas Sebelum SMOTE:", Counter(data['Class']))

# Menggunakan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Menampilkan distribusi kelas setelah SMOTE
st.write("Distribusi Kelas Setelah SMOTE:", Counter(y_resampled))
    """, language='python')

    # Data Cleaning
    st.header("3. Data Cleaning")
    st.write("""
    Data cleaning bertujuan untuk membersihkan dataset dari data yang tidak diperlukan atau dapat menyebabkan noise dalam analisis. Langkah-langkah utama dalam data cleaning termasuk menghapus data yang tidak relevan, menangani duplikasi, dan mengatasi outliers.
    """)

    st.subheader("Langkah-Langkah Data Cleaning")
    st.write("""
    - **Menghapus Kolom yang Tidak Relevan**: Kolom yang tidak memiliki kontribusi signifikan dalam model sebaiknya dihapus.
    - **Menangani Outliers**:
      - **IQR (Interquartile Range) Method**: Menghapus nilai yang berada di luar rentang IQR.
      - **Z-Score**: Menghapus data yang memiliki nilai Z-score di luar batas tertentu (misalnya > 3 atau < -3).
    - **Mengatasi Duplikasi**: Data duplikat dihapus agar tidak mempengaruhi model.
    """)

    st.subheader("Contoh Implementasi")
    st.code("""
# Menghapus kolom yang tidak relevan
data = data.drop(columns=['Unnamed: 0', 'Sample_code_number'])

    """, language='python')

    st.write("Dengan mengikuti tahap-tahap preprocessing ini, dataset menjadi lebih bersih dan siap untuk dimasukkan ke dalam model prediksi.")


# Section: Prediction
elif navigation == "Prediction":
    st.title("Prediction")
    st.write("Masukkan nilai fitur untuk memprediksi apakah tumor adalah kanker jinak atau ganas.")

    # Input features for prediction
    clump_thickness = st.number_input("Clump Thickness", min_value=1, max_value=10, step=1)
    uniformity_of_cell_size = st.number_input("Uniformity of Cell Size", min_value=1, max_value=10, step=1)
    uniformity_of_cell_shape = st.number_input("Uniformity of Cell Shape", min_value=1, max_value=10, step=1)
    marginal_adhesion = st.number_input("Marginal Adhesion", min_value=1, max_value=10, step=1)
    single_epithelial_cell_size = st.number_input("Single Epithelial Cell Size", min_value=1, max_value=10, step=1)
    bare_nuclei = st.number_input("Bare Nuclei", min_value=1, max_value=10, step=1)
    bland_chromatin = st.number_input("Bland Chromatin", min_value=1, max_value=10, step=1)
    normal_nucleoli = st.number_input("Normal Nucleoli", min_value=1, max_value=10, step=1)
    mitoses = st.number_input("Mitoses", min_value=1, max_value=10, step=1)

    # Prepare data for prediction without Sample_code_number
    input_data = [[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape, 
                   marginal_adhesion, single_epithelial_cell_size, bare_nuclei, 
                   bland_chromatin, normal_nucleoli, mitoses]]

    # Transform the input data using the scaler
    scaled_input = scaler.transform(input_data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_input)
        if prediction[0] == 0:
            st.success("Prediksi: Tumor Jinak")
        else:
            st.error("Prediksi: Tumor Ganas")
