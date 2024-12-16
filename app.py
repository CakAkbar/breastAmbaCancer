import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

st.title("Aplikasi Visualisasi dan Perbandingan Metode Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Pratinjau:")
    st.dataframe(data.head())

    st.sidebar.header("Pengaturan Clustering")

    # Pilih metode clustering
    method = st.sidebar.selectbox(
        "Pilih Metode Clustering", ["K-means", "AHC", "K-medoids"])

    # Pilih jumlah klaster
    n_clusters = st.sidebar.slider("Jumlah Klaster", 2, 10, 3)

    if method == "AHC":
        linkage_method = st.sidebar.selectbox(
            "Metode Linkage untuk AHC", ["ward", "complete", "average", "single"])

    # Preprocessing data
    X = StandardScaler().fit_transform(data.select_dtypes(include=np.number))

    if method == "K-means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "AHC":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    elif method == "K-medoids":
        model = KMedoids(n_clusters=n_clusters, random_state=42)

    labels = model.fit_predict(X)
    data['Cluster'] = labels

    st.subheader("Hasil Clustering")
    st.dataframe(data.head())

    # Visualisasi hasil clustering
    st.subheader("Visualisasi Clustering")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.set_title(f"{method} - Scatter Plot")
    st.pyplot(fig)

    if method == "AHC":
        st.subheader("Dendrogram")
        plt.figure(figsize=(10, 7))
        dendrogram = sch.dendrogram(sch.linkage(X, method=linkage_method))
        st.pyplot(plt)

    # Evaluasi model
    silhouette_avg = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)

    st.subheader("Metrik Evaluasi")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Davies-Bouldin Index: {dbi_score:.2f}")

    # Perbandingan hasil
    st.subheader("Grafik Perbandingan")
    comparison_data = pd.DataFrame({
        'Metode': [method],
        'Silhouette Score': [silhouette_avg],
        'Davies-Bouldin Index': [dbi_score]
    })
    st.dataframe(comparison_data)
else:
    st.info("Silakan unggah file CSV untuk memulai.")