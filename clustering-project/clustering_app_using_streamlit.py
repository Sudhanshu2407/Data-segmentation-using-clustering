import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

# Create Streamlit web application
def main():
    st.title("Clustering on Mall Customers")
    st.write("Performing K-means and Agglomerative clustering on Mall Customers dataset")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display column selection for clustering
        st.subheader("Column Selection")
        columns = data.columns.tolist()
        clustering_columns = st.multiselect("Select columns for clustering", columns)

        # Perform K-means clustering
        kmeans_clusters = st.slider("Number of K-means clusters", min_value=2, max_value=10, value=5)
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
        kmeans_predictions = kmeans.fit_predict(data[clustering_columns])
        data['KMeans_Clusters'] = kmeans_predictions

        # Perform Agglomerative clustering
        agg_clusters = st.slider("Number of Agglomerative clusters", min_value=2, max_value=10, value=5)
        agg = AgglomerativeClustering(n_clusters=agg_clusters)
        agg_predictions = agg.fit_predict(data[clustering_columns])
        data['Agglomerative_Clusters'] = agg_predictions

        # Display the updated dataframe
        st.subheader("Updated Dataframe")
        st.write(data)

        # Resolve warning for using pyplot
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Plot the clusters using scatter plot
        st.subheader("Cluster Visualization")
        x_column = st.selectbox("Select X-axis column", columns)
        y_column = st.selectbox("Select Y-axis column", columns)
        cluster_column = st.selectbox("Select Cluster column", ['KMeans_Clusters', 'Agglomerative_Clusters'])
        plt.scatter(data[x_column], data[y_column], c=data[cluster_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title("Cluster Visualization")
        st.pyplot()

# Run the web application
if __name__ == "__main__":
    main()