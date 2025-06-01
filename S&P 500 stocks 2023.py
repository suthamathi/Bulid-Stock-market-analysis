import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# ---------------------------------------------------
#  Step 1: Get S&P 500 Tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        columns = row.find_all("td")
        ticker = columns[0].text.strip()
        if "." in ticker:
            ticker = ticker.replace(".", "-")
        tickers.append(ticker)
    return tickers

# ---------------------------------------------------
# Step 2: Download Adjusted Close Prices
tickers = get_sp500_tickers()
print(f" Fetched {len(tickers)} S&P 500 tickers.")

start_date = "2023-01-01"
end_date = "2023-12-31"

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
print(" Data Structure:\n", data.head())

# ---------------------------------------------------
#  Step 3: Extract 'Adj Close' Prices Only
if 'Adj Close' in data.columns:
    data = data['Adj Close']
elif isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
else:
    print(" 'Adj Close' column not found. Exiting script.")
    exit()

# ---------------------------------------------------
#  Step 4: Clean Missing Data
data.dropna(axis=1, how='all', inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# ---------------------------------------------------
# Step 5: Z-Score Normalization of Prices
scaler = StandardScaler()
prices_transposed = data.T
zscore_prices = scaler.fit_transform(prices_transposed)

zscore_df = pd.DataFrame(
    zscore_prices,
    index=prices_transposed.index,
    columns=prices_transposed.columns.strftime('%Y-%m-%d')
)

zscore_df.to_csv("sp500_zscore_prices_for_clustering.csv")
print("Z-score normalized prices saved.")

# ---------------------------------------------------
#  Step 6: Elbow Method & Silhouette Score (for KMeans)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(zscore_df)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(zscore_df, labels))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title(" Elbow Method - Inertia vs Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='s', color='green')
plt.title(" Silhouette Score vs Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", dpi=300)
plt.show()
print("Elbow and Silhouette plots saved as 'kmeans_elbow_silhouette.png'")

# ---------------------------------------------------
# Step 7: Clustering Methods

# Best k chosen from Elbow/Silhouette plot (adjust if needed)
best_k = 5

#  KMeans
kmeans = KMeans(n_clusters=best_k, random_state=42)
zscore_df['Cluster_KMeans'] = kmeans.fit_predict(zscore_df)
print(" KMeans clustering done.")

#  Agglomerative
agglo = AgglomerativeClustering(n_clusters=best_k)
zscore_df['Cluster_Agglo'] = agglo.fit_predict(zscore_df.drop(columns=['Cluster_KMeans']))
print(" Agglomerative clustering done.")

#  DBSCAN with custom eps
dbscan_eps = 6.5  
dbscan = DBSCAN(eps=dbscan_eps, min_samples=5)
zscore_df['Cluster_DBSCAN'] = dbscan.fit_predict(zscore_df.drop(columns=['Cluster_KMeans', 'Cluster_Agglo']))
print(f" DBSCAN clustering done (eps={dbscan_eps}).")

# Save all clustering results
zscore_df.to_csv("sp500_all_clusterings.csv")
print(" All clustering labels saved to 'sp500_all_clusterings.csv'")
#  Step 8: PCA and Cluster Visualization
pca = PCA(n_components=2)
features = zscore_df.drop(columns=['Cluster_KMeans', 'Cluster_Agglo', 'Cluster_DBSCAN'])
pca_result = pca.fit_transform(features)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'gray', 'olive']

# Plot for each clustering method
cluster_methods = {
    "KMeans": 'Cluster_KMeans',
    "Agglomerative": 'Cluster_Agglo',
    "DBSCAN": 'Cluster_DBSCAN'
}

for method_name, label_col in cluster_methods.items():
    labels = zscore_df[label_col]
    unique_labels = sorted(set(labels))
    plt.figure(figsize=(10, 7))

    for cluster_id in unique_labels:
        mask = (labels == cluster_id)
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                    c=colors[cluster_id % len(colors)] if cluster_id != -1 else 'black',
                    label=f"Cluster {cluster_id}" if cluster_id != -1 else "Noise",
                    alpha=0.6, s=50)

    if method_name != "DBSCAN":
        model = KMeans(n_clusters=best_k, random_state=42).fit(pca_result)
        centroids = model.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=200, marker='*', label='Centroids')

    plt.title(f"{method_name} Clustering (PCA 2D Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{method_name.lower()}_pca_clusters.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f" {method_name} PCA-based cluster plot saved as '{filename}'")
