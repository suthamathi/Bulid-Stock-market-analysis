import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
# Step 2: Download Adjusted Close Prices for 2022
tickers = get_sp500_tickers()
print(f"Fetched {len(tickers)} S&P 500 tickers.")

start_date = "2022-01-01"
end_date = "2022-12-31"

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
print(" Data Structure:\n", data.head())

# ---------------------------------------------------
#  Step 3: Extract 'Adj Close' Prices Only
if 'Adj Close' in data.columns:
    data = data['Adj Close']
elif isinstance(data.columns, pd.MultiIndex) and 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
else:
    print("'Adj Close' column not found. Exiting script.")
    exit()

# ---------------------------------------------------
#  Step 4: Clean Missing Data
data.dropna(axis=1, how='all', inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# ---------------------------------------------------
#  Step 5: Z-Score Normalization of Prices
scaler = StandardScaler()
prices_transposed = data.T
zscore_prices = scaler.fit_transform(prices_transposed)

zscore_df = pd.DataFrame(
    zscore_prices,
    index=prices_transposed.index,
    columns=prices_transposed.columns.strftime('%Y-%m-%d')
)

zscore_df.to_csv("sp500_2022_zscore_prices.csv")
print(" Z-score normalized prices for 2022 saved.")

# ---------------------------------------------------
# Step 6: Elbow Method & Silhouette Score
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
plt.savefig("kmeans_2022_elbow_silhouette.png", dpi=300)
plt.show()
print(" Elbow and Silhouette plots saved as 'kmeans_2022_elbow_silhouette.png'")

# ---------------------------------------------------
#  Step 7: Final KMeans Clustering
best_k = 5  #  Adjust after inspecting Elbow/Silhouette plots

kmeans = KMeans(n_clusters=best_k, random_state=42)
zscore_df['Cluster'] = kmeans.fit_predict(zscore_df)

# Save clustering result
zscore_df.to_csv("sp500_2022_kmeans_clusters.csv")
print(f" Final KMeans clustering complete with k={best_k}. Saved to 'sp500_2022_kmeans_clusters.csv'")

# ---------------------------------------------------
#  Step 8: PCA-Based Cluster Visualization
features = zscore_df.drop(columns=['Cluster'])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']

for cluster_id in range(best_k):
    mask = zscore_df['Cluster'] == cluster_id
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                c=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id + 1}", alpha=0.6, s=50)

# Plot centroids
pca_centroids = PCA(n_components=2).fit_transform(centroids)
plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1],
            c='yellow', s=200, marker='*', label='Centroids')

plt.title(f"KMeans Clustering (PCA 2D Projection, 2022, k={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_2022_pca_clusters.png", dpi=300)
plt.show()

print(" PCA-based cluster plot saved as 'kmeans_2022_pca_clusters.png'")
