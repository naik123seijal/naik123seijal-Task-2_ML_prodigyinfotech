import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Customer_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Total_Spent': [150, 200, 50, 120, 300, 180, 220, 90, 400, 250],
    'Items_Purchased': [15, 18, 5, 12, 25, 14, 20, 7, 30, 22],
    'Purchase_Frequency': [5, 7, 2, 4, 10, 6, 8, 3, 12, 9]
}

df = pd.DataFrame(data)

features = df[['Total_Spent', 'Items_Purchased', 'Purchase_Frequency']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

 using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Total_Spent'], y=df['Items_Purchased'], hue=df['Cluster'], palette='viridis', s=100)
plt.xlabel('Total Spent')
plt.ylabel('Items Purchased')
plt.title('Customer Clusters')
plt.legend(title='Cluster')
plt.show()

print(df)
