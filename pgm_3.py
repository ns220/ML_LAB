from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
iris = load_iris()
features = iris.data  
target = iris.target  
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_standardized)
pca_df = pd.DataFrame(data=features_pca, columns=["Principal Component 1", "Principal Component 2"])
pca_df["Target"] = target
plt.figure(figsize=(8, 6))
colors = ["red", "green", "blue"]
for label, color in zip(iris.target_names, colors):
    plt.scatter(
        pca_df.loc[pca_df["Target"] == list(iris.target_names).index(label), "Principal Component 1"],
        pca_df.loc[pca_df["Target"] == list(iris.target_names).index(label), "Principal Component 2"],
        label=label,
        alpha=0.7,
        color=color
    )
plt.title("PCA on Iris Dataset (4 features to 2 features)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Species")
plt.grid()
plt.show()
