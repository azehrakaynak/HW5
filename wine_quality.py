"""""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset
url = 'winequality-red.csv'  # replace with the actual path to your CSV file
wine_data = pd.read_csv(url)

# Separate features and labels
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Plot settings
sns.set(style="whitegrid")

# Plot using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.title('PCA')

# Plot using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis')
plt.title('t-SNE')

plt.tight_layout()
plt.show()
"""""
import pandas as pd
from scipy.stats import pearsonr

# Load the dataset
url = 'winequality-red.csv'  # replace with the actual path to your CSV file
wine_data = pd.read_csv(url)

# Extract relevant columns
ph = wine_data['pH']
quality = wine_data['quality']

# Calculate Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(ph, quality)

# Output results
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-Value: {p_value}")
