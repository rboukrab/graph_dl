import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic 2D data
np.random.seed(42)
mean = [2, 5]
cov = [[3, 2], [2, 2]]  # covariance matrix
X = np.random.multivariate_normal(mean, cov, 200)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Get eigenvectors and eigenvalues
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# Plot data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.4)

# Plot principal components as vectors
origin = pca.mean_
for length, vector in zip(eigenvalues, eigenvectors):
    v = vector * np.sqrt(length) * 3  # scale for visibility
    plt.quiver(*origin, v[0], v[1], angles='xy', scale_units='xy',
               scale=1, color='red', width=0.005)

plt.xlabel("x")
plt.ylabel("X2")
plt.title("Scatter Plot with Principal Components")
plt.axis("equal")
plt.show()
