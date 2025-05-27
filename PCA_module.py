import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading the dataset for new PCA set
df = pd.read_csv("Ready_Iraq_dataset.csv")

# splitting datat to features and class
features = [col for col in df.columns if col != 'CLASS']
X = df[features]
y = df['CLASS']

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA without reducing dimensions initially
pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)

# Inspect explained variance ratio and apply 5% cutoff
explained_variance = pca.explained_variance_ratio_
cutoff = 0.05
mask = explained_variance >= cutoff
n_keep = mask.sum()
print(f"Keeping {n_keep} principal components (>= {cutoff*100:.0f}% variance each)")

# Build reduced PCA dataset
pca_columns = [f'PC{i+1}' for i, keep in enumerate(mask) if keep]
X_pca_reduced = X_pca_full[:, mask]
df_pca = pd.DataFrame(X_pca_reduced, columns=pca_columns)
df_pca['class'] = y.values

# Optionally, show how much total variance is retained
variance_retained = explained_variance[mask].sum()
print(f"Total variance retained with {n_keep} components: {variance_retained:.2%}")

# Save the new PCA dataset with cutoff applied
df_pca.to_csv('dataset_pca_cutoff.csv', index=False)
print("PCA-reduced dataset saved to 'dataset_pca_cutoff.csv'")
