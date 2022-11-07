import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.set_printoptions(precision=2,suppress=True)

df = pd.read_csv('spiraldatensatz.csv')

print(df.shape)
print(df.head())

plt.figure(figsize=(20, 20))

ax1 = plt.subplot(311)
ax1.scatter(df.x1, df.x2, s=1)
ax1.set_aspect(1)

ax2 = plt.subplot(312)
ax2.scatter(df.x1, df.x3, s=1)
ax2.set_aspect(1)

ax3 = plt.subplot(313)
ax3.scatter(df.x2, df.x3, s=1)
ax3.set_aspect(1)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection="3d")
df_seg = df
ax.scatter3D(df_seg.x1, df_seg.x2, df_seg.x3)
np.cov(df.to_numpy().T)
pca3 = PCA()
df_t = pca3.fit_transform(df)
print(df_t.shape)

plt.figure(figsize=(10,10))
plt.scatter(df_t[:,0], df_t[:,1], s=1, c=df_t[:,2], cmap='cool')
plt.gca().set_aspect(1)
plt.colorbar()
vars(pca3)
plt.figure(figsize=(20, 20))

ax1 = plt.subplot(311)
ax1.hist(df_t[:,0])

ax2 = plt.subplot(312)
ax2.hist(df_t[:,1])

ax3 = plt.subplot(313)
ax3.hist(df_t[:,2])

pca2 = PCA(2)
pca2.fit(df)

plt.show()
print("--------------------------------")
print("pca2.components_")
print(pca2.components_)
print("--------------------------------")
print("pca3.components_")
print(pca3.components_)
print("--------------------------------")
print("pca2.singular_values_")
print(pca2.singular_values_)
print("--------------------------------")
print("pca2.explained_variance_")
print(pca2.explained_variance_)
print("--------------------------------")
print("pca2.explained_variance_ratio_")
print(pca2.explained_variance_ratio_)





