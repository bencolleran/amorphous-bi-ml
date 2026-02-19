import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

train=np.load(f"{PROJECT_ROOT}/soaps_train.npy")
lammps=np.load(f"{PROJECT_ROOT}/soaps_20260210_102745.npy")
a=np.load(f"{PROJECT_ROOT}/soaps_20260210_102745.npy")
b=np.load(f"{PROJECT_ROOT}/soaps_20260210_130003.npy")
c=np.load(f"{PROJECT_ROOT}/soaps_20260210_143203.npy")
d=np.load(f"{PROJECT_ROOT}/soaps_20260210_154820.npy")
X=np.vstack([a,b,c,d])
#X=np.vstack([train,lammps])# X shape = (N, 859)


a=np.load("soaps_20260212_152158.npy")#0.8
b=np.load("soaps_20260212_152203.npy")#1.0
c=np.load("soaps_20260212_152204.npy")#0.9
d=np.load("soaps_20260212_152218.npy")#1.1
e=np.load("soaps_20260212_152219.npy")#1.2
new_structures=np.vstack([a,b,c,d,e])


# # Compute PCA → 2 components
# X=new_structures
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# # Plot
# fig, ax = plt.subplots()
# ax.scatter(X_pca[:, 0], X_pca[:, 1], s=10)
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_title("PCA of SOAP vectors")
# fig.savefig(f"{PROJECT_ROOT}/images/pca_lammps_plot_new_structures.png")

# stack both datasets
A=train
B=new_structures
X=np.vstack([A,B])
# fit PCA on combined data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# split back
N1 = A.shape[0]
A_pca = X_pca[:N1]
B_pca = X_pca[N1:]

fig, ax = plt.subplots()
ax.scatter(A_pca[:, 0], A_pca[:, 1], label="training data",s=5,alpha=0.6)
ax.scatter(B_pca[:, 0], B_pca[:, 1], label="lammps data",s=5,alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

highlight_indices = [3205, 3204, 3226, 3288, 2410,   2, 3208, 2404, 2427, 2564]
# for idx in highlight_indices:
#     x = B_pca[idx, 0]
#     y = B_pca[idx, 1]

#     ax.annotate(
#         "",                          # no text, just arrow
#         xy=(x, y),                   # arrow tip (the data point)
#         xytext=(x + 0.5, y + 0.5),   # where arrow starts (offset)
#         arrowprops=dict(
#             arrowstyle="->",
#             color="black",
#             lw=1.5
#         )
#     )

ax.scatter(
    B_pca[highlight_indices, 0],
    B_pca[highlight_indices, 1],
    color="black",
    s=30,
    zorder=5,
)

# Add numbers (0–9 corresponding to position in highlight list)
for i, idx in enumerate(highlight_indices):
    x, y = B_pca[idx]

    ax.annotate(
        str(i),                     # number in highlight list
        xy=(x, y),
        xytext=(5, 5),              # offset in pixels
        textcoords="offset points", # important!
        fontsize=9,
        color="black",
        weight="bold",
    )

ax.legend()
fig.savefig(f"{PROJECT_ROOT}/images/pca_plot_color_new_structures.png")
