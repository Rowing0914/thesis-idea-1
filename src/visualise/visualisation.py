import numpy as np

weights = np.load("./data/Hopper-v2/policy_weight.npy")
scores = np.load("./data/Hopper-v2/scores.npy")
print(weights.shape, scores.shape)

from sklearn.manifold import TSNE
embedding = TSNE(n_components=2).fit_transform(weights)
print(embedding.shape)

import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('RdYlBu')
plt.scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap=cm)
plt.colorbar()
plt.show()
