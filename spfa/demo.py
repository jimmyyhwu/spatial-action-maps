import numpy as np
from PIL import Image
import spfa

# Create test map
num_rows = 250
num_cols = 350
source_i = 5
source_j = 100
source = (source_i, source_j)
free_space_map = np.ones([num_rows, num_cols], dtype=bool)
free_space_map[200:202, 1:300] = False

# Run SPFA
dists, parents = spfa.spfa(free_space_map, source)

# Highlight the path to a target
parents_ij = np.stack((parents // parents.shape[1], parents % parents.shape[1]), axis=2)
parents_ij[parents < 0, :] = (-1, -1)
target_i = 245
target_j = 50
i, j = target_i, target_j
while not (i == source_i and j == source_j):
    i, j = parents_ij[i, j]
    if i + j < 0:
        break
    dists[i][j] = 2 * 255

# Visualize the output
Image.fromarray((0.5 * dists).astype(np.uint8)).save('output.png')
