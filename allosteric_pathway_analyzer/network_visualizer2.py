import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
import matplotlib.animation as animation

# Define a 5x5 symmetric interaction matrix (diagonals are zero)
interaction_matrix = np.array([
    [0, 3, 1, 2, 4],
    [3, 0, 2, 1, 5],
    [1, 2, 0, 3, 2],
    [2, 1, 3, 0, 4],
    [4, 5, 2, 4, 0]
])

# Create an undirected graph from the matrix
G = nx.Graph()
num_residues = interaction_matrix.shape[0]

# Add nodes for each residue
for i in range(num_residues):
    G.add_node(i, label=f"Residue {i+1}")

# Add edges with weights corresponding to interaction strengths
for i in range(num_residues):
    for j in range(i + 1, num_residues):
        weight = interaction_matrix[i, j]
        G.add_edge(i, j, weight=weight)

# Generate random 3D positions for each node
np.random.seed(42)
pos = {i: np.random.rand(3) for i in range(num_residues)}

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw nodes as scatter points and add labels
for node, (x, y, z) in pos.items():
    ax.scatter(x, y, z, s=100, color='lightblue')
    ax.text(x, y, z, f"Residue {node+1}", size=10, color='black')

# Draw all edges in grey
for (i, j, d) in G.edges(data=True):
    xs = [pos[i][0], pos[j][0]]
    ys = [pos[i][1], pos[j][1]]
    zs = [pos[i][2], pos[j][2]]
    ax.plot(xs, ys, zs, linewidth=d["weight"], color='grey')

# Define a custom longer path:
# Residue 1 (index 0) → Residue 3 (index 2) → Residue 2 (index 1) → Residue 4 (index 3) → Residue 5 (index 4)
path_nodes = [0, 2, 1, 3, 4]
path_edges = list(zip(path_nodes, path_nodes[1:]))

# Overlay the path edges with red lines that are initially invisible (alpha=0)
path_line_objects = []
for (i, j) in path_edges:
    weight = G[i][j]["weight"]
    xs = [pos[i][0], pos[j][0]]
    ys = [pos[i][1], pos[j][1]]
    zs = [pos[i][2], pos[j][2]]
    line, = ax.plot(xs, ys, zs, linewidth=weight, color='red', alpha=0)
    path_line_objects.append(line)

# Define update function for animation: reveal one edge per frame
def update(frame):
    if frame < len(path_line_objects):
        path_line_objects[frame].set_alpha(1)
    return path_line_objects

# Create the animation: each frame reveals one more edge in the path
ani = animation.FuncAnimation(fig, update, frames=len(path_line_objects), interval=1000, blit=False)

ax.set_title("3D Residue Interaction Network\n(Animating a Longer Path)")
plt.show()