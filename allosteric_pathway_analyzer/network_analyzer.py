import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# TEST DATA

# Define a simple list of residue labels.
residues = ["TYR", "CYS", "ARG", "TYR", "ASP"]

# Create one frame of center-of-mass (COM) data.
# Each row is a residue and the columns are x, y, z coordinates.
com_matrix_frame0 = np.array([
    [1.0, 0.0, -2.0],
    [2.0, 0.0, 2.0],
    [0.5, 1.0, 2.0],
    [0.3, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])

# Create one frame of interaction matrices (symmetric matrices with zeros on the diagonal).
elec_frame0 = np.array([
    [0, 1, 2, 3, 4],
    [1, 0, 5, 6, 7],
    [2, 5, 0, 8, 9],
    [3, 6, 8, 0, 10],
    [4, 7, 9, 10, 0],
])

vdw_frame0 = np.array([
    [0, 2, 3, 4, 5],
    [2, 0, 6, 7, 8],
    [3, 6, 0, 9, 10],
    [4, 7, 9, 0, 11],
    [5, 8, 10, 11, 0],
])

hbond_frame0 = np.array([
    [0,   0.5, 0.6, 0.7, 0.8],
    [0.5, 0,   0.9, 1.0, 1.1],
    [0.6, 0.9, 0,   1.2, 1.3],
    [0.7, 1.0, 1.2, 0,   1.4],
    [0.8, 1.1, 1.3, 1.4, 0  ],
])

# Structure network_components as expected:
# - [0]: residues (list)
# - [1]: list of COM matrices (one per frame)
# - [2]: a two-element list: first element is a list of electrostatic interaction matrices (one per frame),
#        second element (e.g., transition probabilities) is left as None for now.
# - [3] and [4]: similarly for vdw and hbond interactions.
network_components = [
    residues,
    [com_matrix_frame0],
    [[elec_frame0], None],
    [[vdw_frame0], None],
    [[hbond_frame0], None],
]

class NetworkAnalyzer:
    def __init__(self, network_directory_path: str) -> None:
        # For testing purposes, we use the sample network_components defined above.
        network_components = [
            residues,
            [com_matrix_frame0],
            [[elec_frame0], None],
            [[vdw_frame0], None],
            [[hbond_frame0], None],
        ]
        
        # Set up public instance attributes.
        self.residues = network_components[0]
        self.com_matrices = network_components[1]
        self.elec_matrices = network_components[2][0]  # interactions; transition-probabilities ignored.
        self.vdw_matrices  = network_components[3][0]
        self.hbond_matrices= network_components[4][0]
        
        self.number_residues = len(self.residues)
        self.number_matrices = len(self.com_matrices)
    
        self.network = nx.Graph()
    
        # Add nodes for each residue
        for i in range(self.number_residues):
            self.network.add_node(i, label=f"{self.residues[i]}-{i+1}")
    
    def visualize_frame(self, frame_num, interaction_type: str = None):
        # Adjust for 1-indexed frame numbers
        frame_num -= 1
        row, col = np.triu_indices(self.number_residues, k=1)
    
        if interaction_type == "elec":
            interaction_matrix = self.elec_matrices[frame_num]
            edges = list(zip(row, col, interaction_matrix[row, col]))
            self.network.add_weighted_edges_from(edges)
        elif interaction_type == "vdw":
            interaction_matrix = self.vdw_matrices[frame_num]
            edges = list(zip(row, col, interaction_matrix[row, col]))
            self.network.add_weighted_edges_from(edges)
        elif interaction_type == "hbond":
            interaction_matrix = self.hbond_matrices[frame_num]
            edges = list(zip(row, col, interaction_matrix[row, col]))
            self.network.add_weighted_edges_from(edges)
        else:
            # If no interaction type is provided, we don't add any network edges.
            interaction_matrix = None
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Use the COM matrix of the current frame for node positions.
        for node, (x, y, z) in enumerate(self.com_matrices[frame_num]):
            ax.scatter(x, y, z, s=100, color='blue')
            ax.text(x, y, z, f"{self.residues[node]} {node+1}", size=10, color='black')
    
        # Draw all network edges in grey (if interaction_matrix is defined)
        if interaction_matrix is not None:
            for (i, j, d) in self.network.edges(data=True):
                xs = [self.com_matrices[frame_num][i][0], self.com_matrices[frame_num][j][0]]
                ys = [self.com_matrices[frame_num][i][1], self.com_matrices[frame_num][j][1]]
                zs = [self.com_matrices[frame_num][i][2], self.com_matrices[frame_num][j][2]]
                ax.plot(xs, ys, zs, linewidth=d["weight"], color='grey')
    
        plt.show()

    def visualize_path(self, frame_num, path):
        # Adjust for 1-indexed frame numbers
        frame_num -= 1  
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Draw nodes using the COM positions for the given frame.
        for node, (x, y, z) in enumerate(self.com_matrices[frame_num]):
            ax.scatter(x, y, z, s=100, color='blue')
            ax.text(x, y, z, f"{self.residues[node]} {node+1}", size=10, color='black')
    
        # For demonstration, draw network edges using the electrostatic (elec) interaction matrix.
        row, col = np.triu_indices(self.number_residues, k=1)
        interaction_matrix = self.elec_matrices[frame_num]
        edges = list(zip(row, col, interaction_matrix[row, col]))
        self.network.add_weighted_edges_from(edges)
        for (i, j, d) in self.network.edges(data=True):
            xs = [self.com_matrices[frame_num][i][0], self.com_matrices[frame_num][j][0]]
            ys = [self.com_matrices[frame_num][i][1], self.com_matrices[frame_num][j][1]]
            zs = [self.com_matrices[frame_num][i][2], self.com_matrices[frame_num][j][2]]
            ax.plot(xs, ys, zs, linewidth=d["weight"], color='grey')
    
        # Process the path:
        # If the first element of the path is a tuple, assume each tuple is (residue, frame)
        # and convert the residue to 0-indexed.
        if isinstance(path[0], tuple):
            path_nodes = [t[0] - 1 for t in path]
        else:
            path_nodes = path
    
        # Build edges from the path.
        path_edges = list(zip(path_nodes, path_nodes[1:]))
        path_line_objects = []
        for (i, j) in path_edges:
            try:
                weight = self.network[i][j]["weight"]
            except KeyError:
                weight = 1  # fallback if the edge isn't in the network
            xs = [self.com_matrices[frame_num][i][0], self.com_matrices[frame_num][j][0]]
            ys = [self.com_matrices[frame_num][i][1], self.com_matrices[frame_num][j][1]]
            zs = [self.com_matrices[frame_num][i][2], self.com_matrices[frame_num][j][2]]
            # Draw the red edge with alpha=0 (invisible initially)
            line, = ax.plot(xs, ys, zs, linewidth=weight, color='red', alpha=0)
            path_line_objects.append(line)
    
        ax.set_title("Allosteric Pathway Propagation")
    
        # Define update function for animation: reveal one red edge per frame.
        def update(frame):
            if frame < len(path_line_objects):
                path_line_objects[frame].set_alpha(1)
            return path_line_objects
    
        ani = animation.FuncAnimation(fig, update, frames=len(path_line_objects), interval=1000, blit=False)
        plt.show()

# Example usage:
na = NetworkAnalyzer("")
# Visualize frame 1 with electrostatic interactions.
# na.visualize_frame(1, "elec")
# Define a custom path.
# Here the path is given as a list of tuples (residue, frame) using 1-indexed residue numbers.
path = [(1, 1), (3, 1), (2, 1), (5, 1)]
na.visualize_path(1, path)


# Note, this is still basically scratch work, a lot of optimisation, testing, and refinement needs to be done