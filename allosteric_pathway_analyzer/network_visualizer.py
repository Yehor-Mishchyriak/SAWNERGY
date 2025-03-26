# from subprocess import run

# top = "/home/yehor/Desktop/AllostericPathwayAnalyzer/untracked/MD_data/p53_WT_nowat.prmtop"
# traj = "/home/yehor/Desktop/AllostericPathwayAnalyzer/untracked/MD_data/p53_WT_md1000_str.nc"
# cpptraj = "/home/yehor/miniforge3/envs/AmberTools23/bin/cpptraj"

# residue_num = 1
# COM = (
#     f"echo 'parm {top}\n"
#     f"trajin {traj}\n"
#     f"vector com_res{residue_num} center out res{residue_num}_com.dat :{residue_num}\n"
#     f"run' | {cpptraj}"
# )
# run(COM, check=True, shell=True) # -> the output is the coordnates for the center of mass of each residue in each individual frame

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting

# Example data: 10 frames, 5 residues, 3 coordinates (x, y, z)
# In practice, replace this with your actual COM data
data = np.random.random((10, 5, 3))  

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for each residue (5 residues)
colors = ['r', 'g', 'b', 'c', 'm']

# For each residue, plot its COM coordinates over the frames
for res in range(data.shape[1]):  # Loop over residues (columns)
    # data[:, res, 0] gives x, data[:, res, 1] gives y, and data[:, res, 2] gives z coordinates over frames
    ax.scatter(data[:, res, 0], data[:, res, 1], data[:, res, 2],
               color=colors[res], label=f"Residue {res+1}")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Center-of-Mass of Residues Over Frames")
ax.legend()

plt.show()
