import numpy as np
import os
from elastica import *
from surface_functions import (
    surface_grid,
)


# setting up test params

n_elem = 100  # number of discretized elements of the arm
base_length = 0.2  # total length of the arm
env_idx = 1  # 1 for m32_Viekoda_Bay, 2 for mars-landscape

# radius_tip = 0.0012  # radius of the arm at the tip
# radius_base = 0.012  # radius of the arm at the base
radius_base = base_length / 20
radius_tip = radius_base / 10
radius = np.linspace(radius_base, radius_tip, n_elem + 1)
radius_mean = (radius[:-1] + radius[1:]) / 2
print(2 * max(radius_mean), base_length / n_elem)
if env_idx == 1:
    mesh = Mesh(filepath=r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj")
    mesh.translate(-np.array(mesh.mesh_center))
    mesh.translate(np.array([0, 0, radius_base + np.min(mesh.face_centers[2])]))
    filename = "m32_Viekoda_Bay/grid_m32_Viekoda_Bay.dat"

elif env_idx == 2:
    mesh = Mesh(filepath=r"mars-landscape/model.obj")
    mesh.translate(-np.array(mesh.mesh_center))
    mesh.rotate(axis=np.array([1.0, 0.0, 0.0]), angle=90)
    mesh.scale(np.array([10.0, 10.0, 10.0]) / np.max(mesh.mesh_scale))
    mesh.translate(np.array([0.0, 0.0, 0.0]))
    filename = "mars-landscape/grid_mars-landscape.dat"

grid_size = max(2 * max(radius_mean), base_length / n_elem)
faces_grid = dict(surface_grid(mesh.faces, grid_size))
print("Adding grid info")
faces_grid["grid_size"] = grid_size


import pickle

if os.path.exists(filename):
    os.remove(filename)
file = open(filename, "wb")
pickle.dump(faces_grid, file)
file.close()
