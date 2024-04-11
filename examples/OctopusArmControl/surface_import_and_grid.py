import numpy as np
import os
from elastica import *
from surface_functions import (
    surface_grid,
)


# setting up test params

n_elem = 100  # number of discretized elements of the arm
base_length = 0.2  # total length of the arm
# radius_tip = 0.0012  # radius of the arm at the tip
# radius_base = 0.012  # radius of the arm at the base
radius_base = base_length / 20
radius_tip = radius_base / 10
radius = np.linspace(radius_base, radius_tip, n_elem + 1)
radius_mean = (radius[:-1] + radius[1:]) / 2
print(2 * max(radius_mean), base_length / n_elem)

mesh = Mesh(filepath=r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj")
mesh.translate(-np.array(mesh.mesh_center))
mesh.translate(-4 * radius_base - np.array([0, 0, np.min(mesh.face_centers[2])]))
# mesh.povray_mesh_export(texture_path=r"m32_Viekoda_Bay/m32_Viekoda_Bay.jpg",export_to=)
grid_size = max(2 * max(radius_mean), base_length / n_elem)
faces_grid = dict(surface_grid(mesh.faces, grid_size))

print("Adding grid info")
faces_grid["grid_size"] = grid_size


import pickle

filename = "m32_Viekoda_Bay/grid_m32_Viekoda_Bay.dat"
if os.path.exists(filename):
    os.remove(filename)
file = open(filename, "wb")
pickle.dump(faces_grid, file)
file.close()
