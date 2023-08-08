import numpy as np
from matplotlib import pyplot as plt
import os

from surface_functions import (
    import_surface_from_obj,
    surface_grid,
)

import timeit


model_path = "3D_models/obj/Mars1"

# collection of snake characteristics 
n_elem_collect = np.array([25,50])
base_length_collect = np.array([0.35,0.8])
base_radius_collect = np.array([0.009,0.009])
snake_nu_collect = np.array([5e-3,1e-2])
snake_torque_ratio_collect = np.array([30,20.0])

# select snake to run
snake_ID = 1

# setting up test params
n_elem = n_elem_collect[snake_ID]
base_length = base_length_collect[snake_ID]
base_radius = base_radius_collect[snake_ID]

max_extent = 5.0
surface_reorient = [[1, 2],[2, 1]]

facets,vertices_normals = import_surface_from_obj(
    model_path = model_path,
    max_extent = max_extent,
    max_z = 0.0,
    surface_reorient = surface_reorient,
    povray_viz = False,
)



grid_size = max(2*base_radius,base_length/n_elem)
print(grid_size)
facets_grid = dict(surface_grid(facets,grid_size))

print('Adding grid info')
facets_grid["grid_size"] = grid_size
facets_grid["model_path"] = model_path
facets_grid["max_extent"] = max_extent
facets_grid["surface_reorient"] = surface_reorient




import pickle
filename = model_path+"/grid_"+str(snake_ID)+"_"+str(max_extent)+".dat"
if os.path.exists(filename):
    os.remove(filename)
file = open(filename, "wb")
pickle.dump(facets_grid, file)
file.close()