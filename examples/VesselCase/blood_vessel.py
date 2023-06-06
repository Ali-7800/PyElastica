import os
import numpy as np
import sys
import argparse
import pickle
from CustomFrictionSurface import (
CustomInteractionSurfaceGrid,
CustomFrictionalSurfaceGrid,
)

from surface_functions import (
    import_surface_from_obj,
    create_surface_from_parameterization,
    surface_grid,
)

sys.path.append("../../")
sys.path.append("../../../../")
sys.path.append("../../../")


from elastica import *
from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik
from elastica.external_forces import inplace_addition




class VesselSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass



# Initialize the simulation class
vessel_sim = VesselSimulator()


# Simulation parameters
final_time = 9
damping_constant = 3e-3
time_step = 1e-4
total_steps = int(final_time / time_step)
rendering_fps = 100
step_skip = int(1.0 / (rendering_fps * time_step))

# collection of wire characteristics 
n_elem = 50
base_length = 0.35
base_radius = 0.35*0.011


# setting up test params
start = np.array([-0.06,1.74,0.35])
direction = np.array([0.0, 0.0, -1.0])
normal = np.array([0.0, 1.0, 0.0])
density = 1000
E = 1e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,
    E,
    shear_modulus=shear_modulus,
)

vessel_sim.append(shearable_rod)



# use linear damping with constant damping ratio
vessel_sim.dampen(shearable_rod).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)

gravitational_acc = -9.80665




class AlternatingForces(NoForces):

    def __init__(self, accel_mag):


        super(AlternatingForces, self).__init__()
        self.accel_mag = accel_mag

    def apply_forces(self, system, time=0.0):
        self.compute_end_point_forces(
            system.external_forces,
            self.accel_mag,
            time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, accel_mag, time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        Returns
        -------

        """
        if time%3<1.0:
            accel_vector = accel_mag*np.array([0,0,-1])
        elif time%3<2.0:
            accel_vector = accel_mag*np.array([0,-1,0])
        else:
            accel_vector = accel_mag*np.array([1,0,0])

        external_forces[..., -1] += accel_vector



vessel_sim.add_forcing_to(shearable_rod).using(
    AlternatingForces, accel_mag=0.3
)

povray_viz  = True
    
# select obj 
model_path = "3D_models/obj/vessel1"
max_extent = 2.0
surface_reorient = [[1, 2],[2, 1]]
facets,facet_vertex_normals = import_surface_from_obj(
    model_path = model_path,
    max_extent = max_extent,
    max_z = 0.0,
    surface_reorient = surface_reorient,
    normals_invert= True,
    povray_viz = povray_viz,
)

import_grid = True #set to True if you have a grid already made using surface_import_and_grid.py
# create grid
grid_size = max(2*base_radius,base_length/n_elem)
if not import_grid:
    facets_grid = surface_grid(facets,grid_size)
else:
    filename = model_path+"/vessel_grid.dat"
    with open(filename, "rb") as fptr:
        facets_grid = pickle.load(fptr)

    assert facets_grid["grid_size"] == grid_size, "imported grid has different grid size than for the current rod"
    assert facets_grid["model_path"] == model_path, "imported grid is for a different model"
    print(facets_grid["model_path"])
    assert facets_grid["max_extent"] == max_extent, "imported grid is for a different extent"
    assert facets_grid["surface_reorient"] == [[1, 2],[2, 1]], "imported grid is for a different surface orientation"

static_mu_array = np.zeros(3,)
froude = 0.1
mu = base_length / (2 * 2 * np.abs(gravitational_acc) * froude)
kinetic_mu_array = np.array([0,1.5*mu,2*mu])
slip_velocity_tol = 1e-8


vessel_sim.add_forcing_to(shearable_rod).using(
    CustomFrictionalSurfaceGrid,
    k=2e1,
    nu=1e-1,
    facets=facets,
    facet_vertex_normals = facet_vertex_normals,
    facets_grid=facets_grid,
    grid_size=grid_size,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
    gamma = 0.1,
)



# Add call backs
class VesselCallBack(CallBackBaseClass):
    """
    Call back function for continuum rod
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            self.callback_params["radius"].append(
                system.radius.copy()
            )
            self.callback_params["velocity"].append(
                system.velocity_collection.copy()
            )
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass()
            )
            self.callback_params["curvature"].append(system.kappa.copy())

            return

pp_list = defaultdict(list)
vessel_sim.collect_diagnostics(shearable_rod).using(
    VesselCallBack, step_skip=step_skip, callback_params=pp_list
)

vessel_sim.finalize()

timestepper = PositionVerlet()
integrate(timestepper, vessel_sim, final_time, total_steps)




filename = "vessel.dat"
file = open(filename, "wb")
pickle.dump(pp_list, file)
file.close()

