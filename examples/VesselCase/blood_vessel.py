import os
import numpy as np
import sys
import argparse
import pickle
from CustomFrictionSurface import (
CustomFrictionalSurfaceGrid,
)

from surface_functions import (
    import_surface_from_obj,
    create_surface_from_parameterization,
    surface_grid,
    calculate_facet_normals_centers_areas,
)

from post_processing import (
    plot_video_with_surface
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
final_time = 4
damping_constant = 3e-2

# collection of wire characteristics 
n_elem = 100
base_length = 0.4
base_radius = (5)/1000

time_step = 0.025 * base_length / n_elem
total_steps = int(final_time / time_step)
rendering_fps = 100
step_skip = int(1.0 / (rendering_fps * time_step))


# setting up test params
max_extent = 2
# end = max_extent*np.array([-0.06/2,1.74/2,0])
end = max_extent*np.array([0.35,-0.033,-0.34])/2
direction = np.array([-1.0, -1.0, -1.0])
direction /= np.linalg.norm(direction)
start = end - base_length*direction
normal = np.array([-1.0, 0.0, 1.0])
normal /= np.linalg.norm(normal)
density = 4000
E = 1e5
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
    youngs_modulus = E,
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

A = np.pi*base_radius**2
class ConstantVelocityPush(NoForces):

    def __init__(self, direction, speed,stop_time):
        """
        """
        super(ConstantVelocityPush, self).__init__()
        self.direction = direction
        self.speed = speed
        self.stop_time = stop_time

    def apply_forces(self, system: SystemType, time=0.0):
        if time<self.stop_time:
            self.compute_forces(
                system.velocity_collection,
                self.direction,
                self.speed,
            )
        else:
            self.compute_forces(
                system.velocity_collection,
                self.direction,
                0.0,
            )

    @staticmethod
    @njit(cache=True)
    def compute_forces(
        velocity_collection, direction, speed,
    ):
        """
        """
        for i in range(75):
            velocity_collection[..., i] = direction*speed


# vessel_sim.add_forcing_to(shearable_rod).using(
#     EndpointForces, start_force=0.1*E*A*direction,end_force = np.zeros(3,), ramp_up_time = time_step,
# )
vessel_sim.add_forcing_to(shearable_rod).using(
    ConstantVelocityPush, direction=direction, speed = 0.1,stop_time = 4
)

# vessel_sim.constrain(shearable_rod).using(
#     GeneralConstraint,
#     constrained_position_idx=(0,),
#     constrained_director_idx=(0,),
#     translational_constraint_selector=np.array([True, True, False]),
#     rotational_constraint_selector=np.array([True, True, True]),)

povray_viz  = False
    
# select obj 
model_path = "3D_models/obj/vessel3"
surface_reorient = [[1, 2],[2, 1]]
facets,facet_vertex_normals = import_surface_from_obj(
    model_path = model_path,
    max_extent = max_extent,
    max_z = 0.0,
    surface_reorient = surface_reorient,
    normals_invert= True,
    povray_viz = povray_viz,
)
facets_normals,facets_centers,facet_areas = calculate_facet_normals_centers_areas(facets = facets,facet_vertex_normals = facet_vertex_normals)
n_facets = facets.shape[-1]
side_vectors = np.zeros((3,3,n_facets)) #coords,sides,faces
side_vectors[:,0,:] = facets[:,1,:] - facets[:,0,:] #AB
side_vectors[:,1,:] = facets[:,2,:] - facets[:,0,:] #AC
side_vectors[:,2,:] = facets[:,2,:] - facets[:,1,:] #BC
import_grid = True #set to True if you have a grid already made using surface_import_and_grid.py
# create grid
grid_size = np.sqrt((2*base_radius)**2+(base_length/n_elem)**2)
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
kinetic_mu_array = np.array([0,0,0])
slip_velocity_tol = 1e-8
facet_pattern_idx = (facets_centers[2,:]>0.1)
mesh_contact_callback = defaultdict(list)
vessel_sim.add_forcing_to(shearable_rod).using(
    CustomFrictionalSurfaceGrid,
    k=1,
    nu=1e-1,
    facets=facets,
    facets_centers = facets_centers,
    facets_normals = facets_normals,
    side_vectors = side_vectors,
    facets_grid=facets_grid,
    grid_size=grid_size,
    slip_velocity_tol=slip_velocity_tol,
    static_mu_array=static_mu_array,
    kinetic_mu_array=kinetic_mu_array,
    kinetic_mu_sideways_pattern= 1.0 * mu,
    facet_pattern_idx = facet_pattern_idx,
    gamma = 0.1,
    step_skip = step_skip,
    callback_params = mesh_contact_callback,
    callback = True,
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
            self.callback_params["com_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["com"].append(
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


# plotting the videos
filename_video = "vessel.mp4"
plot_video_with_surface(
    [pp_list],
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-5/100, 5/100],
    y_limits=[-5/100, 5/100],
    z_limits=[-5/100, 5/100],
)



filename = "vessel.dat"
file = open(filename, "wb")
pickle.dump(pp_list, file)
file.close()

