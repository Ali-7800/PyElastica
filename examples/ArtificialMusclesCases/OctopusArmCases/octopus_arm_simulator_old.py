import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm

from typing import Dict, Sequence
from numba import njit
import sys

from elastica import *


# from connect_straight_rods import *
from examples.ArtificialMusclesCases.post_processing import (
    plot_video_with_surface,
)
from examples.ArtificialMusclesCases import *

from examples.ArtificialMusclesCases.muscle.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)


class OctopusArmCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


# Add callback functions for plotting position of the rod later on
class RodCallBack(CallBackBaseClass):
    """ """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["com_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            total_energy = (
                system.compute_translational_energy()
                + system.compute_rotational_energy()
                + system.compute_bending_energy()
                + system.compute_shear_energy()
            )
            self.callback_params["total_energy"].append(total_energy)
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["internal_force"].append(system.internal_forces.copy())
            self.callback_params["external_force"].append(system.external_forces.copy())

            return


def conical_helix_length(start_radius, cone_slope, height, n_turns_per_length):
    k = 2 * np.pi * n_turns_per_length
    m = cone_slope
    end_radius = start_radius + m * height
    a = k * start_radius / np.sqrt((m ** 2) + 1)
    b = k * end_radius / np.sqrt((m ** 2) + 1)
    length_at_a = (a * np.sqrt(a ** 2 + 1) + np.arcsinh(a)) / 2
    length_at_b = (b * np.sqrt(b ** 2 + 1) + np.arcsinh(b)) / 2
    length = (m ** 2 + 1) * (length_at_b - length_at_a) / (m * k)
    return length


post_processing_dict_list = []


octopus_arm_sim = OctopusArmCase()

final_time = 30
length_scale = 1e-3
mass_scale = 1e-3

n_muscles = 8
n_rows = 8

taper_ratio = 1 / 9  # 1/12
backbone_start_radius = 12 * length_scale  # 12
backbone_length = 200 * length_scale  # 200
muscle_length_estimate = 26 * length_scale
muscle_height = backbone_length / n_rows
n_turns_per_backbone_length = 1 / (backbone_length)

n_muscle_turns_per_length = 0.732 / length_scale

link_scale = 1
initial_link_per_length = link_scale * 0.487125 / length_scale  # turns per unit length
E_scale = 1
n_elem_per_turn = 12
n_elem_backbone = 200
n_elem_muscle = 200

E = 1925
Thompson_model = False
Contraction = True
Self_Contact = False
save_data = True
current_path = os.getcwd()

dt = 0.3 * muscle_length_estimate / n_elem_muscle
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rest of the rod parameters and construct rod
base_radius = (0.74 / 2) * length_scale
helix_radius = (2.12 / 2) * length_scale
room_temperature = 25
end_temperature = 120
thermal_expansion_coeficient = 7e-5
youngs_modulus_coefficients = [
    2.26758447119,
    -0.00996645676489,
    0.0000323219668553,
    -3.8696662364 * 1e-7,
    -6.3964732027 * 1e-7,
    2.0149695202 * 1e-8,
    -2.5861167614 * 1e-10,
    1.680136396 * 1e-12,
    -5.4956153529 * 1e-15,
    7.2138065668 * 1e-18,
]  # coefficients of youngs modulus interpolation polynomial


base_area = np.pi * base_radius ** 2
I = np.pi / 4 * base_radius ** 4
volume = base_area * backbone_length
nu = 3e-3
relaxationNu = 0.0

poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

youngs_modulus_backbone = 1
poisson_ratio_backbone = 0.5
shear_modulus_backbone = youngs_modulus_backbone / (poisson_ratio_backbone + 1.0)
backbone_density = 920

direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
binormal = np.cross(direction, normal)
start = np.zeros(
    3,
)

backbone_radius = backbone_start_radius * np.linspace(1, taper_ratio, n_elem_backbone)
backbone_volume = (
    np.pi * (backbone_radius[0] ** 2 - backbone_radius[-1] ** 2) * backbone_length / 3
)


backbone_rod = CosseratRod.straight_rod(
    n_elem_backbone,
    start,
    direction,
    normal,
    backbone_length,
    backbone_radius,
    backbone_density,
    youngs_modulus=youngs_modulus_backbone,
    shear_modulus=shear_modulus_backbone,
)

octopus_arm_sim.append(backbone_rod)

# Add damping
octopus_arm_sim.dampen(backbone_rod).using(
    AnalyticalLinearDamper,
    damping_constant=1e-5,
    time_step=dt,
)


octopus_arm_sim.constrain(backbone_rod).using(
    GeneralConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
    translational_constraint_selector=np.array(
        [True, True, True]
    ),  # np.array([True, True, False]),
    rotational_constraint_selector=np.array([True, True, True]),
)


backbone_post_processing_dict = defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
octopus_arm_sim.collect_diagnostics(backbone_rod).using(
    RodCallBack,
    step_skip=step_skip,
    callback_params=backbone_post_processing_dict,
)
post_processing_dict_list.append(backbone_post_processing_dict)


youngs_modulus_inner_spine = 50
poisson_ratio_inner_spine = 0.5
shear_modulus_inner_spine = youngs_modulus_inner_spine / (poisson_ratio_backbone + 1.0)
inner_spine_density = 2000

direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
binormal = np.cross(direction, normal)
start = np.zeros(
    3,
)

inner_spine_radius = backbone_radius[-1]
inner_spine_volume = np.pi * (inner_spine_radius ** 2) * backbone_length / 3


inner_spine_rod = CosseratRod.straight_rod(
    n_elem_backbone,
    start,
    direction,
    normal,
    backbone_length,
    inner_spine_radius,
    inner_spine_density,
    youngs_modulus=youngs_modulus_inner_spine,
    shear_modulus=shear_modulus_inner_spine,
)

octopus_arm_sim.append(inner_spine_rod)

# Add damping
octopus_arm_sim.dampen(inner_spine_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)


for elem in range(n_elem_backbone):
    octopus_arm_sim.connect(
        first_rod=inner_spine_rod,
        second_rod=backbone_rod,
        first_connect_idx=elem,
        second_connect_idx=elem,
    ).using(
        ParallelJointInterior,
        k=shear_modulus_inner_spine,
        nu=1e-5,
        k_repulsive=0,
    )


inner_spine_post_processing_dict = defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
octopus_arm_sim.collect_diagnostics(inner_spine_rod).using(
    RodCallBack,
    step_skip=step_skip,
    callback_params=inner_spine_post_processing_dict,
)
post_processing_dict_list.append(inner_spine_post_processing_dict)


muscle_rods = {}
start_kappa = {}
start_sigma = {}
end_kappa = {}
end_sigma = {}
start_bend_matrix = {}
start_shear_matrix = {}
start_mass_second_moment_of_inertia = {}
start_inv_mass_second_moment_of_inertia = {}


slope_angle = np.arctan2(backbone_length, backbone_radius[0] - backbone_radius[-1])
backbone_slope = (backbone_radius[-1] - backbone_radius[0]) / backbone_length
taper_slope_list = [backbone_slope, 0]


rows = range(n_rows)
muscles = range(n_muscles)
n_elem_per_row = int(muscle_height * n_elem_backbone / backbone_length)

all_muscles_coords = []
front_diamonds_coords = []
back_diamonds_coords = []
right_diamonds_coords = []
left_diamonds_coords = []

for diamond in range(int(n_rows / 2)):
    front_diamonds_coords += [
        (2 * diamond, 0, "CCW"),
        (2 * diamond, 0, "CW"),
        (2 * diamond + 1, 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, -2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    right_diamonds_coords += [
        (2 * diamond, np.pi / 2, "CCW"),
        (2 * diamond, np.pi / 2, "CW"),
        (2 * diamond + 1, np.pi / 2 + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, np.pi / 2 - 2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    back_diamonds_coords += [
        (2 * diamond, np.pi, "CCW"),
        (2 * diamond, np.pi, "CW"),
        (2 * diamond + 1, np.pi + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, np.pi - 2 * np.pi / n_rows, "CCW"),
    ]

for diamond in range(int(n_rows / 2)):
    left_diamonds_coords += [
        (2 * diamond, -np.pi / 2, "CCW"),
        (2 * diamond, -np.pi / 2, "CW"),
        (2 * diamond + 1, -np.pi / 2 + 2 * np.pi / n_rows, "CW"),
        (2 * diamond + 1, -np.pi / 2 - 2 * np.pi / n_rows, "CCW"),
    ]

all_muscles_coords = (
    front_diamonds_coords[:14]
    + right_diamonds_coords[:14]
    + back_diamonds_coords[:14]
    + left_diamonds_coords[:14]
)


cw_twist_coords = []
# for row in range(n_rows-1):
#     for i in range(4):
#         theta = -np.pi*row/4 + np.pi*i/2
#         cw_twist_coords.append((row,theta,"CW"))
for coord in all_muscles_coords:
    if coord[2] == "CW":
        cw_twist_coords.append(coord)


bottom_to_top_sequence = []
for diamond in range(int(n_rows / 2)):
    if diamond == int(n_rows / 2) - 1:
        bottom_to_top_sequence += 2 * [(28, 1, 0.1)]
    else:
        bottom_to_top_sequence += 4 * [(7 * diamond, 1, 0.1)]


quick_bend_activation = 12 * [(0, 3, 0.1)] + 2 * [(0, 3, 0.1)]
quick_twist_activation = len(cw_twist_coords) * [(0, 3, 0.05)]  # + [(0,3,1)]

spiral_diamonds_coords = (
    front_diamonds_coords[0:4]
    + right_diamonds_coords[4:8]
    + back_diamonds_coords[8:12]
    + left_diamonds_coords[12:14]
)

included_muscles = all_muscles_coords  # spiral_diamonds_coords
activation_group = right_diamonds_coords[:14]
activation_startTime_untwistTime_force = (
    quick_bend_activation  # bottom_to_top_sequence #quick_twist_activation#
)

for muscle_coords in included_muscles:
    row = muscle_coords[0]
    theta = muscle_coords[1]
    orientation = muscle_coords[2]
    if orientation == "CCW":
        CCW = (True, True)
        link_sign = -1
    else:
        CCW = (False, False)
        link_sign = 1

    muscle_position_radius = (
        backbone_radius[0]
        + backbone_slope * row * muscle_height
        + (helix_radius / np.sin(slope_angle))
    )

    muscle_length = conical_helix_length(
        muscle_position_radius,
        backbone_slope,
        muscle_height,
        n_turns_per_backbone_length,
    )
    divide = 80.63 * length_scale / muscle_length
    n_muscle_turns = n_muscle_turns_per_length * muscle_length
    n_muscle_turns_per_height = int(n_muscle_turns) / muscle_height

    turns_per_length_list = [n_turns_per_backbone_length, n_muscle_turns_per_height]

    start_radius_list = [muscle_position_radius, helix_radius]
    # Helix rod structure
    start_position_of_helix = start + row * muscle_height * direction
    direction_helical_rod = direction
    normal_helical_rod = normal
    binormal_helical_rod = np.cross(direction_helical_rod, normal_helical_rod)
    offset_list = [theta, 0]

    (
        fiber_length,
        fiber_start,
        fiber_position_collection,
        fiber_director_collection,
        intrinsic_link,
        injected_twist,
    ) = get_fiber_geometry(
        n_elem=n_elem_muscle,
        start_radius_list=start_radius_list,
        taper_slope_list=taper_slope_list,
        start_position=start_position_of_helix,
        direction=direction_helical_rod,
        normal=normal_helical_rod,
        offset_list=offset_list,
        length=muscle_height,
        turns_per_length_list=turns_per_length_list,
        initial_link_per_fiber_length=link_sign * initial_link_per_length,
        CCW_list=CCW,
    )

    volume = base_area * fiber_length
    mass = 0.22012 * mass_scale / divide
    density = mass / volume

    if (row, theta, orientation) in activation_group:
        muscle_rods[(row, theta, orientation)] = CosseratRod.straight_rod(
            n_elem_muscle,
            fiber_start,
            direction_helical_rod,
            normal_helical_rod,
            fiber_length,
            base_radius,
            density,
            youngs_modulus=E,
            shear_modulus=shear_modulus,
            position=fiber_position_collection,
            directors=fiber_director_collection,
        )
    else:
        muscle_rods[(row, theta, orientation)] = CosseratRod.straight_rod(
            n_elem_muscle,
            fiber_start,
            direction_helical_rod,
            normal_helical_rod,
            fiber_length,
            base_radius,
            density,
            youngs_modulus=E,
            shear_modulus=shear_modulus,
            position=fiber_position_collection,
            directors=fiber_director_collection,
        )

    octopus_arm_sim.append(muscle_rods[(row, theta, orientation)])

    # Add damping
    octopus_arm_sim.dampen(muscle_rods[(row, theta, orientation)]).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step=dt,
    )

    start_kappa[(row, theta, orientation)] = np.zeros((3, n_elem_muscle - 1))
    start_bend_matrix[(row, theta, orientation)] = np.zeros((3, 3, n_elem_muscle - 1))
    start_shear_matrix[(row, theta, orientation)] = np.zeros((3, 3, n_elem_muscle))
    start_mass_second_moment_of_inertia[(row, theta, orientation)] = np.zeros(
        (3, 3, n_elem_muscle)
    )
    start_inv_mass_second_moment_of_inertia[(row, theta, orientation)] = np.zeros(
        (3, 3, n_elem_muscle)
    )
    start_sigma[(row, theta, orientation)] = np.zeros((3, n_elem_muscle))
    end_kappa[(row, theta, orientation)] = np.zeros((3, n_elem_muscle - 1))
    end_sigma[(row, theta, orientation)] = np.zeros((3, n_elem_muscle))
    post_processing_dict_list.append(
        defaultdict(list)
    )  # list which collected data will be append

    (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    ) = get_connection_vector_straight_straight_rod(
        muscle_rods[(row, theta, orientation)],
        backbone_rod,
        (0, 1),
        (row * n_elem_per_row, (row * n_elem_per_row) + 1),
    )
    octopus_arm_sim.connect(
        first_rod=muscle_rods[(row, theta, orientation)],
        second_rod=backbone_rod,
        first_connect_idx=0,
        second_connect_idx=row * n_elem_per_row,
    ).using(
        SurfaceJointSideBySide,
        k=muscle_rods[(row, theta, orientation)].shear_matrix[2, 2, 0] * 1e4,
        nu=1e-4,
        k_repulsive=muscle_rods[(row, theta, orientation)].shear_matrix[
            2, 2, n_elem_muscle - 1
        ]
        * 1e2,
        rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
            ..., 0
        ],
        rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
            ..., 0
        ],
        offset_btw_rods=offset_btw_rods[0],
        friction_coefficient=0.0,
        velocity_damping_coefficient=0.0,
    )

    (
        rod_one_direction_vec_in_material_frame,
        rod_two_direction_vec_in_material_frame,
        offset_btw_rods,
    ) = get_connection_vector_straight_straight_rod(
        muscle_rods[(row, theta, orientation)],
        backbone_rod,
        (n_elem_muscle - 1, n_elem_muscle),
        (((row + 1) * n_elem_per_row) - 1, (row + 1) * n_elem_per_row),
    )

    octopus_arm_sim.connect(
        first_rod=muscle_rods[(row, theta, orientation)],
        second_rod=backbone_rod,
        first_connect_idx=n_elem_muscle - 1,
        second_connect_idx=((row + 1) * n_elem_per_row) - 1,
    ).using(
        SurfaceJointSideBySide,
        k=muscle_rods[(row, theta, orientation)].shear_matrix[2, 2, n_elem_muscle - 1]
        * 1e4,
        nu=1e-4,
        k_repulsive=muscle_rods[(row, theta, orientation)].shear_matrix[
            2, 2, n_elem_muscle - 1
        ]
        * 1e2,
        rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
            ..., 0
        ],
        rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
            ..., 0
        ],
        offset_btw_rods=offset_btw_rods[0],
        friction_coefficient=0.0,
        velocity_damping_coefficient=0.0,
    )

    for element in range(1, n_elem_muscle - 1):
        backbone_element = int(element * n_elem_per_row / n_elem_muscle)
        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(
            muscle_rods[(row, theta, orientation)],
            backbone_rod,
            (element, element + 1),
            (backbone_element, backbone_element + 1),
        )

        octopus_arm_sim.connect(
            first_rod=muscle_rods[(row, theta, orientation)],
            second_rod=backbone_rod,
            first_connect_idx=element,
            second_connect_idx=backbone_element,
        ).using(
            SurfaceJointSideBySide,
            k=muscle_rods[(row, theta, orientation)].shear_matrix[2, 2, element] * 1e-1,
            nu=0,
            k_repulsive=muscle_rods[(row, theta, orientation)].shear_matrix[
                2, 2, element
            ]
            * 1e0,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                ..., 0
            ],
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                ..., 0
            ],
            offset_btw_rods=offset_btw_rods[0],
            friction_coefficient=0.0,
            velocity_damping_coefficient=0.0,
        )

    # set the diagnostics for rod and collect data
    octopus_arm_sim.collect_diagnostics(muscle_rods[(row, theta, orientation)]).using(
        RodCallBack,
        step_skip=step_skip,
        callback_params=post_processing_dict_list[len(post_processing_dict_list) - 1],
    )


# Thompson Model Force and Torque
force_thompson = np.zeros((3, n_elem_muscle + 1))
torque_thompson = np.zeros((3, n_elem_muscle))


if Contraction:
    assert len(activation_startTime_untwistTime_force) == len(
        activation_group
    ), "Make sure each activated muscle has start time, untwist time, and force specified"
    for muscle_coords, startTime_untwistTime_force in zip(
        activation_group, activation_startTime_untwistTime_force
    ):
        row = muscle_coords[0]
        theta = muscle_coords[1]
        orientation = muscle_coords[2]
        start_time = startTime_untwistTime_force[0]
        untwist_time = startTime_untwistTime_force[1]
        kappa_change = startTime_untwistTime_force[2]
        octopus_arm_sim.add_forcing_to(muscle_rods[(row, theta, orientation)]).using(
            ArtficialMuscleActuation,
            start_radius=base_radius,
            start_density=density,
            start_kappa=start_kappa[(row, theta, orientation)],
            start_sigma=start_sigma[(row, theta, orientation)],
            start_bend_matrix=start_bend_matrix[(row, theta, orientation)],
            start_shear_matrix=start_shear_matrix[(row, theta, orientation)],
            start_mass_second_moment_of_inertia=start_mass_second_moment_of_inertia[
                (row, theta, orientation)
            ],
            start_inv_mass_second_moment_of_inertia=start_inv_mass_second_moment_of_inertia[
                (row, theta, orientation)
            ],
            ramp_up_time=untwist_time,
            start_time=start_time,
            thompson=Thompson_model,
            force_thompson=force_thompson,
            torque_thompson=torque_thompson,
            kappa_change=kappa_change,
            thermal_expansion_coefficient=thermal_expansion_coeficient,
            end_temperature=end_temperature,
            youngs_modulus_coefficients=youngs_modulus_coefficients,
            room_temperature=room_temperature,
            contraction_time=untwist_time,
        )


# finalize simulation
octopus_arm_sim.finalize()

for muscle_coords in included_muscles:
    row = muscle_coords[0]
    theta = muscle_coords[1]
    orientation = muscle_coords[2]

    start_kappa[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].kappa[:]
    start_sigma[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].sigma[:]
    start_shear_matrix[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].shear_matrix[:]
    start_bend_matrix[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].bend_matrix[:]
    start_mass_second_moment_of_inertia[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].mass_second_moment_of_inertia[:]
    start_inv_mass_second_moment_of_inertia[(row, theta, orientation)][:] = muscle_rods[
        (row, theta, orientation)
    ].inv_mass_second_moment_of_inertia[:]
    muscle_rods[(row, theta, orientation)].rest_kappa[:] = muscle_rods[
        (row, theta, orientation)
    ].kappa[:]
    muscle_rods[(row, theta, orientation)].rest_sigma[:] = muscle_rods[
        (row, theta, orientation)
    ].sigma[:]

# Run the simulation
time_stepper = PositionVerlet()
integrate(time_stepper, octopus_arm_sim, final_time, total_steps)


# plotting the videos
filename_video = "mini_muri_softer_backbone.mp4"
plot_video_with_surface(
    post_processing_dict_list,
    video_name=filename_video,
    fps=rendering_fps,
    step=1,
    vis3D=True,
    vis2D=True,
    x_limits=[-(backbone_length), (backbone_length)],
    y_limits=[-(backbone_length), (backbone_length)],
    z_limits=[-2 * backbone_radius[0], backbone_length + 2 * backbone_radius[0]],
)


if save_data:
    # Save data as npz file
    import os

    current_path = os.getcwd()
    save_folder = os.path.join(current_path, "data")
    os.makedirs(save_folder, exist_ok=True)
    time = np.array(post_processing_dict_list[0]["time"])

    n_muscle_rod = len(post_processing_dict_list) - 2

    muscle_rods_position_history = np.zeros(
        (n_muscle_rod, time.shape[0], 3, n_elem_muscle + 1)
    )
    muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem_muscle))
    backbone_position_history = np.zeros((1, time.shape[0], 3, n_elem_muscle + 1))
    backbone_radius_history = np.zeros((1, time.shape[0], n_elem_muscle))
    inner_spine_position_history = np.zeros((1, time.shape[0], 3, n_elem_muscle + 1))
    inner_spine_radius_history = np.zeros((1, time.shape[0], n_elem_muscle))
    marker_position_history = np.zeros((1, time.shape[0], 3, n_elem_muscle))
    marker_radius_history = (
        np.ones((1, time.shape[0], n_elem_muscle - 1)) * inner_spine_radius / 2
    )

    for t in range(len(time)):
        marker_position_history[0, t, :, :] = (
            np.array(post_processing_dict_list[0]["position"])[t, :, :-1]
            + np.array(post_processing_dict_list[0]["radius"])[t, :]
            * np.array(post_processing_dict_list[0]["directors"])[t, 0, :, :]
        )

    backbone_position_history[0, :, :, :] = np.array(
        post_processing_dict_list[0]["position"]
    )
    backbone_radius_history[0, :, :] = np.array(post_processing_dict_list[0]["radius"])

    inner_spine_position_history[0, :, :, :] = np.array(
        post_processing_dict_list[1]["position"]
    )
    inner_spine_radius_history[0, :, :] = np.array(
        post_processing_dict_list[1]["radius"]
    )
    for i in range(0, n_muscle_rod):
        muscle_rods_position_history[i, :, :, :] = np.array(
            post_processing_dict_list[i + 2]["position"]
        )
        muscle_rods_radius_history[i, :, :] = np.array(
            post_processing_dict_list[i + 2]["radius"]
        )

    np.savez(
        os.path.join(save_folder, "octopus_arm.npz"),
        time=time,
        muscle_rods_position_history=muscle_rods_position_history,
        muscle_rods_radius_history=muscle_rods_radius_history,
        backbone_position_history=backbone_position_history,
        backbone_radius_history=backbone_radius_history,
        inner_spine_position_history=inner_spine_position_history,
        inner_spine_radius_history=inner_spine_radius_history,
        marker_position_history=marker_position_history,
        marker_radius_history=marker_radius_history,
    )
