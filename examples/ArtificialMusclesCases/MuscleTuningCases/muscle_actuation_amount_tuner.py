import numpy as np
from elastica import *
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator import (
    muscle_stretching_actuation_simulator,
)


# how the muscle is constrained
constrain_settings_dict = {
    "isometric_test": False,
    "isobaric_test": True,
    "constrain_start_time": 0.0,
}

# what external forcing it is experiencing
forcing_settings_dict = {
    "force_mag": 25.0,
    "desired_muscle_strain": 1.0,
    "start_force_direction": np.array([0.0, 0.0, -1.0]),
    "end_force_direction": np.array([0.0, 0.0, 1.0]),
    "ramp_up_time": 1.0,
    "forcing_start_time": 0.0,
}

# how is it actuated
actuation_settings_dict = {
    "actuation": True,
    "actuation_duration": 6.0,
    "start_time": 0.0,
}

# self contact settings
self_contact_settings_dict = {
    "self_contact": True,
    "contact_radius_ratio": 1 / 10,
    "k_val": 0e0,
    "nu": 0.0,
    "k_repulsive_val": 4e0,  # 4e0
    "separation_distance": 1e-3,
}

# fiber-fiber interaction settings
# fiber_connection_settings_dict = {
#     "connection_range": 1,
#     "k_val": 0.0,
#     "nu": 0.0,
#     "k_repulsive_val": 0.0,
#     "friction_coefficient": 0.0,
#     "velocity_damping_coefficient": 1e5,
# }

fiber_connection_settings_dict = {
    "connection_range": 2,
    "k_val": 2.5e2 / 16,
    "nu": 0.0,
    "k_repulsive_val": 1e2,
    "friction_coefficient": 5e-1,
    "velocity_damping_coefficient": 1e5,
}

# convert settings dicts to classes for ease
constrain_settings = Dict2Class(constrain_settings_dict)
forcing_settings = Dict2Class(forcing_settings_dict)
actuation_settings = Dict2Class(actuation_settings_dict)
self_contact_settings = Dict2Class(self_contact_settings_dict)
fiber_connection_settings = Dict2Class(fiber_connection_settings_dict)


sim_settings_dict = {
    "sim_name": "Tuning",
    "final_time": 10,  # seconds
    "rendering_fps": 20,
    "LaplaceFilter": False,
    "LaplaceFilterOrder": 7,
    "plot_video": True,
    "save_data": False,
    "return_data": True,
    "povray_viz": False,
    "save_folder": "",  # do not change
    "additional_identifier": "",
}
sim_settings = Dict2Class(sim_settings_dict)

strain_difference = 10
tuning_muscle = Liuyang_monocoil(experimental_data=True)
# tuning_muscle = Samuel_supercoil_stl(experimental_data=False)


n_muscles = 1
for i in range(len(tuning_muscle.geometry.n_ply_per_coil_level)):
    n_muscles *= tuning_muscle.geometry.n_ply_per_coil_level[i]
base_area = np.pi * (tuning_muscle.geometry.fiber_radius ** 2)
forcing_settings.force_mag = tuning_muscle.active_force_experimental[0] / (
    1e-3
    * n_muscles
    * tuning_muscle.properties.youngs_modulus
    * tuning_muscle.sim_settings.E_scale
    * base_area
)
# forcing_settings.force_mag = 0.0

k = 1e-3

current_path = os.getcwd()
sim_settings.save_folder = os.path.join(
    current_path, sim_settings.sim_name + "/" + tuning_muscle.name + "/data"
)
# tuning_muscle.sim_settings.actuation_kappa_change = 0.1
while abs(strain_difference) > 1e-6:
    print(
        "Current Kappa Change:" + str(tuning_muscle.sim_settings.actuation_kappa_change)
    )
    data = muscle_stretching_actuation_simulator(
        input_muscle=tuning_muscle,  # the type of muscle used
        constrain_settings=constrain_settings,  # how the muscle is constrained
        forcing_settings=forcing_settings,  # what external forcing it is experiencing
        actuation_settings=actuation_settings,  # how is it actuated
        self_contact_settings=self_contact_settings,  # self contact settings
        fiber_connection_settings=fiber_connection_settings,  # fiber-fiber interaction settings
        sim_settings=sim_settings,
    )

    # internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
    # for muscle_rods in data:
    #     internal_force += np.array(muscle_rods["internal_force"])

    # force_difference = (
    #     tuning_muscle.total_force_experimental[0]
    #     - internal_force[-1, 2, 0] * tuning_muscle.sim_settings.E_scale
    # )

    centerline_position = np.zeros_like(np.array(data[0]["position"]))
    for muscle_rods in data:
        centerline_position += np.array(muscle_rods["position"]) / n_muscles

    n = centerline_position.shape[-1]
    strain_difference = np.dot(
        tuning_muscle.geometry.direction,
        (
            (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
            - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
        )
        / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
    )
    print("Current Strain Difference:" + str(strain_difference))
    tuning_muscle.sim_settings.actuation_kappa_change += k * strain_difference

print("Tuned Kappa Change: " + str(tuning_muscle.sim_settings.actuation_kappa_change))
