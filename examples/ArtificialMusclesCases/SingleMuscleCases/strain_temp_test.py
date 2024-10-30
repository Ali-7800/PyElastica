import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator import (
    muscle_stretching_actuation_simulator,
)
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from elastica._linalg import (
    _batch_product_i_ik_to_k,
    _batch_norm,
)


# how the muscle is constrained
constrain_settings_dict = {
    "isometric_test": False,
    "isobaric_test": False,
    "constrain_start_time": 0.0,
}

# what external forcing it is experiencing
forcing_settings_dict = {
    "force_mag": 0.0,
    "desired_muscle_strain": 1.0,
    "start_force_direction": np.array([0.0, 0.0, -1.0]),
    "end_force_direction": np.array([0.0, 0.0, 1.0]),
    "ramp_up_time": 1.0,
    "forcing_start_time": 0.0,
}

# how is it actuated
actuation_settings_dict = {
    "actuation": True,
    "actuation_duration": 1.0,
    "start_time": 0.0,
}

# self contact settings
self_contact_settings_dict = {
    "self_contact": True,
    "contact_radius_ratio": 1 / 10,
    "k_val": 1e1,
    "nu": 0.0,
    "k_repulsive_val": 1e3,
    "separation_distance": 1e-6,
    "decay_factor": 2.5e-4,
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
    "connection_range": 1,
    "k_val_list": [0.04 * 1.5e2, 0.04 * 1.5e2],
    "nu": 0.0,
    "k_repulsive_val": 1e1,
    "friction_coefficient": 5e-1,
    "velocity_damping_coefficient": 1e5,
    "separation_distance_list": [5e-4, 3.59e-4],
    "decay_factor_list": [2.5e-5, 2.5e-7],
}

# convert settings dicts to classes for ease
constrain_settings = Dict2Class(constrain_settings_dict)
forcing_settings = Dict2Class(forcing_settings_dict)
actuation_settings = Dict2Class(actuation_settings_dict)
self_contact_settings = Dict2Class(self_contact_settings_dict)
fiber_connection_settings = Dict2Class(fiber_connection_settings_dict)
test_muscle = Samuel_supercoil_stl(experimental_data=True)
# test_muscle = Samuel_supercoil_stl_single_fiber(experimental_data=True)
# test_muscle = Liuyang_monocoil(experimental_data=True)
# test_muscle = Liuyang_monocoil()
# test_muscle = Samuel_monocoil(experimental_data=True)


sim_settings_dict = {
    "sim_name": "StrainTempTest",
    "final_time": 10,  # seconds
    "rendering_fps": 20,
    "LaplaceFilter": False,
    "LaplaceFilterOrder": 7,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": False,
    "additional_curves": True,
    "additional_identifier": "new_data_with_k_val_{0}_kappa_change_{1}_slope_adjuster{2}".format(
        self_contact_settings.k_repulsive_val,
        test_muscle.sim_settings.actuation_kappa_change,
        test_muscle.sim_settings.actuation_slope_adjuster,
    ),
    "save_folder": "",  # do not change
}
sim_settings = Dict2Class(sim_settings_dict)


n_muscles = 1
for i in range(len(test_muscle.geometry.n_ply_per_coil_level)):
    n_muscles *= test_muscle.geometry.n_ply_per_coil_level[i]

n_temperatures = 5
temperature_sim = np.linspace(
    test_muscle.sim_settings.actuation_start_temperature,
    test_muscle.sim_settings.actuation_end_temperature,
    n_temperatures,
)

additional_curve_list = [(test_muscle.temperature_strain, "Experimental Data")]
# additional_curve_list = []

if len(additional_curve_list) > 0 and sim_settings.additional_curves == False:
    raise Exception(
        "You have additional curves to plot but you have set the plotting option to false"
    )
elif len(additional_curve_list) == 0 and sim_settings.additional_curves == True:
    raise Exception(
        "You have no additional curves to plot but you have set the plotting option to True"
    )


current_path = os.getcwd()
sim_settings.save_folder = os.path.join(
    current_path, sim_settings.sim_name + "/" + test_muscle.name + "/data"
)


base_area = np.pi * (test_muscle.geometry.fiber_radius ** 2)
strain_sim = np.zeros_like(temperature_sim)

for temperature, i in zip(temperature_sim[1:], range(1, n_temperatures)):
    test_muscle.sim_settings.actuation_end_temperature = temperature
    print(
        "Current Actuation Temperature:",
        test_muscle.sim_settings.actuation_end_temperature,
    )
    data = muscle_stretching_actuation_simulator(
        input_muscle=test_muscle,  # the type of muscle used
        constrain_settings=constrain_settings,  # how the muscle is constrained
        forcing_settings=forcing_settings,  # what external forcing it is experiencing
        actuation_settings=actuation_settings,  # how is it actuated
        self_contact_settings=self_contact_settings,  # self contact settings
        fiber_connection_settings=fiber_connection_settings,  # fiber-fiber interaction settings
        sim_settings=sim_settings,
    )

    time = data[0]["time"]

    centerline_position = np.zeros_like(np.array(data[0]["position"]))
    for muscle_rods in data:
        centerline_position += np.array(muscle_rods["position"]) / n_muscles

    n = centerline_position.shape[-1]
    final_length = np.dot(
        test_muscle.geometry.direction,
        (centerline_position[-1, :, -1] - centerline_position[-1, :, 0]),
    )
    start_length = np.dot(
        test_muscle.geometry.direction,
        (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
    )
    strain_sim[i] = (final_length - start_length) / start_length

    print("Current Strain: " + str(strain_sim[i]))

plt.rc("font", size=8)  # controls default text sizes
if sim_settings.additional_curves:
    for curve, name in additional_curve_list:
        plt.plot(
            curve[:, 0], curve[:, 1], color="k", linewidth=1, label=name
        )  # to show curve list name only once

plt.plot(
    temperature_sim,
    strain_sim,
    linewidth=3,
    marker="o",
    markersize=7,
    label="Simulation",
)
plt.suptitle("Temperature vs Strain")
plt.ylabel("Strain (mm/mm)")
plt.xlabel("Temperature (°C)")

plt.legend()
plt.savefig(
    sim_settings.save_folder + "/plot_" + sim_settings.additional_identifier + ".png",
    dpi=300,
)
plt.savefig(
    sim_settings.save_folder + "/plot_" + sim_settings.additional_identifier + ".eps",
    dpi=300,
)
plt.show()
plt.close()
