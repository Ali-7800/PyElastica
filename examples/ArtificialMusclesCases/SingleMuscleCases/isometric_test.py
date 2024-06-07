import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_stretching_actuation_simulatior
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# how the muscle is constrained
constrain_settings_dict = {
    "isometric_test": True,
    "isobaric_test": False,
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
    "actuation_duration": 1.0,
    "start_time": 6.0,
}

# self contact settings
self_contact_settings_dict = {
    "self_contact": False,
    "contact_radius_ratio": 1 / 10,
    "k_val": 1e1,
    "nu": 0.0,
    "k_repulsive_val": 1e2,
    "separation_distance": 1e-3,
}

# fiber-fiber interaction settings
fiber_connection_settings_dict = {
    "connection_range": 2,
    "k_val": 2.5e2,
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
    "sim_name": "IsometricTest",
    "final_time": 16,  # seconds
    "rendering_fps": 20,
    "LaplaceFilter": False,
    "LaplaceFilterOrder": 7,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": False,
    "additional_curves": True,
    "additional_identifier": "",
    "save_folder": "",  # do not change
}


sim_settings = Dict2Class(sim_settings_dict)

# test_muscle = Liuyang_monocoil()
test_muscle = Liuyang_monocoil()
tested_strains_idx = range(len(test_muscle.strain_experimental))

passive_force_sim = np.zeros_like(test_muscle.strain_experimental)
total_force_sim = np.zeros_like(test_muscle.strain_experimental)

additional_curve_list = []
# additional_curve_list = test_muscle.trained_tensile_test_list
# additional_curve_list.append(test_muscle.active_force_curve)


if len(additional_curve_list) > 0 and sim_settings.additional_curves == False:
    raise (
        "You have additional curves to plot but you have set the plotting option to false"
    )
elif len(additional_curve_list) == 0 and sim_settings.additional_curves == True:
    raise (
        "You have no additional curves to plot but you have set the plotting option to True"
    )

current_path = os.getcwd()
save_folder = os.path.join(
    current_path, sim_settings.sim_name + "/" + test_muscle.name + "/data"
)

for i in tested_strains_idx:  # range(len(test_muscle.strain_experimental)):
    sim_settings.muscle_strain = test_muscle.strain_experimental[i]
    print("Current Strain:" + str(sim_settings.muscle_strain))
    data = muscle_stretching_actuation_simulatior(
        input_muscle=test_muscle,  # the type of muscle used
        constrain_settings=constrain_settings,  # how the muscle is constrained
        forcing_settings=forcing_settings,  # what external forcing it is experiencing
        actuation_settings=actuation_settings,  # how is it actuated
        self_contact_settings=self_contact_settings,  # self contact settings
        fiber_connection_settings=fiber_connection_settings,  # fiber-fiber interaction settings
        sim_settings=sim_settings,
    )
    time = data[0]["time"]
    passive_force_measurement_time = (
        int((len(time)) * actuation_settings.start_time / sim_settings.final_time) - 2
    )  # just right before the muscle is actuated

    internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
    for muscle_rods in data:
        internal_force += np.array(muscle_rods["internal_force"])

    passive_force_sim[i] = (
        internal_force[passive_force_measurement_time, 2, 0]
        * test_muscle.sim_settings.E_scale
    )
    total_force_sim[i] = internal_force[-1, 2, 0] * test_muscle.sim_settings.E_scale
    print(
        "Current Passive Force: " + str(passive_force_sim[i]),
        "Current Total Force: " + str(total_force_sim[i]),
    )

plt.rc("font", size=8)  # controls default text sizes
# plt.plot(
#     test_muscle.strain_experimental[tested_strains_idx],
#     passive_force_sim[tested_strains_idx]/test_muscle.cross_sectional_area ,
#     marker="o",
#     label="Passive Stress (Sim)",
# )
# plt.plot(
#     test_muscle.strain_experimental[tested_strains_idx],
#     total_force_sim[tested_strains_idx]/test_muscle.cross_sectional_area,
#     marker="o",
#     label="Total Stress (Sim at "
#     + str(test_muscle.sim_settings.actuation_end_temperature)
#     + "C)",
# )

plt.plot(
    test_muscle.strain_experimental[tested_strains_idx],
    (total_force_sim[tested_strains_idx] - passive_force_sim[tested_strains_idx])
    / test_muscle.cross_sectional_area,
    marker="o",
    label="Active Stress (Sim at "
    + str(test_muscle.sim_settings.actuation_end_temperature)
    + "C)",
)

if sim_settings.additional_curves:
    for curve, name, color in additional_curve_list:
        plt.plot(
            curve[:, 0],
            curve[:, 1] / test_muscle.cross_sectional_area,
            color=color,
            linewidth=1,
            label=name,
        )

plt.suptitle("Coil Strain vs Stress")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Stress (MPa)")
plt.xlim([0, 1.1 * test_muscle.strain_experimental[-1]])
plt.legend()
plt.savefig(save_folder + "/plot.png", dpi=300)
plt.show()
