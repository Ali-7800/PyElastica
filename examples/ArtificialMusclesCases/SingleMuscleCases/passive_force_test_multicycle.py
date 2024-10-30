import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *
from examples.ArtificialMusclesCases.SingleMuscleCases.single_muscle_simulator_multicycle import (
    muscle_stretching_actuation_simulator_multicycle,
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
    "isobaric_test": True,
    "constrain_start_time": 0.0,
}

# how is it actuated
actuation_settings_dict = {
    "actuation": False,
    "actuation_duration": 6.0,
    "start_time": 0.0,
}

# self contact settings
self_contact_settings_dict = {
    "self_contact": False,
    "contact_radius_ratio": 1 / 10,
    "k_val": 1e1,
    "nu": 0.0,
    "k_repulsive_val": 1e2,
    "separation_distance": 1e-6,
    "decay_factor": 2.5e-4,
}

# fiber-fiber interaction settings
fiber_connection_settings_dict = {
    "connection_range": 1,
    "k_val_list": [1.5e2, 1.5e2],
    "nu": 0.0,
    "k_repulsive_val": 1e1,
    "friction_coefficient": 5e-1,
    "velocity_damping_coefficient": 1e5,
    "separation_distance_list": [5e-4, 3.59e-4],
    "decay_factor_list": [2.5e-5, 2.5e-7],
}

# convert settings dicts to classes for ease
constrain_settings = Dict2Class(constrain_settings_dict)
actuation_settings = Dict2Class(actuation_settings_dict)
self_contact_settings = Dict2Class(self_contact_settings_dict)
fiber_connection_settings = Dict2Class(fiber_connection_settings_dict)
test_muscle = Samuel_supercoil_stl(experimental_data=True)

test_muscle.sim_settings.radial_noise_scale = (
    1e-2 * test_muscle.geometry.fiber_radius
)  # 1% fiber radius
test_muscle.sim_settings.position_noise_scale = (
    1e-2 * test_muscle.geometry.radial_profile_list[0](0.5)
)  # 1# coil radius
# test_muscle.sim_settings.velocity_noise_scale = 0.01*test_muscle.geometry.fiber_radius
n_cycles = 5

# test_muscle = Samuel_supercoil_stl_single_fiber()
# test_muscle = Liuyang_monocoil(experimental_data=True)
# test_muscle = Samuel_monocoil()
# test_muscle = Samuel_monocoil_no_gap()


n_muscles = 1
for i in range(len(test_muscle.geometry.n_ply_per_coil_level)):
    n_muscles *= test_muscle.geometry.n_ply_per_coil_level[i]

base_area = np.pi * (test_muscle.geometry.fiber_radius ** 2)

# what external forcing it is experiencing
def force_mag_func(cycle, given_mag):
    if cycle < n_cycles - 1:
        force_mag = 10 / (
            1e-3
            * n_muscles
            * test_muscle.properties.youngs_modulus
            * test_muscle.sim_settings.E_scale
            * base_area
        )
    else:
        force_mag = given_mag  # last cycle is the actual test cycle
    return force_mag


def forcing_duration_func(cycle):
    if cycle < n_cycles - 1:
        duration = 2
    else:
        duration = 10  # last cycle is the actual test cycle
    return duration


forcing_settings_dict = {
    "default_force_mag": 25.0,
    "desired_muscle_strain": 1.0,
    "start_force_direction": np.array([0.0, 0.0, -1.0]),
    "end_force_direction": np.array([0.0, 0.0, 1.0]),
    "ramp_up_time": 1.0,
    "forcing_start_time": 0.0,
    "force_mag_func": force_mag_func,
    "n_cycles": n_cycles,
    "forcing_duration_func": forcing_duration_func,
    "rest_duration": 2,
}

forcing_settings = Dict2Class(forcing_settings_dict)

final_time = forcing_duration_func(n_cycles)
for cycle in range(n_cycles - 1):
    final_time += forcing_settings.rest_duration + forcing_duration_func(cycle)

sim_settings_dict = {
    "sim_name": "PassiveForceTest{0}Cycle".format(n_cycles),
    "final_time": final_time,  # seconds
    "rendering_fps": 20,
    "LaplaceFilter": False,
    "LaplaceFilterOrder": 7,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": False,
    "additional_curves": True,
    "additional_identifier": "_with_radial_noise_{0}_and_position_noise_{1}_velocity_noise_{2}".format(
        test_muscle.sim_settings.radial_noise_scale,
        test_muscle.sim_settings.position_noise_scale,
        test_muscle.sim_settings.velocity_noise_scale,
    ),
    "save_folder": "",  # do not change
}
sim_settings = Dict2Class(sim_settings_dict)

# tested_idx = range(len(test_muscle.strain_experimental)) #[0,1,2,3,4] #
tested_idx = range(len(test_muscle.passive_force_experimental))  # [0,1,2,3,4] #
# tested_idx = [0, 2, 4, 6, 8]
# tested_strains_idx = range(len(test_muscle.strain_experimental))
# tested_strains_idx = [0,2,4,6,8]
# additional_curve_list = [(test_muscle.experimental_tensile_test,"Tensile Test")]
# additional_curve_list = [(test_muscle.experimental_tensile_test_single_fiber,"Single Fiber Tensile Test")]
# additional_curve_list = [
#     (
#         test_muscle.experimental_tensile_test_single_fiber_times_3,
#         "Single Fiber Tensile Test ×3",
#     ),
#     (test_muscle.experimental_tensile_test, "Supercoil Tensile Test"),
# ]
additional_curve_list = [(test_muscle.passive_list, "Passive Force", "k")]
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


if not constrain_settings.isometric_test and constrain_settings.isobaric_test:
    strain_sim = np.zeros_like(test_muscle.passive_force_experimental)
    coil_radius_sim = np.zeros_like(test_muscle.passive_force_experimental)

    # sim_settings.final_time = 5
    for i in tested_idx[1:]:
        forcing_settings.default_force_mag = test_muscle.passive_force_experimental[
            i
        ] / (
            1e-3
            * n_muscles
            * test_muscle.properties.youngs_modulus
            * test_muscle.sim_settings.E_scale
            * base_area
        )
        print(
            "Current Force:",
            test_muscle.passive_force_experimental[i],
            "Current Normalized Force Mag:",
            forcing_settings.default_force_mag,
        )
        data = muscle_stretching_actuation_simulator_multicycle(
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
        strain_sim[i] = np.dot(
            test_muscle.geometry.direction,
            (
                (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
                - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
            )
            / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
        )
        coil_radius_sim[i] = np.mean(
            _batch_norm(
                centerline_position[-1, :, :]
                - _batch_product_i_ik_to_k(
                    test_muscle.geometry.direction, centerline_position[-1, :, :]
                ).reshape(1, n)
                * test_muscle.geometry.direction.reshape(3, 1)
            )
        )

        print("Current Strain: " + str(strain_sim[i]))

    plt.rc("font", size=8)  # controls default text sizes
    if sim_settings.additional_curves:
        for curve_list, name, color in additional_curve_list:
            # for curve in curve_list:
            x1 = curve_list[0][:, 0] / 100
            x2 = curve_list[-1][:, 0] / 100
            y1 = curve_list[0][:, 1]
            y2 = curve_list[-1][:, 1]
            x_mean = np.zeros_like(x1)
            y_mean = np.zeros_like(y1)
            plt.plot(x1, y1, color, label="Experimental Range")
            plt.plot(x2, y2, color)
            plt.fill(
                np.append(x1, x2[::-1]), np.append(y1, y2[::-1]), color=color, alpha=0.2
            )
    plt.plot(
        strain_sim[tested_idx],
        test_muscle.passive_force_experimental[tested_idx],
        linewidth=2,
        marker="o",
        markersize=5,
        color="red",
        label="Passive force (Sim,Isobaric)",
    )
    # plt.suptitle("Coil Strain vs Force")
    plt.xlabel("Strain (mm/mm)", fontsize=16)
    plt.ylabel("Force (N)", fontsize=16)
    # plt.xlim([0,1.1*test_muscle.strain_experimental[tested_idx[-1]]])
else:
    print("Please make sure one of isometric_test or isobaric_test is True, not both")

plt.rc("font", size=16)  # controls default text sizes
plt.legend()
plt.tick_params(labelsize=16)
plt.savefig(
    sim_settings.save_folder + "/plot" + sim_settings.additional_identifier + ".png",
    dpi=300,
)
plt.savefig(
    sim_settings.save_folder + "/plot_" + sim_settings.additional_identifier + ".eps",
    dpi=300,
)
plt.show()
plt.close()
# plt.plot(
#     strain_sim[tested_idx],
#     coil_radius_sim[tested_idx] * 1000,
#     linewidth=3,
#     marker="o",
#     markersize=7,
#     label="Coil Radius (Sim,Isobaric)",
# )
# plt.plot(
#     strain_sim[tested_idx],
#     initial_coil_radius * np.ones_like(coil_radius_sim[tested_idx]) * 1000,
#     linewidth=3,
#     linestyle="dashed",
#     markersize=7,
#     label="Start Coil Radius",
# )

# plt.suptitle("Coil Strain vs Coil Radius")
# plt.xlabel("Strain (mm/mm)")
# plt.ylabel("Radius (mm)")
# plt.legend()
# plt.show()
