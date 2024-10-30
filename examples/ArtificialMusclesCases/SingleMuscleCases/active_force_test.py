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
    "forcing_start_time": 5.0,
}

# how is it actuated
actuation_settings_dict = {
    "actuation": True,
    "actuation_duration": 0.1,
    "start_time": 0.0,
}

# self contact settings
self_contact_settings_dict = {  # for mono
    "self_contact": True,
    "contact_radius_ratio": 1 / 10,
    "k_val": 5e2,
    "nu": 0.0,
    "k_repulsive_val": 1e2,
    "separation_distance": 1e-6,
    "decay_factor": 7.5e-5,
}
# self_contact_settings_dict = {
#     "self_contact": True,
#     "contact_radius_ratio": 1 / 10,
#     "k_val": 1e1,
#     "nu": 0.0,
#     "k_repulsive_val": 1e3,
#     "separation_distance":1e-5,
#     "decay_factor": 7.5e-4,
# }

# fiber-fiber interaction settings
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
# test_muscle = Samuel_supercoil_stl(experimental_data=True)
# test_muscle = Samuel_supercoil_stl_single_fiber(experimental_data=True)
# test_muscle = Liuyang_monocoil(experimental_data=True)
test_muscle = Samuel_monocoil(experimental_data=True)


sim_settings_dict = {
    "sim_name": "ActiveForceTest",
    "final_time": 10,  # seconds
    "rendering_fps": 20,
    "LaplaceFilter": False,
    "LaplaceFilterOrder": 7,
    "plot_video": True,
    "save_data": True,
    "return_data": True,
    "povray_viz": True,
    "additional_curves": True,
    "additional_identifier": "for_render_with_fiber_k_{0}_decay_factor_{1}_separation_distance_list_{2}".format(
        self_contact_settings.k_val,
        self_contact_settings.decay_factor,
        self_contact_settings.separation_distance,
    ),
    "save_folder": "",  # do not change
}
sim_settings = Dict2Class(sim_settings_dict)


n_muscles = 1
for i in range(len(test_muscle.geometry.n_ply_per_coil_level)):
    n_muscles *= test_muscle.geometry.n_ply_per_coil_level[i]

# tested_idx = range(len(test_muscle.strain_experimental)) #[0,1,2,3,4] #
# tested_idx = range(len(test_muscle.passive_force_experimental))  # [0,1,2,3,4] #
# tested_idx = range(len(test_muscle.active_force_experimental))  # [0,1,2,3,4] #
tested_idx = [-1]

additional_curve_list = [(test_muscle.active_list, "Experiments", "k")]


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

if constrain_settings.isometric_test and not constrain_settings.isobaric_test:
    passive_force_sim = np.zeros_like(test_muscle.strain_experimental)
    for i in tested_idx:
        forcing_settings.desired_muscle_strain = test_muscle.strain_experimental[i]
        print("Current Strain:" + str(forcing_settings.desired_muscle_strain))
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

        passive_force_measurement_time = -1  # end of simulation

        internal_force = np.zeros_like(np.array(data[0]["internal_force"]))
        for muscle_rods in data:
            internal_force += np.array(muscle_rods["internal_force"])

        passive_force_sim[i] = (
            internal_force[passive_force_measurement_time, 2, 0]
            * test_muscle.sim_settings.E_scale
        )
        print("Current Passive Force: " + str(passive_force_sim[i]))

    plt.rc("font", size=8)  # controls default text sizes
    if sim_settings.additional_curves:
        for curve_list, name in additional_curve_list:
            for curve in curve_list:
                plt.plot(curve[:, 0] / 100, curve[:, 1], color="k", linewidth=1)
            plt.plot(
                curve[:, 0] / 100, curve[:, 1], color="k", linewidth=1, label=name
            )  # to show curve list name only once

    plt.plot(
        test_muscle.active_strain_experimental[tested_idx],
        passive_force_sim[tested_idx],
        linewidth=3,
        marker="o",
        markersize=7,
        label="Passive force (Sim,Isometric)",
    )
    plt.suptitle("Coil Strain vs Force")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Force (N)")
    plt.xlim([0, 1.1 * test_muscle.strain_experimental[tested_idx[-1]]])


elif not constrain_settings.isometric_test and constrain_settings.isobaric_test:
    base_area = np.pi * (test_muscle.geometry.fiber_radius ** 2)
    strain_sim = np.zeros_like(test_muscle.active_force_experimental)

    # sim_settings.final_time = 5
    for i in tested_idx:
        forcing_settings.force_mag = test_muscle.active_force_experimental[i] / (
            1e-3
            * n_muscles
            * test_muscle.properties.youngs_modulus
            * test_muscle.sim_settings.E_scale
            * base_area
        )
        print(
            "Current Force:",
            test_muscle.active_force_experimental[i],
            "Current Normalized Force Mag:",
            forcing_settings.force_mag,
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

        # coil_radius_sim[i] = np.mean(
        #     _batch_norm(
        #         centerline_position[-1, :, :]
        #         - _batch_product_i_ik_to_k(
        #             test_muscle.geometry.direction, centerline_position[-1, :, :]
        #         ).reshape(1, n)
        #         * test_muscle.geometry.direction.reshape(3, 1)
        #     )
        # )

        print("Current Strain: " + str(strain_sim[i]))

    plt.rc("font", size=8)  # controls default text sizes
    if sim_settings.additional_curves:
        for curve_list, name, color in additional_curve_list:
            # for curve in curve_list:
            x1 = curve_list[0][:, 0] / 100
            x2 = curve_list[-1][:, 0] / 100
            y1 = curve_list[0][:, 1]
            y2 = curve_list[-1][:, 1]
            plt.plot(x1, y1, color, label=name)
            plt.plot(x2, y2, color)
            plt.fill(
                np.append(x1, x2[::-1]), np.append(y1, y2[::-1]), color=color, alpha=0.2
            )

    plt.plot(
        strain_sim[tested_idx] + test_muscle.post_passive_strain,
        test_muscle.active_force_experimental[tested_idx],
        linewidth=3,
        marker="o",
        markersize=7,
        color="r",
        label="Active force (Simulation)",
    )
    plt.suptitle("Coil Strain vs Force")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Force (N)")
    # plt.xlim([0,1.1*test_muscle.strain_experimental[tested_idx[-1]]])
else:
    print("Please make sure one of isometric_test or isobaric_test is True, not both")

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
