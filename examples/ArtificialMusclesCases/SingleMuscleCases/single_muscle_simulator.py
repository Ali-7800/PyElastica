import numpy as np
import os
from elastica import *
from examples.ArtificialMusclesCases import *


class MuscleCase(
    BaseSystemCollection,
    Constraints,
    MemoryBlockConnections,
    Forcing,
    CallBacks,
    Damping,
):
    pass


def muscle_stretching_actuation_simulatior(
    input_muscle,  # the type of muscle used
    constrain_settings,  # how the muscle is constrained
    forcing_settings,  # what external forcing it is experiencing
    actuation_settings,  # how is it actuated
    self_contact_settings,  # self contact settings
    fiber_connection_settings,  # fiber-fiber interaction settings
    sim_settings,
):

    # create new sim
    muscle_sim = MuscleCase()
    # save folder
    os.makedirs(sim_settings.save_folder, exist_ok=True)

    # time step calculation
    n_turns = (
        input_muscle.geometry.turns_per_length_list[0]
        * input_muscle.geometry.muscle_length
    )
    n_elem = int(input_muscle.sim_settings.n_elem_per_coil * n_turns)
    dt = 0.3 * input_muscle.geometry.muscle_length / n_elem
    total_steps = int(sim_settings.final_time / dt)
    time_step = np.float64(sim_settings.final_time / total_steps)
    step_skip = int(1.0 / (sim_settings.rendering_fps * time_step))

    # create coiled muscle
    muscle = CoiledMuscle(
        input_muscle.geometry, input_muscle.properties, input_muscle.sim_settings
    )

    muscle.append_muscle_to_sim(muscle_sim)

    # connects rods if there is more than one
    muscle.connect_muscle_rods(
        muscle_sim, connection_settings=fiber_connection_settings
    )

    # Add damping
    muscle.dampen_muscle(
        muscle_sim,
        AnalyticalLinearDamper,
        damping_constant=input_muscle.sim_settings.nu,
        time_step=dt,
    )
    if sim_settings.LaplaceFilter:
        muscle.dampen_muscle(
            muscle_sim,
            LaplaceDissipationFilter,
            filter_order=sim_settings.LaplaceFilterOrder,
        )

    # Slider constarint
    for i in [0, -1]:
        if i == 0:
            translational_constraint_selector = np.array([True, True, True])
        else:
            translational_constraint_selector = np.array([True, True, False])
        muscle.constrain_muscle(
            muscle_sim,
            GeneralConstraint,
            constrained_position_idx=(i,),
            constrained_director_idx=(i,),
            translational_constraint_selector=translational_constraint_selector,
            rotational_constraint_selector=np.array([False, False, False]),
        )

    # Add self contact to prevent penetration
    if self_contact_settings.self_contact:
        muscle.apply_self_contact(
            muscle_sim, self_contact_settings=self_contact_settings
        )

    base_area = np.pi * input_muscle.geometry.fiber_radius ** 2
    force_scale = (
        1e-3
        * forcing_settings.force_mag
        * input_muscle.properties.youngs_modulus
        * base_area
    )
    end_force = (
        force_scale * forcing_settings.end_force_direction
    )  # np.array([0.0, 0.0, 1.0])
    start_force = (
        force_scale * forcing_settings.start_force_direction
    )  # np.array([0.0, 0.0, -1.0])
    print("Force: " + str(end_force * input_muscle.sim_settings.E_scale))

    # Add endpoint forces to rod
    if forcing_settings.desired_muscle_strain > 0:
        muscle.add_forcing_to_muscle(
            muscle_sim,
            EndpointForcesWithStartTime,
            start_force=start_force,
            end_force=end_force,
            ramp_up_time=forcing_settings.ramp_up_time,
            start_time=forcing_settings.forcing_start_time,
        )

    # actaute muscle
    if actuation_settings.actuation:
        muscle.actuate(
            muscle_sim,
            ArtficialMuscleActuation,
            contraction_time=actuation_settings.actuation_duration,
            start_time=actuation_settings.start_time,
            kappa_change=input_muscle.sim_settings.actuation_kappa_change,
            room_temperature=input_muscle.sim_settings.actuation_start_temperature,
            end_temperature=input_muscle.sim_settings.actuation_end_temperature,
            youngs_modulus_coefficients=input_muscle.properties.youngs_modulus_coefficients,
            thermal_expansion_coefficient=input_muscle.properties.thermal_expansion_coefficient,
        )

    if constrain_settings.isometric_test:
        if forcing_settings.desired_muscle_strain > 0.0:
            muscle.constrain_muscle(
                muscle_sim,
                IsometricStrainBC,
                desired_length=(1 + forcing_settings.desired_muscle_strain)
                * input_muscle.geometry.muscle_length,
                constraint_node_idx=[0, -1],
                length_node_idx=[0, -1],
                direction=input_muscle.geometry.direction,
            )
        else:
            muscle.constrain_muscle(
                muscle_sim,
                IsometricBC,
                constrain_start_time=constrain_settings.constrain_start_time,
                constrained_nodes=[0, -1],
            )

    post_processing_dict_list = []  # list which collected data will be append
    # set the diagnostics for rod and collect data
    muscle.muscle_callback(
        muscle_sim,
        post_processing_dict_list,
        step_skip,
    )

    # finalize simulation
    muscle_sim.finalize()

    # store initial properties for actuation
    # fix muscle shape/annealing
    muscle.fix_shape_and_store_start_properties()

    # Run the simulation
    time_stepper = PositionVerlet()
    integrate(time_stepper, muscle_sim, sim_settings.final_time, total_steps)

    # plotting the videos
    if sim_settings.plot_video:
        if constrain_settings.isometric_test == True:
            filename_video = (
                sim_settings.sim_name
                + "_strain_"
                + str(forcing_settings.desired_muscle_strain)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".mp4"
            )
        elif constrain_settings.isobaric_test == True:
            filename_video = (
                sim_settings.sim_name
                + "_force_mag_"
                + str(forcing_settings.force_mag)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".mp4"
            )

        plot_video_with_surface(
            post_processing_dict_list,
            folder_name=sim_settings.save_folder,
            video_name=filename_video,
            fps=sim_settings.rendering_fps,
            step=1,
            vis3D=True,
            vis2D=True,
            x_limits=[
                -input_muscle.geometry.muscle_length / 2,
                input_muscle.geometry.muscle_length / 2,
            ],
            y_limits=[
                -input_muscle.geometry.muscle_length / 2,
                input_muscle.geometry.muscle_length / 2,
            ],
            z_limits=[
                -1.1 * input_muscle.geometry.muscle_length,
                2.1 * input_muscle.geometry.muscle_length,
            ],
        )

    if sim_settings.povray_viz:
        import pickle

        if constrain_settings.isometric == True:
            filename = (
                sim_settings.save_folder
                + "/"
                + sim_settings.sim_name
                + "_strain_"
                + str(forcing_settings.desired_muscle_strain)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".dat"
            )
        elif constrain_settings.isobaric == True:
            filename = (
                sim_settings.save_folder
                + "/"
                + sim_settings.sim_name
                + "_force_mag_"
                + str(forcing_settings.force_mag)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".dat"
            )

        file = open(filename, "wb")
        pickle.dump(post_processing_dict_list, file)
        file.close()

    if sim_settings.save_data:
        # Save data as npz file
        time = np.array(post_processing_dict_list[0]["time"])

        n_muscle_rod = len(muscle.muscle_rods)

        muscle_rods_position_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )
        muscle_rods_radius_history = np.zeros((n_muscle_rod, time.shape[0], n_elem))
        muscle_rods_tangent_history = np.zeros((n_muscle_rod, time.shape[0], 3, n_elem))
        muscle_rods_internal_force_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )
        muscle_rods_external_force_history = np.zeros(
            (n_muscle_rod, time.shape[0], 3, n_elem + 1)
        )

        for i in range(n_muscle_rod):
            muscle_rods_position_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["position"]
            )
            muscle_rods_radius_history[i, :, :] = np.array(
                post_processing_dict_list[i]["radius"]
            )
            directors = np.array(post_processing_dict_list[i]["directors"])
            muscle_rods_tangent_history[i, :, :, :] = np.array(directors[:, 2, :, :])
            muscle_rods_internal_force_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["internal_force"]
            )
            muscle_rods_external_force_history[i, :, :, :] = np.array(
                post_processing_dict_list[i]["external_force"]
            )
        if constrain_settings.isometric_test == True:
            data_filename = (
                sim_settings.sim_name
                + "_strain_"
                + str(forcing_settings.desired_muscle_strain)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".npz"
            )
        elif constrain_settings.isobaric_test == True:
            data_filename = (
                sim_settings.sim_name
                + "_force_mag_"
                + str(forcing_settings.force_mag)
                + "_"
                + input_muscle.name
                + "_"
                + sim_settings.additional_identifier
                + ".npz"
            )

        np.savez(
            os.path.join(sim_settings.save_folder, data_filename),
            time=time,
            muscle_rods_position_history=muscle_rods_position_history,
            muscle_rods_radius_history=muscle_rods_radius_history,
        )

    if sim_settings.return_data:
        return post_processing_dict_list
