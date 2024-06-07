from elastica import *
import numpy as np
from examples.ArtificialMusclesCases.muscle.muscle_fiber_init_symbolic import (
    get_fiber_geometry,
)
from examples.ArtificialMusclesCases.muscle.muscle_utils import (
    helix_arclength,
)
from examples.ArtificialMusclesCases.post_processing._callback import (
    MuscleCallBack,
)

from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)

from examples.ArtificialMusclesCases.muscle.connect_straight_rods import (
    ContactSurfaceJoint,
    SurfaceJointSideBySide,
    SurfaceJointSideBySideTwo,
    get_connection_vector_straight_straight_rod_with_rest_matrix,
)
from examples.ArtificialMusclesCases.muscle.muscle_forcing import PointSpring


class CoiledMuscle:
    """
    This class initilizes a coiled muscle out of a collection of coiled cosserat rods.

        Attributes
        ----------


    """

    def __init__(
        self,
        muscle_geometry,
        muscle_properties,
        muscle_sim_settings,
    ):

        """

        Parameters
        ----------
        muscle_geometry: dict
            Dictionary containing muscle geometric properties.
        muscle_properties: dict
            Dictionary containing muscle physcial properties.
        sim_settings: dict
            Dictionary containing simulation settings.

        """
        self.muscle_sim_settings = muscle_sim_settings
        self.muscle_properties = muscle_properties
        self.muscle_geometry = muscle_geometry
        self.n_turns = (
            muscle_geometry.turns_per_length_list[0] * muscle_geometry.muscle_length
        )
        self.n_elem = int(muscle_sim_settings.n_elem_per_coil * self.n_turns)
        self.muscle_rods = {}
        self.muscle_rods_start_actuation_properties = {}

        self.rod_ids = [
            ("rod_id", l) for l in range(muscle_geometry.n_ply_per_coil_level[0])
        ]
        self.n_coil_levels = len(muscle_geometry.n_ply_per_coil_level)

        # create unique rod ids for each fiber
        i = 1
        while i < self.n_coil_levels:
            self.rod_ids = [
                (*a, b)
                for a in self.rod_ids
                for b in range(muscle_geometry.n_ply_per_coil_level[i])
            ]
            i += 1

        for rod_id in self.rod_ids:
            fiber_offset_list = np.zeros((self.n_coil_levels,))
            fiber_offset_list[0] = muscle_geometry.angular_offset
            for coil_level in range(1, self.n_coil_levels):
                fiber_offset_list[coil_level] = (
                    2
                    * np.pi
                    * rod_id[coil_level + 1]
                    / muscle_geometry.n_ply_per_coil_level[coil_level]
                )
            # initialize start actuation properties
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_kappa")
            ] = np.zeros((3, self.n_elem - 1))
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_sigma")
            ] = np.zeros((3, self.n_elem))
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_bend_matrix")
            ] = np.zeros((3, 3, self.n_elem - 1))
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_shear_matrix")
            ] = np.zeros((3, 3, self.n_elem))
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_mass_second_moment_of_inertia")
            ] = np.zeros((3, 3, self.n_elem))
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_inv_mass_second_moment_of_inertia")
            ] = np.zeros((3, 3, self.n_elem))

            # get element positions and directors of each fiber
            (
                fiber_length,
                fiber_start,
                fiber_position_collection,
                fiber_director_collection,
                intrinsic_link,
                injected_twist,
            ) = get_fiber_geometry(
                n_elem=self.n_elem,
                start_radius_list=muscle_geometry.start_radius_list,
                taper_slope_list=muscle_geometry.taper_slope_list,
                start_position=muscle_geometry.start_position,
                direction=muscle_geometry.direction,
                normal=muscle_geometry.normal,
                offset_list=fiber_offset_list,
                length=muscle_geometry.muscle_length,
                turns_per_length_list=muscle_geometry.turns_per_length_list,
                initial_link_per_fiber_length=muscle_geometry.initial_link_per_fiber_length,
                CCW_list=muscle_geometry.CCW_list,
                check_twist_difference=True,
            )

            # intialize constrain properties
            self.muscle_rods_start_actuation_properties[
                (rod_id, "constrain_start_positions")
            ] = np.zeros_like(fiber_position_collection)
            self.muscle_rods_start_actuation_properties[
                (rod_id, "constrain_start_directors")
            ] = np.zeros_like(fiber_director_collection)

            # construct muscle fiber
            self.muscle_rods[rod_id] = CosseratRod.straight_rod(
                self.n_elem,
                fiber_start,
                muscle_geometry.direction,
                muscle_geometry.normal,
                fiber_length,
                muscle_geometry.fiber_radius,
                muscle_properties.density,
                youngs_modulus=muscle_properties.youngs_modulus,
                shear_modulus=muscle_properties.shear_modulus,
                position=fiber_position_collection,
                directors=fiber_director_collection,
            )

    def append_muscle_to_sim(self, simulation):
        """Appends muscle rods to a simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.append(self.muscle_rods[rod_id])

    def dampen_muscle(self, simulation, *args, **kwargs):
        """Adds damping to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.dampen(self.muscle_rods[rod_id]).using(
                *args,
                **kwargs,
            )

    def fix_shape_and_store_start_properties(self):
        """Fixes the shape of the muscle rods in simulation and stores start actuation properties after finalizing the sim.


        Parameters
        ----------

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            # store start actuation properties
            self.muscle_rods_start_actuation_properties[(rod_id, "start_kappa")][
                :
            ] = self.muscle_rods[rod_id].kappa[:]
            self.muscle_rods_start_actuation_properties[(rod_id, "start_sigma")][
                :
            ] = self.muscle_rods[rod_id].sigma[:]
            self.muscle_rods_start_actuation_properties[(rod_id, "start_shear_matrix")][
                :
            ] = self.muscle_rods[rod_id].shear_matrix[:]
            self.muscle_rods_start_actuation_properties[(rod_id, "start_bend_matrix")][
                :
            ] = self.muscle_rods[rod_id].bend_matrix[:]
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_mass_second_moment_of_inertia")
            ][:] = self.muscle_rods[rod_id].mass_second_moment_of_inertia[:]
            self.muscle_rods_start_actuation_properties[
                (rod_id, "start_inv_mass_second_moment_of_inertia")
            ][:] = self.muscle_rods[rod_id].inv_mass_second_moment_of_inertia[:]
            self.muscle_rods_start_actuation_properties[
                (rod_id, "constrain_start_positions")
            ][:] = self.muscle_rods[rod_id].position_collection[:]
            self.muscle_rods_start_actuation_properties[
                (rod_id, "constrain_start_directors")
            ][:] = self.muscle_rods[rod_id].director_collection[:]

            # fix muscle shape
            self.muscle_rods[rod_id].rest_kappa[:] = self.muscle_rods[rod_id].kappa[:]
            self.muscle_rods[rod_id].rest_kappa[:] = self.muscle_rods[rod_id].kappa[:]
            # self.muscle_rods[rod_id].shear_matrix[2,2,:] *= 7e1

    def muscle_callback(self, simulation, post_processing_dict_list, step_skip):
        """Adds callback to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        post_processing_dict_list: List
            list to store callback dictionary for each fiber

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            post_processing_dict_list.append(
                defaultdict(list)
            )  # list which collected data will be append
            # set the diagnostics for rod and collect data
            simulation.collect_diagnostics(self.muscle_rods[rod_id]).using(
                MuscleCallBack,
                step_skip=step_skip,
                callback_params=post_processing_dict_list[
                    len(post_processing_dict_list) - 1
                ],
            )

    def add_forcing_to_muscle(self, simulation, *args, **kwargs):
        """Adds forcing to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        args:
            forcing arguments
        kwargs:
            forcing keyword arguments

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.add_forcing_to(self.muscle_rods[rod_id]).using(*args, **kwargs)

    def connect_muscle_to_point(self, simulation, nu, point, index, *args, **kwargs):
        """connect muscle rods to a point in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        nu:
            Damping constant
        point:
            point to connect the muscle to
        index:
            node index where rods connect
        args:
            forcing arguments
        kwargs:
            forcing keyword arguments

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.add_forcing_to(self.muscle_rods[rod_id]).using(
                PointSpring,
                k=self.muscle_rods[rod_id].shear_matrix[2, 2, -1] * 10,
                nu=nu,
                point=point,
                index=index,
                *args,
                **kwargs,
            )

    def connect_to_rod(
        self,
        simulation,
        rod,
        joint,
        first_connect_idx,
        second_connect_idx,
        *args,
        **kwargs,
    ):
        """Adds connection between muscle rods and external rod in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        rod:
            rod-like object
        first_connect_idx:
            first connection idx
        second_connect_idx:
            second connection idx
        args:
            connection arguments
        kwargs:
            connection keyword arguments

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.connect(
                first_rod=self.muscle_rods[rod_id],
                second_rod=rod,
                first_connect_idx=first_connect_idx,
                second_connect_idx=second_connect_idx,
            ).using(joint, *args, **kwargs)

    def constrain_muscle(self, simulation, *args, **kwargs):
        """Adds constraint to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        args:
            constrain arguments
        kwargs:
            constrain keyword arguments

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.constrain(self.muscle_rods[rod_id]).using(
                *args,
                constrain_start_positions=self.muscle_rods_start_actuation_properties[
                    (rod_id, "constrain_start_positions")
                ],
                constrain_start_directors=self.muscle_rods_start_actuation_properties[
                    (rod_id, "constrain_start_directors")
                ],
                **kwargs,
            )

    def actuate(self, simulation, *args, **kwargs):
        """Adds forcing to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        args:
            actuation arguments
        kwargs:
            actuation keyword arguments

        Returns
        -------

        """
        for rod_id in self.rod_ids:
            simulation.add_forcing_to(self.muscle_rods[rod_id]).using(
                *args,
                start_density=self.muscle_properties.density,
                start_radius=self.muscle_geometry.fiber_radius,
                start_kappa=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_kappa")
                ],
                start_sigma=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_sigma")
                ],
                start_bend_matrix=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_bend_matrix")
                ],
                start_shear_matrix=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_shear_matrix")
                ],
                start_mass_second_moment_of_inertia=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_mass_second_moment_of_inertia")
                ],
                start_inv_mass_second_moment_of_inertia=self.muscle_rods_start_actuation_properties[
                    (rod_id, "start_inv_mass_second_moment_of_inertia")
                ],
                **kwargs,
            )

    def apply_self_contact(self, simulation, self_contact_settings):
        """Apply self contact to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        Returns
        -------

        """
        # contact_radius = int(self.muscle_sim_settings.n_elem_per_coil/10)
        contact_radius = int(
            self_contact_settings.contact_radius_ratio
            * self.muscle_sim_settings.n_elem_per_coil
        )
        for rod_id in self.rod_ids:
            for elem in range(self.n_elem):
                contact_range = (
                    max(
                        0,
                        elem
                        + self.muscle_sim_settings.n_elem_per_coil
                        - contact_radius,
                    ),
                    min(
                        self.n_elem,
                        elem
                        + self.muscle_sim_settings.n_elem_per_coil
                        + contact_radius,
                    ),
                )
                for contact_elem in range(contact_range[0], contact_range[1]):
                    (
                        rod_one_direction_vec_in_material_frame,
                        rod_two_direction_vec_in_material_frame,
                        offset_btw_rods,
                    ) = get_connection_vector_straight_straight_rod(
                        self.muscle_rods[rod_id],
                        self.muscle_rods[rod_id],
                        (elem, elem + 1),
                        (contact_elem, contact_elem + 1),
                    )

                    simulation.connect(
                        first_rod=self.muscle_rods[rod_id],
                        second_rod=self.muscle_rods[rod_id],
                        first_connect_idx=elem,
                        second_connect_idx=contact_elem,
                    ).using(
                        ContactSurfaceJoint,
                        k=self_contact_settings.k_val
                        * self.muscle_rods[rod_id].shear_matrix[2, 2, elem - 1],
                        nu=self_contact_settings.nu,
                        k_repulsive=self_contact_settings.k_repulsive_val
                        * self.muscle_rods[rod_id].shear_matrix[
                            2, 2, elem - 1
                        ],  # self.muscle_rods[rod_id].shear_matrix[2,2,elem-1]*1e2,
                        rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                            ..., 0
                        ],
                        rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                            ..., 0
                        ],
                        offset_btw_rods=2 * (self.muscle_rods[rod_id].radius)[0],
                        separation_distance=self_contact_settings.separation_distance,
                    )

    def connect_muscle_rods(self, simulation, connection_settings):
        """Apply parallel connection to adjacent rods to muscle rods in simulation.


        Parameters
        ----------
        simulation :
            Elastica Simulation
        Returns
        -------

        """
        if len(self.rod_ids) > 1:
            rod_id_pairs = [
                (a, b)
                for idx, a in enumerate(self.rod_ids)
                for b in self.rod_ids[idx + 1 :]
            ]

            # Connect the three fibers in each supercoil
            # k_val = 6e0# 1.5e2
            # k_repulsive_val = 1e1#1e1
            rod_one_direction_vec_in_material_frame_above = {}
            rod_one_direction_vec_in_material_frame_below = {}
            rod_two_direction_vec_in_material_frame_above = {}
            rod_two_direction_vec_in_material_frame_below = {}
            offset_btw_rods_above = {}
            offset_btw_rods_below = {}
            rest_matrix_above = {}
            rest_matrix_below = {}

            for pair in rod_id_pairs:
                (
                    rod_one_direction_vec_in_material_frame,
                    rod_two_direction_vec_in_material_frame,
                    offset_btw_rods,
                    rest_matrix,
                ) = get_connection_vector_straight_straight_rod_with_rest_matrix(
                    self.muscle_rods[pair[0]],
                    self.muscle_rods[pair[1]],
                    (0, self.n_elem),
                    (0, self.n_elem),
                )
                for i in range(connection_settings.connection_range):
                    (
                        rod_one_direction_vec_in_material_frame_above[(pair, i)],
                        rod_two_direction_vec_in_material_frame_above[(pair, i)],
                        offset_btw_rods_above[(pair, i)],
                        rest_matrix_above[(pair, i)],
                    ) = get_connection_vector_straight_straight_rod_with_rest_matrix(
                        self.muscle_rods[pair[0]],
                        self.muscle_rods[pair[1]],
                        (0, self.n_elem - 1 - i),
                        (1 + i, self.n_elem),
                    )
                    (
                        rod_one_direction_vec_in_material_frame_below[(pair, i)],
                        rod_two_direction_vec_in_material_frame_below[(pair, i)],
                        offset_btw_rods_below[(pair, i)],
                        rest_matrix_below[(pair, i)],
                    ) = get_connection_vector_straight_straight_rod_with_rest_matrix(
                        self.muscle_rods[pair[0]],
                        self.muscle_rods[pair[1]],
                        (1 + i, self.n_elem),
                        (0, self.n_elem - 1 - i),
                    )

                for elem in range(self.n_elem):
                    # if abs(offset_btw_rods[elem]) > 2.0*self.muscle_geometry.start_radius_list[-1]:
                    #     continue
                    simulation.connect(
                        first_rod=self.muscle_rods[pair[0]],
                        second_rod=self.muscle_rods[pair[1]],
                        first_connect_idx=elem,
                        second_connect_idx=elem,
                    ).using(
                        SurfaceJointSideBySide,
                        k=connection_settings.k_val,
                        nu=connection_settings.nu,
                        k_repulsive=connection_settings.k_repulsive_val
                        * self.muscle_rods[pair[0]].shear_matrix[2, 2, elem - 1],
                        friction_coefficient=connection_settings.friction_coefficient,
                        velocity_damping_coefficient=connection_settings.velocity_damping_coefficient,
                        rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                            :, elem
                        ],
                        rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                            :, elem
                        ],
                        offset_btw_rods=offset_btw_rods[elem],
                        rest_rotation_matrix=rest_matrix[:, :, elem],
                    )
                    for i in range(connection_settings.connection_range):
                        if elem > i:
                            simulation.connect(
                                first_rod=self.muscle_rods[pair[0]],
                                second_rod=self.muscle_rods[pair[1]],
                                first_connect_idx=elem,
                                second_connect_idx=elem - i - 1,
                            ).using(
                                SurfaceJointSideBySide,
                                k=connection_settings.k_val,
                                nu=connection_settings.nu,
                                k_repulsive=connection_settings.k_repulsive_val
                                * self.muscle_rods[pair[0]].shear_matrix[
                                    2, 2, elem - i - 1
                                ],
                                friction_coefficient=connection_settings.friction_coefficient,
                                velocity_damping_coefficient=connection_settings.velocity_damping_coefficient,
                                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame_below[
                                    (pair, i)
                                ][
                                    :, elem - i - 1
                                ],
                                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame_below[
                                    (pair, i)
                                ][
                                    :, elem - i - 1
                                ],
                                offset_btw_rods=offset_btw_rods_below[(pair, i)][
                                    elem - i - 1
                                ],
                                rest_rotation_matrix=rest_matrix_below[(pair, i)][
                                    :, :, elem - i - 1
                                ],
                            )
                        if elem < self.n_elem - 1 - i:
                            simulation.connect(
                                first_rod=self.muscle_rods[pair[0]],
                                second_rod=self.muscle_rods[pair[1]],
                                first_connect_idx=elem,
                                second_connect_idx=elem + 1 + i,
                            ).using(
                                SurfaceJointSideBySide,
                                k=connection_settings.k_val,
                                nu=connection_settings.nu,
                                k_repulsive=connection_settings.k_repulsive_val
                                * self.muscle_rods[pair[0]].shear_matrix[
                                    2, 2, elem - 1
                                ],
                                friction_coefficient=connection_settings.friction_coefficient,
                                velocity_damping_coefficient=connection_settings.velocity_damping_coefficient,
                                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame_above[
                                    (pair, i)
                                ][
                                    :, elem
                                ],
                                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame_above[
                                    (pair, i)
                                ][
                                    :, elem
                                ],
                                offset_btw_rods=offset_btw_rods_above[(pair, i)][elem],
                                rest_rotation_matrix=rest_matrix_above[(pair, i)][
                                    :, :, elem
                                ],
                            )
