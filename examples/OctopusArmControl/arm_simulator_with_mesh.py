import numpy as np
from elastica import *
from elastica.contact_forces import RodCylinderContact
from elastica.external_forces import NoForces
from drag_force import DragForce
from elastica.rod.cosserat_rod import (
    _compute_shear_stretch_strains,
    _compute_bending_twist_strains,
)
from CustomFrictionSurface import RodMeshSurfaceContactWithGrid
import matplotlib.pyplot as plt
import pickle
from examples.OctopusArmControl.actuations.muscles import (
    LongitudinalMuscle,
    TransverseMuscle,
    ObliqueMuscle,
    ApplyMuscleGroups,
    MuscleGroup,
    force_length_weight_poly,
)
from elastica._calculus import _isnan_check
from post_processing import plot_video_with_surface


def z_rotation(zrotation):
    z1Rot = np.array(
        [
            [np.cos(zrotation), -np.sin(zrotation), 0],
            [np.sin(zrotation), np.cos(zrotation), 0],
            [0, 0, 1],
        ]
    )
    return z1Rot


def y_rotation(yrotation):
    yRot = np.array(
        [
            [np.cos(yrotation), 0, np.sin(yrotation)],
            [0, 1, 0],
            [-np.sin(yrotation), 0, np.cos(yrotation)],
        ]
    )
    return yRot


def x_rotation(xrotation):
    xRot = np.array(
        [
            [1, 0, 0],
            [0, np.cos(xrotation), -np.sin(xrotation)],
            [0, np.sin(xrotation), np.cos(xrotation)],
        ]
    )
    return xRot


def do_normalization(data, limit):
    return (data - limit[0]) / (limit[1] - limit[0]) * 2 - 1


def do_denormalization(data, limit):
    return (data + 1) / 2 * (limit[1] - limit[0]) + limit[0]


class ParameterDictionary(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__.update(kwargs)


class BallCallBack(CallBackBaseClass):
    """
    Call back function for the ball
    """

    def __init__(self, step_skip: int, callback_params: dict, radius):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params
        self.callback_params["radius"] = radius

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())


class ArmCallBack(CallBackBaseClass):
    """
    Call back function for the arm
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["angular_velocity"].append(
                system.omega_collection.copy()
            )
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass()
            )
            self.callback_params["kappa"].append(system.kappa.copy())
            self.callback_params["sigma"].append(
                system.sigma.copy()  # + np.array([0, 0, 1])[:, None]
            )
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())

        return


class HeadCallBack(CallBackBaseClass):
    """
    Call back function for the arm
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
        return


class SuckerForce(NoForces):
    def __init__(self, sucker, n_p, n_d, k):
        self.sucker = sucker
        self.fixed_pos = n_p
        self.fixed_dir = n_d
        self.k = k
        self.sucker_time = 0.0

    def apply_forces(self, system, time=0.0):
        if not self.sucker:
            self.sucker_time = 0.0
            return
        factor = min(1.0, self.sucker_time)
        # Update external forces
        force = (
            -factor * self.k * (system.position_collection[..., -1] - self.fixed_pos)
        )
        system.external_forces[..., -1] += force
        self.sucker_time += 0.001


class OneArmSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping, Contact
):
    pass


class ArmSimulator:
    def __init__(self, env_config, task, callback):
        """General setup"""
        self.n_arm = env_config["num_arms"]
        self.callback = callback
        self.timestepper = PositionVerlet()
        self.recording_fps = 30

        """ Task """
        self.reset = (
            self.locomotion_reset
            if "locomotion" in env_config["env_name"]
            else self.manipulation_reset
        )
        # self.sucker = task.sucker
        # self.obstacle = task.obstacle

        # self.arm_states = np.zeros((self.n_arm, task.arm_state_dim))
        # self.state_type = task.state_type
        # if task.state_type == 'pos':
        #     self.pos = np.zeros((self.n_arm, self.n_elem * 2))
        #     self.prev_pos = np.zeros((self.n_arm, self.n_elem * 2))
        #     if task.arm_state_dim < 400:
        #         self.cal_arm_state = self.cal_mean_pos_arm_state
        #     else:
        #         self.cal_arm_state = self.cal_pos_arm_state
        # else:
        #     self.kappa = np.zeros((self.n_arm, self.n_elem - 1))
        #     self.prev_kappa = np.zeros((self.n_arm, self.n_elem - 1))
        #     if task.arm_state_dim < 198:
        #         self.cal_arm_state = self.cal_mean_arm_state
        #     else:
        #         self.cal_arm_state = self.cal_all_arm_state
        self.target_radius = env_config["target_radius"]
        self.obstacle_radius = env_config["target_radius"] * 0.2

        """ Rest lengths and normalization parameters """
        # arm_info = np.load(env_config['arm_info'])
        self.kappa_range = [-140, 140]  # arm_info['kappa_range']#
        self.kappa_rate_range = [-35, 35]  # arm_info['kappa_rate_range']#
        # self.rest_lengths = arm_info['rest_lengths']
        # self.rest_voronoi_lengths = arm_info['rest_voronoi_lengths']

        """ Arm """
        self.base_length = env_config["base_length"]
        self.n_elem = env_config["n_elem"]
        self.node_skip = env_config["node_skip"]
        self.step_skip = env_config["step_skip"]
        self.arm_dt = env_config["arm_dt"]
        self.final_time = env_config["final_time"]
        self.total_steps = int(self.final_time / self.arm_dt)
        self.angle_list = (
            np.array([360 / self.n_arm * i for i in range(self.n_arm)]) / 180 * np.pi
        )  #

        self.start = np.array([0.0, 0.0, 0.0])
        self.direction = np.array([1.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, -1.0])

        radius_base = self.base_length / 20
        radius_tip = radius_base / 10
        radius = np.linspace(radius_base, radius_tip, self.n_elem + 1)
        self.base_radius = (radius[:-1] + radius[1:]) / 2
        self.density = 1042
        damping = 0.05
        damping_torque = 0.01
        self.dissipation = damping * ((self.base_radius / radius_base) ** 2)
        self.nu_torque = damping_torque * ((self.base_radius / radius_base) ** 4)
        self.E = 1e4
        self.poisson_ratio = 1

        """ Mesh Surface """
        mesh = Mesh(filepath=env_config["mesh_path"])
        mesh.translate(-np.array(mesh.mesh_center))
        mesh.translate(
            -4 * radius_base - np.array([0, 0, np.min(mesh.face_centers[2])])
        )
        # mesh.visualize()
        self.mesh_surface = MeshSurface(mesh)

        """ Set up contact between arm and mesh surface """
        self.gravitational_acc = -9.80665
        mu = 0.4
        self.kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
        filename = env_config["mesh_grid_path"]
        with open(filename, "rb") as fptr:
            self.faces_grid = pickle.load(fptr)

        """ Muscles """
        n_max = np.array([1, 1, -2])
        self.n_muscle = len(n_max)  # number of muscle fiber
        max_force = (self.base_radius / radius_base) ** 2 * n_max[:, None]
        passive_ratio = 1
        radius_ratio = np.array([1, -1, 0])
        self.muscle_groups = []
        self.muscle_callback_params_list = []

        self.muscle_fiber = ParameterDictionary(
            n_muscle=self.n_muscle,
            n_max=n_max,
            max_force=max_force,
            passive_ratio=passive_ratio,
            radius_ratio=radius_ratio,
        )

        """ Flow """
        self.rho_water = 1022
        self.c_per = 0.5 * self.rho_water * env_config["arm_c_per"]
        self.c_tan = 0.5 * self.rho_water * env_config["arm_c_tan"]
        # self.cylinder_drag_coeff = env_config['cylinder_drag_coeff']

    def add_damping(self, rod):
        damping_constant = 0.05 / 20  # 0.01 / 1.5
        self.one_arm_fixed_sim.dampen(rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.arm_dt,
        )
        # self.one_arm_fixed_sim.dampen(rod).using(
        #     LaplaceDissipationFilter,
        #     filter_order=5,
        # )

    def set_muscles(self, base_radius, arm):
        """Add muscle actuation"""
        muscle_groups = []

        LM_ratio_muscle_position = 0.0075 / base_radius
        OM_ratio_muscle_position = 0.01125 / base_radius

        AN_ratio_radius = 0.002 / base_radius
        TM_ratio_radius = 0.0045 / base_radius
        LM_ratio_radius = 0.003 / base_radius
        OM_ratio_radius = 0.00075 / base_radius

        OM_rotation_number = 6

        shearable_rod_area = np.pi * arm.radius ** 2
        TM_rest_muscle_area = shearable_rod_area * (
            TM_ratio_radius ** 2 - AN_ratio_radius ** 2
        )
        LM_rest_muscle_area = shearable_rod_area * (LM_ratio_radius ** 2)
        OM_rest_muscle_area = shearable_rod_area * (OM_ratio_radius ** 2)

        # stress is in unit [Pa]
        TM_max_muscle_stress = 15_000.0
        LM_max_muscle_stress = 10_000.0
        OM_max_muscle_stress = 100_000.0

        muscle_dict = dict(
            force_length_weight=force_length_weight_poly,
        )

        # Add a transverse muscle
        muscle_groups.append(
            MuscleGroup(
                muscles=[
                    TransverseMuscle(
                        rest_muscle_area=TM_rest_muscle_area,
                        max_muscle_stress=TM_max_muscle_stress,
                        **muscle_dict,
                    )
                ],
                type_name="TM",
            )
        )

        # Add 4 longitudinal muscles
        for k in range(4):
            muscle_groups.append(
                MuscleGroup(
                    muscles=[
                        LongitudinalMuscle(
                            muscle_init_angle=np.pi * 0.5 * k,
                            ratio_muscle_position=LM_ratio_muscle_position,
                            rest_muscle_area=LM_rest_muscle_area,
                            max_muscle_stress=LM_max_muscle_stress,
                            **muscle_dict,
                        )
                    ],
                    type_name="LM",
                )
            )

        # Add a clockwise oblique muscle group (4 muscles)
        muscle_groups.append(
            MuscleGroup(
                muscles=[
                    ObliqueMuscle(
                        muscle_init_angle=np.pi * 0.5 * m,
                        ratio_muscle_position=OM_ratio_muscle_position,
                        rotation_number=OM_rotation_number,
                        rest_muscle_area=OM_rest_muscle_area,
                        max_muscle_stress=OM_max_muscle_stress,
                        **muscle_dict,
                    )
                    for m in range(4)
                ],
                type_name="OM",
            )
        )

        # Add a counter-clockwise oblique muscle group (4 muscles)
        muscle_groups.append(
            MuscleGroup(
                muscles=[
                    ObliqueMuscle(
                        muscle_init_angle=np.pi * 0.5 * m,
                        ratio_muscle_position=OM_ratio_muscle_position,
                        rotation_number=-OM_rotation_number,
                        rest_muscle_area=OM_rest_muscle_area,
                        max_muscle_stress=OM_max_muscle_stress,
                        **muscle_dict,
                    )
                    for m in range(4)
                ],
                type_name="OM",
            )
        )

        for muscle_group in muscle_groups:
            muscle_group.set_current_length_as_rest_length(arm)

        return muscle_groups

    def set_mesh_surface_arm_contact(self, shearable_rod):
        """Set up contact between arms and mesh surface"""
        self.one_arm_fixed_sim.add_forcing_to(shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, self.gravitational_acc])
        )
        static_mu_array = np.zeros(self.kinetic_mu_array.shape)
        self.one_arm_fixed_sim.detect_contact_between(
            shearable_rod, self.mesh_surface
        ).using(
            RodMeshSurfaceContactWithGrid,
            k=1e1,
            nu=1e-2,
            slip_velocity_tol=1e-8,
            static_mu_array=static_mu_array,
            kinetic_mu_array=self.kinetic_mu_array,
            faces_grid=self.faces_grid,
            gamma=0.1,
        )

    def manipulation_reset(
        self, rod_start_list, ball_start, ball_vel=[0.0, 0.0], obstacle_start=None
    ):
        """Reset parameters"""
        self.u = np.zeros([self.n_arm, self.n_muscle, self.n_elem + 1])
        # self.arm_states[...] = 0.0
        self.head_list = None

        """ Set up arm simulator """
        self.one_arm_fixed_sim = OneArmSimulator()
        self.one_arm_fixed_sim.append(self.mesh_surface)
        self.shearable_rods = []
        for i_arm in range(self.n_arm):
            R = x_rotation(self.angle_list[i_arm])
            rod_start = rod_start_list[i_arm]
            rod = CosseratRod.straight_rod(
                self.n_elem,
                self.start,
                self.direction,
                self.normal,
                self.base_length,
                self.base_radius,
                self.density,
                # 0.0,  # self.dissipation,
                youngs_modulus=self.E,
                shear_modulus=self.E / (1 + self.poisson_ratio),
                position=R @ rod_start["pos"] + self.start[:, np.newaxis],
                directors=R @ rod_start["dir"],
            )

            # rod.rest_lengths[...] = self.rest_lengths
            # rod.rest_voronoi_lengths[...] = self.rest_voronoi_lengths
            # rod.velocity_collection[...] = rod_start['vel']
            # rod.omega_collection[...] = rod_start['omega']
            self.shearable_rods.append(rod)
            self.one_arm_fixed_sim.append(self.shearable_rods[-1])

        """ Add damping """
        for i_arm in range(self.n_arm):
            self.add_damping(self.shearable_rods[i_arm])

        """ Add ball (cylinder)"""
        if ball_start is not None:
            cylinder_length = 0.015  # 0.4 * self.base_length
            self.cylinder = Cylinder(
                start=np.array([ball_start[0], ball_start[1], -0.5 * cylinder_length])
                + self.start,
                direction=np.array([0, 0, 1]),
                normal=np.array([1, 0, 0]),
                base_length=cylinder_length,
                base_radius=0.05 * self.base_length,
                density=750,
            )

            self.one_arm_fixed_sim.append(self.cylinder)

            self.cylinder.velocity_collection[0] = ball_vel[0]
            self.cylinder.velocity_collection[1] = ball_vel[1]

            # self.one_arm_fixed_sim.add_forcing_to(self.cylinder).using(
            #     RigidBodyDragForce,
            #     self.fluid_param,
            #     self.cylinder.n_elems,
            #     self.cylinder_drag_coeff,
            # )

            sucker_k = 1e1
            for i_arm in range(self.n_arm):
                self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                    SuckerForce,
                    sucker=self.sucker_control[i_arm, :],
                    n_p=self.sucker_fixed_position[i_arm, :],
                    n_d=self.sucker_fixed_director[i_arm, ...],
                    k=sucker_k,
                )

            self.arm_ball_strike = np.full((self.n_arm, 1), False)
            for i_arm in range(self.n_arm):
                self.one_arm_fixed_sim.detect_contact_between(
                    self.shearable_rods[i_arm], self.cylinder
                ).using(
                    RodCylinderContact,
                    k=5e0,
                    nu=1e-1,  # 1e2,  # , arm_ball_strike=self.arm_ball_strike[i_arm, :]
                    # velocity_damping_coefficient=0.83*0.65, friction_coefficient=0.65
                    #     http://sssa.bioroboticsinstitute.it/sites/default/files/user_docs/LM2014_CalistiCorucci_BipedalWalking.pdf
                )

        """ Add obstacle """
        if self.obstacle:
            self.cylinder_obs = Cylinder(
                start=np.array([obstacle_start[0], obstacle_start[1], 0.0]),
                direction=np.array([0, 1, 0]),
                normal=np.array([1, 0, 0]),
                base_length=1.0 * self.base_length,
                base_radius=self.obstacle_radius,
                density=200,
            )

            self.one_arm_fixed_sim.append(self.cylinder_obs)

        """  Add callbacks """
        if self.callback:
            self.pp_lists = []
            for i_arm in range(self.n_arm):
                pp_list = defaultdict(list)
                self.pp_lists.append(pp_list)
                self.one_arm_fixed_sim.collect_diagnostics(
                    self.shearable_rods[i_arm]
                ).using(
                    ArmCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.pp_lists[-1],
                )
            if ball_start is not None:
                self.ball_list = defaultdict(list)
                self.one_arm_fixed_sim.collect_diagnostics(self.cylinder).using(
                    BallCallBack,
                    step_skip=self.step_skip,
                    callback_params=self.ball_list,
                    radius=self.cylinder.radius,
                )

        """  Add fluid drag forces """
        # for i_arm in range(self.n_arm):
        #     self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
        #         DragForce,
        #         rho_environment=self.rho_water,
        #         c_per=self.c_per,
        #         c_tan=self.c_tan,
        #         system=self.shearable_rods[i_arm],
        #         step_skip=self.step_skip,
        #         callback_params=self.pp_lists[i_arm],  # self.rod_parameters_dict
        # )
        """ Create a continuous muscle model """
        # muscle = ContinuousActuation(self.n_elem, self.muscle_fiber)
        for i_arm in range(self.n_arm):
            self.muscle_groups.append(
                self.set_muscles(self.base_radius, self.shearable_rods[i_arm])
            )
            self.muscle_callback_params_list.append(
                [defaultdict(list) for _ in self.muscle_groups]
            )
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                ApplyMuscleGroups,
                muscle_groups=self.muscle_groups[i_arm],
                step_skip=self.step_skip,
                callback_params_list=self.muscle_callback_params_list[i_arm],
            )
        #     self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
        #         SensoryMuscleCtrl,
        #         self.u[i_arm],
        #         muscle,
        #         # target,
        #         # shoot_angle,
        #         # shear_matrix,
        #         # bend_matrix,
        #         ramp_up_time=0.5,
        #     )

        """Add arm mesh surface contact"""
        for i_arm in range(self.n_arm):
            self.set_mesh_surface_arm_contact(self.shearable_rods[i_arm])

        """  Finalize """
        self.one_arm_fixed_sim.finalize()
        for i_arm in range(self.n_arm):
            self.compute_strains(self.shearable_rods[i_arm])

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.timestepper, self.one_arm_fixed_sim
        )
        # if self.state_type == 'pos':
        #     self.pos[...] = [rod.position_collection[:2, 1:].flatten() for rod in self.shearable_rods]
        #     self.prev_pos[...] = rod_start['prev_pos']
        # else:
        #     self.kappa[...] = [rod.kappa[0] for rod in self.shearable_rods]
        #     self.prev_kappa[...] = rod_start['prev_kappa']

    def compute_strains(self, rod):
        _compute_shear_stretch_strains(
            rod.position_collection,
            rod.volume,
            rod.lengths,
            rod.tangents,
            rod.radius,
            rod.rest_lengths,
            rod.rest_voronoi_lengths,
            rod.dilatation,
            rod.voronoi_dilatation,
            rod.director_collection,
            rod.sigma,
        )

        # Compute bending twist strains
        _compute_bending_twist_strains(
            rod.director_collection, rod.rest_voronoi_lengths, rod.kappa
        )

    def make_straight_crawling_arms(self):
        reference_rod_pos = np.zeros((3, 101))
        reference_rod_pos[0, ...] = np.linspace(0.0, 0.2, 101)
        reference_rod_dir = np.zeros((3, 3, 100))
        reference_rod_dir[0, 2, ...] = -1
        reference_rod_dir[1, 1, ...] = 1
        reference_rod_dir[2, 0, ...] = 1
        for i_arm in range(self.n_arm):
            R = z_rotation(self.angle_list[i_arm])
            rod = CosseratRod.straight_rod(
                self.n_elem,
                self.start,
                self.direction,
                self.normal,
                self.base_length,
                self.base_radius,
                self.density,
                # 0.0,  # self.dissipation,
                youngs_modulus=self.E,
                shear_modulus=self.E / (1 + self.poisson_ratio),
                position=R @ reference_rod_pos + self.start[:, np.newaxis],
                directors=R @ reference_rod_dir,
            )

            # rod.rest_lengths[...] = self.rest_lengths
            # rod.rest_voronoi_lengths[...] = self.rest_voronoi_lengths

            self.shearable_rods.append(rod)
            self.one_arm_fixed_sim.append(self.shearable_rods[-1])

    def locomotion_reset(self):
        """Reset parameters"""
        self.u = np.zeros([self.n_arm, self.n_muscle, self.n_elem + 1])
        # self.arm_states[...] = 0.0

        """ Set up arm simulator """
        self.one_arm_fixed_sim = OneArmSimulator()
        self.one_arm_fixed_sim.append(self.mesh_surface)
        self.shearable_rods = []

        self.angle_list = (
            np.array(
                [360 / self.n_arm * i + 360 / self.n_arm / 2 for i in range(self.n_arm)]
            )
            / 180
            * np.pi
        )

        self.make_straight_crawling_arms()
        # self.make_straight_symmetric_crawling_arms()
        # self.make_curved_symmetric_crawling_arms(rod_start_list)

        """ Constraints on arms' rotations """
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.constrain(self.shearable_rods[i_arm]).using(
                GeneralConstraint,
                # constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                # translational_constraint_selector = np.array([True, True, True]),
                rotational_constraint_selector=np.array([True, True, True]),
            )

        """ Add head """
        slenderness_ratio_head = 5.787
        arm_radius = self.base_length / 20 / 2
        octopus_head_length = 2 * arm_radius * slenderness_ratio_head / 2
        octopus_head_n_elems = int(self.n_elem * octopus_head_length / self.base_length)

        octopus_head_radius = (
            2 * arm_radius * np.linspace(0.9, 1.0, octopus_head_n_elems) ** 3
        )
        self.body_rod = CosseratRod.straight_rod(
            n_elements=octopus_head_n_elems,
            start=self.start,
            direction=-self.direction,
            normal=self.normal,
            base_length=octopus_head_length,
            base_radius=octopus_head_radius,
            density=self.density,
            youngs_modulus=self.E,
            shear_modulus=self.E / (1 + self.poisson_ratio),
        )
        self.one_arm_fixed_sim.append(self.body_rod)

        self.one_arm_fixed_sim.constrain(self.body_rod).using(
            GeneralConstraint,
            # constrained_position_idx=(0,),
            constrained_director_idx=tuple(range(self.body_rod.n_elems)),
            # translational_constraint_selector = np.array([True, True, True]),
            rotational_constraint_selector=np.array([True, True, True]),
        )

        self.one_arm_fixed_sim.add_forcing_to(self.body_rod).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, self.gravitational_acc])
        )

        """ Connect to head """
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.connect(
                self.shearable_rods[i_arm],
                self.body_rod,
                first_connect_idx=0,
                second_connect_idx=0,
            ).using(FreeJoint, k=self.E / 100, nu=0)

        self.sucker_control = np.full((self.n_arm, 1), False)
        self.sucker_fixed_position = np.zeros(
            (
                self.n_arm,
                3,
            )
        )
        self.sucker_fixed_director = np.zeros(
            (
                self.n_arm,
                3,
                3,
            )
        )
        # for i_arm in range(self.n_arm):
        #     self.one_arm_fixed_sim.connect(self.shearable_rods[i_arm], self.cylinder).using(
        #         ExternalContact_sucker, k=0.0025, nu=0.1, sucker_flag=self.sucker_control[i_arm, :]
        #     )
        # for i_arm in range(self.n_arm):
        #     self.one_arm_fixed_sim.constrain(self.shearable_rods[i_arm]).using(
        #         SuckerFixedBC, constrained_position_idx=(-1,), constrained_director_idx=(-1,),
        #         sucker=self.sucker_control[i_arm, :],
        #         n_p=self.sucker_fixed_position[i_arm, :],
        #         n_d=self.sucker_fixed_director[i_arm, ...],
        #     )

        """ Add damping """
        for i_arm in range(self.n_arm):
            self.add_damping(self.shearable_rods[i_arm])
        self.add_damping(self.body_rod)

        """  Add callbacks """
        if self.callback:
            self.make_callbacks()
            self.callback = False

        """  Add fluid drag forces """
        # for i_arm in range(self.n_arm):
        #     self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
        #         DragForce,
        #         rho_environment=self.rho_water,
        #         c_per=self.c_per,
        #         c_tan=self.c_tan,
        #         system=self.shearable_rods[i_arm],
        #         step_skip=self.step_skip,
        #         callback_params=self.pp_lists[i_arm],  # self.rod_parameters_dict
        # )
        # self.one_arm_fixed_sim.add_forcing_to(self.body_rod).using(
        #     DragForce,
        #     rho_environment=self.rho_water,
        #     c_per=self.c_per,
        #     c_tan=self.c_tan,
        #     system=self.body_rod,
        #     step_skip=self.step_skip,
        #     callback_params=self.head_list,  # self.rod_parameters_dict
        # )

        # ''' Create a continuous muscle model '''
        # muscle = ContinuousActuation(self.n_elem, self.muscle_fiber)
        for i_arm in range(self.n_arm):
            self.muscle_groups.append(
                self.set_muscles(self.base_radius, self.shearable_rods[i_arm])
            )
            self.muscle_callback_params_list.append(
                [defaultdict(list) for _ in self.muscle_groups]
            )
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                ApplyMuscleGroups,
                muscle_groups=self.muscle_groups[i_arm],
                step_skip=self.step_skip,
                callback_params_list=self.muscle_callback_params_list[i_arm],
            )
        #     self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
        #         SensoryMuscleCtrl,
        #         self.u[i_arm],
        #         muscle,
        #         # target,
        #         # shoot_angle,
        #         # shear_matrix,
        #         # bend_matrix,
        #         ramp_up_time=0.5,
        #     )

        """Add arm mesh surface contact"""
        for i_arm in range(self.n_arm):
            self.set_mesh_surface_arm_contact(self.shearable_rods[i_arm])

        """  Finalize """
        self.one_arm_fixed_sim.finalize()

        for i_arm in range(self.n_arm):
            self.compute_strains(self.shearable_rods[i_arm])
        self.compute_strains(self.body_rod)

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.timestepper, self.one_arm_fixed_sim
        )

        # if self.state_type == 'pos':
        #     self.pos[...] = [rod.position_collection[:2, 1:].flatten() for rod in self.shearable_rods]
        #     self.prev_pos[...] = self.pos
        # else:
        #     self.kappa[...] = [rod.kappa[0] for rod in self.shearable_rods]
        #     self.prev_kappa[...] = self.kappa
        return self.total_steps, self.one_arm_fixed_sim

    def step(self, time, muscle_activations):

        """Set muscle activations"""
        for i_arm in range(self.n_arm):
            for muscle_group, activation in zip(
                self.muscle_groups[i_arm], muscle_activations[i_arm]
            ):
                muscle_group.apply_activation(activation)

        """ Run the simulation for one step """
        time = self.do_step(
            self.timestepper,
            self.stages_and_updates,
            self.one_arm_fixed_sim,
            time,
            self.arm_dt,
        )

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        # invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        # if invalid_values_condition == True:
        #     print("NaN detected in the simulation !!!!!!!!")
        #     done = True

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.one_arm_fixed_sim, done

    def make_callbacks(self):
        self.pp_lists = []
        for i_arm in range(self.n_arm):
            pp_list = defaultdict(list)
            self.pp_lists.append(pp_list)
            self.one_arm_fixed_sim.collect_diagnostics(
                self.shearable_rods[i_arm]
            ).using(
                ArmCallBack, step_skip=self.step_skip, callback_params=self.pp_lists[-1]
            )

        self.head_list = defaultdict(list)
        self.pp_lists.append(self.head_list)
        self.one_arm_fixed_sim.collect_diagnostics(self.body_rod).using(
            HeadCallBack, step_skip=self.step_skip, callback_params=self.head_list
        )

    def cal_mean_pos_arm_state(self):
        for i_arm in range(self.n_arm):
            self.pos[i_arm] = (
                self.shearable_rods[i_arm].position_collection[:2, 1:].flatten()
            )
        vel = self.pos - self.prev_pos

        mean_pos = np.mean(
            np.reshape(self.pos, (self.n_arm, -1, self.node_skip)), axis=2
        )
        mean_vel = np.mean(np.reshape(vel, (self.n_arm, -1, self.node_skip)), axis=2)

        # self.arm_states[...] = np.concatenate([mean_pos / 0.2, mean_vel / 0.04], axis=-1)
        # self.prev_pos[...] = self.pos

    def cal_pos_arm_state(self):
        for i_arm in range(self.n_arm):
            self.pos[i_arm] = (
                self.shearable_rods[i_arm].position_collection[:2, 1:].flatten()
            )
        vel = self.pos - self.prev_pos

        # self.arm_states[...] = np.concatenate([self.pos / 0.2, vel / 0.04], axis=-1)
        # self.prev_pos[...] = self.pos

    def cal_all_arm_state(self):
        for i_arm in range(self.n_arm):
            self.kappa[i_arm] = self.shearable_rods[i_arm].kappa[0]
        kappa_rate = self.kappa - self.prev_kappa

        # self.arm_states[...] = self.normalize_state(self.kappa, kappa_rate)
        # self.prev_kappa[...] = self.kappa

    def cal_mean_arm_state(self):
        for i_arm in range(self.n_arm):
            self.kappa[i_arm] = self.shearable_rods[i_arm].kappa[0]
        kappa_rate = self.kappa - self.prev_kappa

        self.mean_kappa = np.mean(
            self.kappa.reshape((self.n_arm, -1, self.node_skip)), axis=2
        )
        self.mean_kappa_rate = np.mean(
            kappa_rate.reshape((self.n_arm, -1, self.node_skip)), axis=2
        )

        # self.arm_states[...] = self.normalize_state(self.mean_kappa, self.mean_kappa_rate)
        # self.prev_kappa[...] = self.kappa

    def normalize_state(self, kappa, kappa_rate):
        k = do_normalization(kappa, self.kappa_range)
        kr = do_normalization(kappa_rate, self.kappa_rate_range)
        return np.concatenate([k, kr], axis=-1)

    def save_data(self, filename="simulation", **kwargs):

        import pickle

        print("Saving data to pickle files ...", end="\r")

        with open(filename + "_data.pickle", "wb") as data_file:
            data = dict(
                recording_fps=self.recording_fps,
                systems=self.pp_lists,
                # muscle_groups=self.muscle_callback_params_list,
                **kwargs,
            )
            pickle.dump(data, data_file)

        with open(filename + "_systems.pickle", "wb") as system_file:
            data = dict(
                systems=self.one_arm_fixed_sim,
                muscle_groups=self.muscle_groups,
            )
            pickle.dump(data, system_file)

        print("Saving data to pickle files ... Done!")

    def plot_data(self, filename="simulation.mp4", **kwargs):
        plot_video_with_surface(
            self.pp_lists,
            filename,
            fps=self.recording_fps,
            step=1,
            x_limits=(-0.25, 0.25),
            y_limits=(-0.25, 0.25),
            z_limits=(-0.25, 0.25),
        )
        print("Plotting ... Done!")
