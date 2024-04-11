import copy

import numpy as np
from elastica import (
    BaseSystemCollection,
    Constraints,
    Connections,
    Forcing,
    CallBacks,
    Damping,
    GeneralConstraint,
    GravityForces,
    AnisotropicFrictionalPlane,
    FreeJoint,
)
from dissipation import ExponentialDamper, FilterDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper import extend_stepper_interface
from elastica.rod.cosserat_rod import (
    CosseratRod,
    _compute_shear_stretch_strains,
    _compute_bending_twist_strains,
)
from elastica.boundary_conditions import OneEndFixedBC
from elastica.rigidbody import Cylinder
from elastica.joint import ExternalContact

# from contact import ExternalContact
from obstacle_ball_contact import ExternalContactCylinderCylinder
from force_function import RigidBodyDragForce, DragForce, ExternalContact_sucker
from control_function import SensoryMuscleCtrl
from elastica.timestepper import PositionVerlet
from actuation_muscles import ContinuousActuation
from collections import defaultdict

# from sopht.utils.precision import get_real_t
# import sopht_simulator as sps
import matplotlib.pyplot as plt
from locomotion_strategy import SuckerFixedBC


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
        return


class OneArmSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    pass


class ArmSimulator:
    def __init__(self, env_config, task, callback):
        """General setup"""
        self.n_arm = env_config["num_arms"]
        self.callback = callback
        self.timestepper = PositionVerlet()

        """ Task """
        self.reset = (
            self.locomotion_reset
            if "locomotion" in env_config["env_name"]
            else self.manipulation_reset
        )
        self.sucker = task.sucker
        self.obstacle = task.obstacle
        self.n_elem = env_config["n_elem"]

        self.arm_states = np.zeros((self.n_arm, task.arm_state_dim))
        self.state_type = task.state_type
        if task.state_type == "pos":
            self.pos = np.zeros((self.n_arm, self.n_elem * 2))
            self.prev_pos = np.zeros((self.n_arm, self.n_elem * 2))
            if task.arm_state_dim < 400:
                self.cal_arm_state = self.cal_mean_pos_arm_state
            else:
                self.cal_arm_state = self.cal_pos_arm_state
        else:
            self.kappa = np.zeros((self.n_arm, self.n_elem - 1))
            self.prev_kappa = np.zeros((self.n_arm, self.n_elem - 1))
            if task.arm_state_dim < 198:
                self.cal_arm_state = self.cal_mean_arm_state
            else:
                self.cal_arm_state = self.cal_all_arm_state
        self.target_radius = env_config["target_radius"]
        self.obstacle_radius = env_config["target_radius"] * 0.2

        """ Rest lengths and normalization parameters """
        arm_info = np.load(env_config["arm_info"])
        self.kappa_range = [-140, 140]  # arm_info['kappa_range']#
        self.kappa_rate_range = [-35, 35]  # arm_info['kappa_rate_range']#
        self.rest_lengths = arm_info["rest_lengths"]
        self.rest_voronoi_lengths = arm_info["rest_voronoi_lengths"]

        """ Arm """
        self.base_length = env_config["base_length"]
        self.n_elem = env_config["n_elem"]
        self.node_skip = env_config["node_skip"]
        self.step_skip = env_config["step_skip"]
        self.arm_dt = env_config["arm_dt"]
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
        damping = 0.01
        damping_torque = 0.01
        self.dissipation = damping * ((self.base_radius / radius_base) ** 2)
        self.nu_torque = damping_torque * ((self.base_radius / radius_base) ** 4)
        self.E = 1e4
        self.poisson_ratio = 1

        """ Muscles """
        n_max = np.array([1, 1, -2])
        self.n_muscle = len(n_max)  # number of muscle fiber
        max_force = (self.base_radius / radius_base) ** 2 * n_max[:, None]
        passive_ratio = 1
        radius_ratio = np.array([1, -1, 0])

        self.muscle_fiber = ParameterDictionary(
            n_muscle=self.n_muscle,
            n_max=n_max,
            max_force=max_force,
            passive_ratio=passive_ratio,
            radius_ratio=radius_ratio,
        )

        """ Flow """
        rho_water = 1022
        c_per = 0.5 * rho_water * env_config["arm_c_per"]
        c_tan = 0.5 * rho_water * env_config["arm_c_tan"]
        self.fluid_param = ParameterDictionary(
            rho_water=rho_water, c_per=c_per, c_tan=c_tan
        )
        self.cylinder_drag_coeff = env_config["cylinder_drag_coeff"]

    def add_damping(self, rod):
        damping_constant = 0.01 / 1.5
        self.one_arm_fixed_sim.dampen(rod).using(
            ExponentialDamper,
            damping_constant=damping_constant,
            time_step=self.arm_dt,
        )
        self.one_arm_fixed_sim.dampen(rod).using(
            FilterDamper,
            filter_order=4,
        )

    def manipulation_reset(
        self, rod_start_list, ball_start, ball_vel=[0.0, 0.0], obstacle_start=None
    ):
        """Reset parameters"""
        self.u = np.zeros([self.n_arm, self.n_muscle, self.n_elem + 1])
        self.arm_states[...] = 0.0
        self.head_list = None

        """ Set up arm simulator """
        self.one_arm_fixed_sim = OneArmSimulator()
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

            rod.rest_lengths[...] = self.rest_lengths
            rod.rest_voronoi_lengths[...] = self.rest_voronoi_lengths
            rod.velocity_collection[...] = rod_start["vel"]
            rod.omega_collection[...] = rod_start["omega"]
            self.shearable_rods.append(rod)
            self.one_arm_fixed_sim.append(self.shearable_rods[-1])
        """ Add boundary conditions """
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.constrain(self.shearable_rods[i_arm]).using(
                OneEndFixedBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )
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

            self.one_arm_fixed_sim.add_forcing_to(self.cylinder).using(
                RigidBodyDragForce,
                self.fluid_param,
                self.cylinder.n_elems,
                self.cylinder_drag_coeff,
            )

            if self.sucker:
                self.sucker_control = np.full((self.n_arm, 1), False)
                for i_arm in range(self.n_arm):
                    self.one_arm_fixed_sim.connect(
                        self.shearable_rods[i_arm], self.cylinder
                    ).using(
                        ExternalContact_sucker,
                        k=0.0025 / 2,
                        nu=0.0,
                        sucker_flag=self.sucker_control[i_arm, :],
                    )

            self.arm_ball_strike = np.full((self.n_arm, 1), False)
            for i_arm in range(self.n_arm):
                self.one_arm_fixed_sim.connect(
                    self.shearable_rods[i_arm], self.cylinder
                ).using(
                    ExternalContact,
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

            self.one_arm_fixed_sim.constrain(self.cylinder_obs).using(
                OneEndFixedBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        """  Add fluid drag forces """
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                DragForce,
                self.fluid_param,
                self.n_elem,
            )
        """ Create a continuous muscle model """
        muscle = ContinuousActuation(self.n_elem, self.muscle_fiber)
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                SensoryMuscleCtrl,
                self.u[i_arm],
                muscle,
                # target,
                # shoot_angle,
                # shear_matrix,
                # bend_matrix,
                ramp_up_time=0.5,
            )

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

        """  Finalize """
        self.one_arm_fixed_sim.finalize()
        for i_arm in range(self.n_arm):
            self.compute_strains(self.shearable_rods[i_arm])

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.timestepper, self.one_arm_fixed_sim
        )
        if self.state_type == "pos":
            self.pos[...] = [
                rod.position_collection[:2, 1:].flatten() for rod in self.shearable_rods
            ]
            self.prev_pos[...] = rod_start["prev_pos"]
        else:
            self.kappa[...] = [rod.kappa[0] for rod in self.shearable_rods]
            self.prev_kappa[...] = rod_start["prev_kappa"]

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

            rod.rest_lengths[...] = self.rest_lengths
            rod.rest_voronoi_lengths[...] = self.rest_voronoi_lengths

            self.shearable_rods.append(rod)
            self.one_arm_fixed_sim.append(self.shearable_rods[-1])

    def locomotion_reset(
        self, rod_start_list, ball_start, ball_vel=[0.0, 0.0], obstacle_start=None
    ):
        """Reset parameters"""
        self.u = np.zeros([self.n_arm, self.n_muscle, self.n_elem + 1])
        self.arm_states[...] = 0.0

        """ Set up arm simulator """
        self.one_arm_fixed_sim = OneArmSimulator()
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
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.constrain(self.shearable_rods[i_arm]).using(
                SuckerFixedBC,
                constrained_position_idx=(-1,),
                constrained_director_idx=(-1,),
                sucker=self.sucker_control[i_arm, :],
                n_p=self.sucker_fixed_position[i_arm, :],
                n_d=self.sucker_fixed_director[i_arm, ...],
            )

        """ Add damping """
        damping_constant = 0.01 / 1.5
        for i_arm in range(self.n_arm):
            self.add_damping(self.shearable_rods[i_arm])
        self.add_damping(self.body_rod)

        """  Add fluid drag forces """
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                DragForce,
                self.fluid_param,
                self.n_elem,
            )
        self.one_arm_fixed_sim.add_forcing_to(self.body_rod).using(
            DragForce,
            self.fluid_param,
            self.n_elem,
        )

        """ Create a continuous muscle model """
        muscle = ContinuousActuation(self.n_elem, self.muscle_fiber)
        for i_arm in range(self.n_arm):
            self.one_arm_fixed_sim.add_forcing_to(self.shearable_rods[i_arm]).using(
                SensoryMuscleCtrl,
                self.u[i_arm],
                muscle,
                # target,
                # shoot_angle,
                # shear_matrix,
                # bend_matrix,
                ramp_up_time=0.5,
            )

        """  Add callbacks """
        if self.callback:
            self.make_callbacks()
            self.callback = False

        """  Finalize """
        self.one_arm_fixed_sim.finalize()

        for i_arm in range(self.n_arm):
            self.compute_strains(self.shearable_rods[i_arm])
        self.compute_strains(self.body_rod)

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.timestepper, self.one_arm_fixed_sim
        )

        if self.state_type == "pos":
            self.pos[...] = [
                rod.position_collection[:2, 1:].flatten() for rod in self.shearable_rods
            ]
            self.prev_pos[...] = self.pos
        else:
            self.kappa[...] = [rod.kappa[0] for rod in self.shearable_rods]
            self.prev_kappa[...] = self.kappa

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

        self.arm_states[...] = np.concatenate(
            [mean_pos / 0.2, mean_vel / 0.04], axis=-1
        )
        self.prev_pos[...] = self.pos

    def cal_pos_arm_state(self):
        for i_arm in range(self.n_arm):
            self.pos[i_arm] = (
                self.shearable_rods[i_arm].position_collection[:2, 1:].flatten()
            )
        vel = self.pos - self.prev_pos

        self.arm_states[...] = np.concatenate([self.pos / 0.2, vel / 0.04], axis=-1)
        self.prev_pos[...] = self.pos

    def cal_all_arm_state(self):
        for i_arm in range(self.n_arm):
            self.kappa[i_arm] = self.shearable_rods[i_arm].kappa[0]
        kappa_rate = self.kappa - self.prev_kappa

        self.arm_states[...] = self.normalize_state(self.kappa, kappa_rate)
        self.prev_kappa[...] = self.kappa

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

        self.arm_states[...] = self.normalize_state(
            self.mean_kappa, self.mean_kappa_rate
        )
        self.prev_kappa[...] = self.kappa

    def normalize_state(self, kappa, kappa_rate):
        k = do_normalization(kappa, self.kappa_range)
        kr = do_normalization(kappa_rate, self.kappa_rate_range)
        return np.concatenate([k, kr], axis=-1)
