import numpy as np
from elastica import *
from examples.ArtificialMusclesCases import *
from single_muscle_simulator import muscle_stretching_actuation_simulatior


# how the muscle is constrained
constrain_settings_dict = {
    "isometric_test": False,
    "isobaric_test": True,
    "constrain_start_time": 0.0,
}

# what external forcing it is experiencing
forcing_settings_dict = {
    "force_mag": 0.0,
    "desired_muscle_strain": 0.0,
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
    "self_contact": False,
    "contact_radius_ratio": 1 / 10,
    "k_val": 1e1,
    "nu": 0.0,
    "k_repulsive_val": 1e2,
    "separation_distance": 1e-3,
}

# fiber-fiber interaction settings
fiber_connection_settings_dict = {
    "connection_range": 0,
    "k_val": 0e0,
    "nu": 0.0,
    "k_repulsive_val": 1e1,
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
    "sim_name": "PureContraction",
    "final_time": 1,  # seconds
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

# input_muscle = mock_hypercoil()
input_muscle = Samuel_supercoil_stl()


current_path = os.getcwd()
sim_settings.save_folder = os.path.join(
    current_path, sim_settings.sim_name + "/" + input_muscle.name + "/data"
)

muscle_stretching_actuation_simulatior(
    input_muscle=input_muscle,  # the type of muscle used
    constrain_settings=constrain_settings,  # how the muscle is constrained
    forcing_settings=forcing_settings,  # what external forcing it is experiencing
    actuation_settings=actuation_settings,  # how is it actuated
    self_contact_settings=self_contact_settings,  # self contact settings
    fiber_connection_settings=fiber_connection_settings,  # fiber-fiber interaction settings
    sim_settings=sim_settings,
)
