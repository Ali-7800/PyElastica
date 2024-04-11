import numpy as np
from arm_simulator_with_mesh import ArmSimulator
from tqdm import tqdm
from elastica import *
import pickle


env_config = {
    "env_name": "locomotion",
    "num_envs": 3,
    "num_arms": 4,
    "base_length": 0.2,
    "arm_state_dim": 11,
    "n_elem": 100,
    "node_skip": 9,
    "n_muscle": 2,
    "final_time": 10,
    "arm_dt": 2.0e-4,
    "step_skip": 250,  # int(1.0 / (self.fps * self.arm_dt))
    "arm_c_per": 1.013,
    "arm_c_tan": 0.0256,
    "target_radius": 0.02,  # 0.02 #0.1
    "mesh_path": r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj",
    "mesh_grid_path": "m32_Viekoda_Bay/grid_m32_Viekoda_Bay.dat",
}

task = {}


def main(filename):

    """Create simulation environment"""
    env = ArmSimulator(env_config=env_config, task=task, callback=True)
    total_steps, systems = env.reset()

    """ Read arm params """

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)
    np.random.seed(0)
    activations_octopus = []
    for i_arm in range(env.n_arm):
        activations = []
        current_muscle_groups = env.muscle_groups[i_arm]
        for m in range(len(current_muscle_groups)):
            activations.append(np.zeros(current_muscle_groups[m].activation.shape))
        for m in [5]:
            activations[m] = 0.1 * np.random.rand(
                *current_muscle_groups[m].activation.shape
            )
        activations[4] = np.linspace(1, 0, *current_muscle_groups[4].activation.shape)
        activations_octopus.append(activations)

    for k_sim in tqdm(range(total_steps)):
        time, systems, done = env.step(time, activations_octopus)
        # if np.linalg.norm(env.shearable_rod.velocity_collection[:, -1]) < 1e-4 or done:
        #     break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
    )

    env.plot_data(filename=filename + ".mp4")


if __name__ == "__main__":
    main(filename="octopus_with_mesh")
