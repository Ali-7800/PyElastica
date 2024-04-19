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
    "final_time": 30,
    "arm_dt": 2.0e-4,
    "step_skip": 250,  # int(1.0 / (self.fps * self.arm_dt))
    "arm_c_per": 1.013,
    "arm_c_tan": 0.0256,
    "target_radius": 0.02,
    "env_idx": 0,  # 0 for flat ground, 1 for m32_Viekoda_Bay, 2 for mars-landscape
}

task = {}


def extension_activation(env, time, start_time, i_arm):
    suction_index = -30
    if time - start_time < 1:
        env.sucker_fixed_position[i_arm, ...] = env.shearable_rods[
            i_arm
        ].position_collection[..., suction_index]
        env.sucker_fixed_director[i_arm, ...] = env.shearable_rods[
            i_arm
        ].director_collection[..., suction_index]
        activations_octopus = []
        for j_arm in range(env.n_arm):
            activations = []
            current_muscle_groups = env.muscle_groups[j_arm]
            for m in range(len(current_muscle_groups)):
                activations.append(np.zeros(current_muscle_groups[m].activation.shape))
            if j_arm == i_arm:
                for n in [0]:
                    activations[n] = min(time - start_time, 1) * np.linspace(
                        1, 0, *current_muscle_groups[n].activation.shape
                    )
                # activations[4] = np.linspace(1, 0, *current_muscle_groups[4].activation.shape)
            activations_octopus.append(activations)
    else:
        activations_octopus = []
        env.sucker_control[i_arm] = True
        for j_arm in range(env.n_arm):
            activations = []
            current_muscle_groups = env.muscle_groups[j_arm]
            for m in range(len(current_muscle_groups)):
                activations.append(np.zeros(current_muscle_groups[m].activation.shape))
            if j_arm == i_arm:
                for n in [0]:
                    activations[n] = max(
                        (3.0 - (time - start_time)) / 2, 0
                    ) * np.linspace(1, 0, *current_muscle_groups[n].activation.shape)
                # activations[4] = np.linspace(1, 0, *current_muscle_groups[4].activation.shape)
            activations_octopus.append(activations)
    return activations_octopus


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
        activations_octopus.append(activations)

    i_arm = 3

    for k_sim in tqdm(range(total_steps)):
        time, systems, done = env.step(time, activations_octopus)
        step_start_time = (time // 5) * 5
        if time - step_start_time < 3.0:
            activations_octopus = extension_activation(
                env, time, step_start_time, i_arm
            )
        elif time - step_start_time < 5.0:
            env.sucker_control[i_arm] = False

        # if np.linalg.norm(env.shearable_rod.velocity_collection[:, -1]) < 1e-4 or done:
        #     break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
    )

    env.plot_data(filename=filename + ".mp4")


if __name__ == "__main__":
    main(filename="octopus_with_mesh")
