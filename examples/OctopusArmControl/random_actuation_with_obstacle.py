import numpy as np
from elastica import *
from collections import defaultdict
from octopus_arm import ArmEnvironment
from callback_func import *
from tqdm import tqdm


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict, self.sphere_parameters_dict]

    def setup(self):
        self.set_arm()
        self.set_target()

    def set_target(self):
        """Set up a sphere object"""
        target_radius = 0.02
        self.sphere = Sphere(
            center=np.array([0.15, -0.02, 0.05]),
            base_radius=target_radius,
            density=1000,
        )
        self.sphere.director_collection[:, :, 0] = np.array(
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        )
        self.simulator.append(self.sphere)
        self.sphere_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.sphere).using(
            SphereCallBack,
            step_skip=self.step_skip,
            callback_params=self.sphere_parameters_dict,
        )

        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        self.simulator.constrain(self.sphere).using(
            FixedConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        """ Set up contact between arm and sphere """
        self.simulator.detect_contact_between(self.shearable_rod, self.sphere).using(
            RodSphereContact, k=5e0, nu=1e-1
        )


def main(filename, target_position=None):

    """Create simulation environment"""
    final_time = 20.0
    env = Environment(final_time)
    total_steps, systems = env.reset()

    if not (target_position is None):
        env.sphere.position_collection[:, 0] = target_position

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)
    np.random.seed(0)
    for m in [0, 5]:
        activations[m] = np.random.rand(*env.muscle_groups[m].activation.shape)
    activations[4] = np.linspace(1, 0, *env.muscle_groups[4].activation.shape)

    for k_sim in tqdm(range(total_steps)):
        time, systems, done = env.step(time, activations)
        if np.linalg.norm(env.shearable_rod.velocity_collection[:, -1]) < 1e-4 or done:
            break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
    )

    env.plot_data(filename=filename + ".mp4")


if __name__ == "__main__":
    main(filename="test")
