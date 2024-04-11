import numpy as np
from octopus_arm import ArmEnvironment
from CustomFrictionSurface import RodMeshSurfaceContactWithGrid
from tqdm import tqdm
from elastica import *
import pickle


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()
        self.set_mesh_surface_contact()

    def set_mesh_surface_contact(self):
        """Set up a mesh surface object"""
        mesh = Mesh(filepath=r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj")
        mesh.translate(-np.array(mesh.mesh_center))
        radius_base = 0.012
        mesh.translate(
            -4 * radius_base - np.array([0, 0, np.min(mesh.face_centers[2])])
        )
        # mesh.visualize()
        print(np.min(mesh.face_centers[2]))
        self.mesh_surface = MeshSurface(mesh)
        print(np.min(self.mesh_surface.face_centers[2]))
        self.simulator.append(self.mesh_surface)

        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        """ Set up contact between arm and mesh surface """
        gravitational_acc = -9.80665
        self.simulator.add_forcing_to(self.shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
        )
        mu = 0.4
        kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
        filename = "m32_Viekoda_Bay/grid_m32_Viekoda_Bay.dat"
        with open(filename, "rb") as fptr:
            faces_grid = pickle.load(fptr)
        static_mu_array = np.zeros(kinetic_mu_array.shape)
        self.simulator.detect_contact_between(
            self.shearable_rod, self.mesh_surface
        ).using(
            RodMeshSurfaceContactWithGrid,
            k=1e1,
            nu=1e-1,
            slip_velocity_tol=1e-8,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
            faces_grid=faces_grid,
            gamma=0.1,
        )


def main(filename, target_position=None):

    """Create simulation environment"""
    final_time = 15.0
    env = Environment(final_time)
    total_steps, systems = env.reset()

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)
    np.random.seed(0)
    for m in [0, 2, 5]:
        activations[m] = np.random.rand(*env.muscle_groups[m].activation.shape)
    activations[4] = np.linspace(1, 0, *env.muscle_groups[4].activation.shape)

    for k_sim in tqdm(range(total_steps)):
        time, systems, done = env.step(time, activations)
        if np.linalg.norm(env.shearable_rod.velocity_collection[:, -1]) < 1e-5 or done:
            break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
    )

    env.plot_data(filename=filename + ".mp4")


if __name__ == "__main__":
    main(filename="with_mesh")
