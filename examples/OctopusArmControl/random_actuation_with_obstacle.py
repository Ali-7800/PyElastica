import copy

import numpy as np
from elastica import *
from octopus_arm import ArmEnvironment
from callback_func import *
from tqdm import tqdm
from elastica.contact_utils import _norm, _find_min_dist


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
            center=np.array([0.15, -0.02, 0.02]),
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


def calculate_contact_index(
    x_collection_rod,
    edge_collection_rod,
    x_cylinder,
    edge_cylinder,
    radii_sum,
    length_sum,
):
    index_list = []
    # We already pass in only the first n_elem x
    n_points = x_collection_rod.shape[1]
    for i in range(n_points - 1):
        # Element-wise bounding box
        x_selected = x_collection_rod[..., i]
        # x_cylinder is already a (,) array from outised
        del_x = x_selected - x_cylinder
        norm_del_x = _norm(del_x)

        # If outside then don't process
        if norm_del_x >= (radii_sum[i] + length_sum[i]):
            continue

        # find the shortest line segment between the two centerline
        # segments : differs from normal cylinder-cylinder intersection
        distance_vector, _, _ = _find_min_dist(
            x_selected, edge_collection_rod[..., i], x_cylinder, edge_cylinder
        )
        distance_vector_length = _norm(distance_vector)
        distance_vector /= distance_vector_length

        gamma = radii_sum[i] - distance_vector_length

        # If distance is large, don't worry about it
        if gamma < -1e-5:
            continue
        index_list.append(i)
    return index_list


def find_contact_index(rod, obstacle):
    x_cyl = (
        obstacle.position_collection[..., 0]
        - 0.5 * obstacle.length * obstacle.director_collection[2, :, 0]
    )
    index_list = calculate_contact_index(
        rod.position_collection[..., :-1],
        rod.lengths * rod.tangents,
        x_cyl,
        obstacle.length * obstacle.director_collection[2, :, 0],
        rod.radius + obstacle.radius,
        rod.lengths + obstacle.length,
    )
    return index_list


def get_slip_out_actuation(activations, index_list, relax_offset=5):
    slip_out_activations = [np.zeros(activation.shape) for activation in activations]
    contracting_index_range = range(index_list[0] - relax_offset)

    # determine dominating LM
    LM_start_index = 1
    LMs_magnitude = [
        np.linalg.norm(activation[contracting_index_range])
        for activation in activations[LM_start_index : LM_start_index + 4]
    ]
    contract_LM = np.argmax(LMs_magnitude)
    slip_out_activations[contract_LM + LM_start_index][contracting_index_range] = 1.0

    # determine dominating OM
    OM_start_index = 5
    OMs_magnitude = [
        np.linalg.norm(activation[contracting_index_range])
        for activation in activations[OM_start_index:]
    ]
    contract_OM = np.argmax(OMs_magnitude)
    slip_out_activations[contract_OM + OM_start_index][contracting_index_range] = 0.1

    return slip_out_activations


def main(filename, target_position=None):
    use_local_feedback = False
    mask_flag = False
    g_sim = 0

    """ Create simulation environment """
    final_time = 15.0
    env = Environment(final_time)
    total_steps, systems = env.reset()

    if not (target_position is None):
        env.sphere.position_collection[:, 0] = target_position

    """ Read arm params -> index 0: TM, index 1~4: LM, index 5: OM+, index 6: OM- """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))
    prev_activations = copy.deepcopy(activations)

    """ Set activations """
    np.random.seed(0)
    # TM
    activations[0] = np.random.rand(*env.muscle_groups[0].activation.shape)
    # LM
    activations[4] = np.linspace(1, 0, *env.muscle_groups[4].activation.shape)
    activations[4][: len(activations[4]) // 2] = (
        0.5 * np.random.rand(len(activations[4]) // 2) + 0.5
    )
    # OM (4 muscles in one set)
    activations[5] = 0.2 * np.linspace(1, 0, *env.muscle_groups[5].activation.shape)

    apply_activations = copy.deepcopy(activations)

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    for k_sim in tqdm(range(total_steps)):
        if use_local_feedback:
            contact_index_list = find_contact_index(env.shearable_rod, env.sphere)
            if len(contact_index_list) > 0:
                if not mask_flag:
                    mask_flag = True
                    g_sim = 0

                current_activations = get_slip_out_actuation(
                    activations, contact_index_list
                )
            else:
                if mask_flag:
                    mask_flag = False
                    prev_activations = apply_activations
                    g_sim = 0
                current_activations = activations

            apply_factor = np.min([1.0, g_sim / (0.01 / env.time_step)])
            for i in range(len(activations)):
                apply_activations[i] = current_activations[
                    i
                ] * apply_factor + prev_activations[i] * (1 - apply_factor)
            g_sim += 1
            prev_activations = apply_activations

        time, systems, done = env.step(time, apply_activations)

        if k_sim > 10000 and (
            np.linalg.norm(env.shearable_rod.velocity_collection[:, -1]) < 1e-4 or done
        ):
            break

    """ Save the data of the simulation """
    env.save_data(
        filename=filename,
    )

    env.plot_data(filename=filename + ".mp4")


if __name__ == "__main__":
    main(filename="test")
