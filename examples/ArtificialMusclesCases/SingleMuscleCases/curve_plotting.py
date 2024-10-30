import numpy as np

# import os
# from elastica import *
from examples.ArtificialMusclesCases import *
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import sympy as sp


from os import listdir
from os.path import isfile, join
from os import getcwd

test_muscle = Samuel_supercoil_stl(experimental_data=True)

no_connection_path = "/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PassiveForceTest/Samuel_supercoil_stl/data/passive_no_connection"
# no_connection_path = '/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PassiveForceTest/Samuel_supercoil_stl_single_fiber/data/single_fiber'
# no_connection_path = "/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PassiveForceTest/Samuel_supercoil_stl/data/high_strain_no_connection"
no_connection_files = [
    f for f in listdir(no_connection_path) if isfile(join(no_connection_path, f))
]

# with_spring_path = '/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PassiveForceTest/Samuel_supercoil_stl/data/passive_with_spring'
with_spring_path = "/Users/ali-7800/Desktop/Research/dev_artificial_muscle/PyElastica/examples/ArtificialMusclesCases/SingleMuscleCases/PassiveForceTest/Samuel_supercoil_stl/data/high_strain_connection"
with_spring_files = [
    f for f in listdir(with_spring_path) if isfile(join(with_spring_path, f))
]

n_muscles = 3
strain_no_connection = np.zeros(len(no_connection_files) + 1)
force_no_connection = np.zeros(len(no_connection_files) + 1)

plt.plot(
    test_muscle.experimental_tensile_test_single_fiber_times_3[:, 0],
    test_muscle.experimental_tensile_test_single_fiber_times_3[:, 1],
    linewidth=3,
    label="Single Fiber Passive Force ×3 (Exp)",
    color="black",
)

plt.plot(
    test_muscle.experimental_tensile_test[:, 0],
    test_muscle.experimental_tensile_test[:, 1],
    linewidth=3,
    label="Passive Force (Exp)",
    color="green",
)

# plt.plot(
#     test_muscle.experimental_tensile_test_single_fiber_times_3[:, 0], test_muscle.experimental_tensile_test_single_fiber_times_3[:, 1], linewidth=2, label="Single Fiber Passive Force (Exp)", color = "black",

# )


for datafile_name, i in zip(no_connection_files, range(len(no_connection_files))):
    force_no_connection[i] = float(datafile_name[27:43])
    data = np.load(no_connection_path + "/" + datafile_name)

    muscle_positions = np.array(data["muscle_rods_position_history"])
    centerline_position = np.zeros(muscle_positions.shape[1:])
    for j in range(muscle_positions.shape[0]):
        centerline_position += muscle_positions[j, :, :, :] / n_muscles

    # n = centerline_position.shape[-1]
    strain_no_connection[i] = np.dot(
        np.array([0.0, 0.0, 1.0]),
        (
            (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
            - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
        )
        / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
    )


no_connection_idx = np.argsort(strain_no_connection)
plt.plot(
    strain_no_connection[no_connection_idx],
    force_no_connection[no_connection_idx],
    linewidth=2,
    # linestyle = "--",
    marker="o",
    markersize=5,
    color="tab:red",
    label="Passive force (Sim, no adhesion)",
)
# plt.suptitle("Coil Strain vs Force")


strain_with_spring = np.zeros(len(with_spring_files) + 1)
force_with_spring = np.zeros(len(with_spring_files) + 1)
for datafile_name, i in zip(with_spring_files, range(len(with_spring_files))):
    if "npz" not in datafile_name:
        continue
    force_with_spring[i] = float(datafile_name[27:43])
    data = np.load(with_spring_path + "/" + datafile_name)

    muscle_positions = np.array(data["muscle_rods_position_history"])
    centerline_position = np.zeros(muscle_positions.shape[1:])
    for j in range(muscle_positions.shape[0]):
        centerline_position += muscle_positions[j, :, :, :] / n_muscles

    # n = centerline_position.shape[-1]
    strain_with_spring[i] = np.dot(
        np.array([0.0, 0.0, 1.0]),
        (
            (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
            - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
        )
        / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
    )


with_spring_idx = np.argsort(strain_with_spring)
plt.plot(
    strain_with_spring[with_spring_idx][:-2],
    force_with_spring[with_spring_idx][:-2],
    linewidth=2,
    # linestyle = "--",
    color="blue",
    marker="o",
    markersize=5,
    label="Passive force (Sim, with adhesion)",
)
# plt.xlim(0,1.0)
# plt.ylim(0,30.0)
plt.xlabel("Strain (mm/mm)", fontsize=26)
plt.ylabel("Force (N)", fontsize=26)
plt.rc("font", size=32)  # controls default text sizes
# plt.legend()
plt.tick_params(labelsize=32)
plt.show()

print(strain_no_connection[no_connection_idx], force_no_connection[no_connection_idx])
