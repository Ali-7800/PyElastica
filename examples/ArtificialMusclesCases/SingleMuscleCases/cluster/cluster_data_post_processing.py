import numpy as np
import os
import matplotlib
from examples.ArtificialMusclesCases import *

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

data = {}
test_muscle = Samuel_supercoil_stl()
muscle_name = test_muscle.name
test_type = "Isobaric"
testing_range = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0])

for i in testing_range[1:]:
    file_path = (
        "cluster_data/"
        + muscle_name
        + "_"
        + test_type
        + "_"
        + str(i)
        + "/data/PassiveForceTest"
        + muscle_name
        + ".npz"
    )
    data[str(i)] = np.load(file_path)

strain_sim = np.zeros_like(testing_range)

for k, i in enumerate(testing_range[1:]):
    current_position = data[str(i)]["muscle_rods_position_history"]
    n_rods = current_position.shape[0]
    centerline_position = np.zeros(current_position.shape[1:])
    for j in range(n_rods):
        centerline_position += current_position[j, :, :, :] / n_rods
    strain_sim[k + 1] = np.dot(
        test_muscle.geometry.direction,
        (
            (centerline_position[-1, :, -1] - centerline_position[-1, :, 0])
            - (centerline_position[0, :, -1] - centerline_position[0, :, 0])
        )
        / (centerline_position[0, :, -1] - centerline_position[0, :, 0]),
    )

plt.rc("font", size=8)  # controls default text sizes
plt.plot(
    strain_sim,
    testing_range,
    linewidth=3,
    marker="o",
    markersize=7,
    label="Passive force (Sim,Isobaric)",
)
plt.rc("font", size=8)  # controls default text sizes
plt.suptitle("Coil Strain vs Force")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Force (N)")
plt.legend()
plt.savefig("plot.png", dpi=300)
plt.show()
