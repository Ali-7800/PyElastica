"""
Created on Dec. 20, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import sys

sys.path.append("../../../../")  # include elastica-python directory
sys.path.append("../../")  # include ActuationModel directory

import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from elastica import Mesh

from _rendering_tool import check_folder
from povray import (
    POVRAYFrame,
    POVRAYCamera,
)
from povray.rod import POVRAYRod
from povray.sphere import POVRAYSphere


def main(filename):
    with open(filename + "_data.pickle", "rb") as f:
        data = pickle.load(f)
        rod_data = data["systems"]
        with_obstacle = bool(int(input("with obstacle? 0: no 1: yes")))
        with_mesh = bool(int(input("with mesh? 0: no 1:yes")))
        if with_obstacle:
            sphere_data = data["systems"][-1]

    povray_data_folder = filename + "_povray"
    check_folder(povray_data_folder)

    povray_camera = POVRAYCamera(
        position=[1.5 / 4, -5.0 / 4, 0.6 / 2],
        # position=[0.5, -5.0, 5.0],
        # position=[0.0, 0.0, 20.0],
        look_at=[0.0, 0.0, 0.6 / 2],
        # look_at=[0.0, 0.0, 0.0],
        angle=40.0,
        floor=False,
    )

    if with_mesh:
        povray_frame = POVRAYFrame(
            included_files=[
                povray_data_folder + "/camera.inc",
                povray_data_folder + "/frame0000.inc",
                povray_data_folder + "/surface.inc",
            ]
        )
    else:
        povray_frame = POVRAYFrame(
            included_files=[
                povray_data_folder + "/camera.inc",
                povray_data_folder + "/frame0000.inc",
            ]
        )
    povray_rod = POVRAYRod(color="<0.45, 0.39, 1.0>")
    povray_head = POVRAYRod(color="<0.8, 0.0, 0.0>")
    povray_target = POVRAYSphere(color=np.array([1.0, 0.498039, 0.0]))

    print("Exporting povray files and frames ...")
    frame_camera_name = "camera.inc"
    with open(povray_data_folder + "/" + frame_camera_name, "w") as file_camera:
        povray_camera.write_to(file_camera)

    if with_mesh:
        mesh = Mesh(filepath=r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj")
        mesh.translate(-np.array(mesh.mesh_center))
        radius_base = 0.2 / 20  # 0.012
        mesh.translate(
            -4 * radius_base - np.array([0, 0, np.min(mesh.face_centers[2])])
        )
        mesh.povray_mesh_export(
            texture_path=r"m32_Viekoda_Bay/m32_Viekoda_Bay.jpg",
            export_to=povray_data_folder,
        )

    for k in tqdm(range(len(rod_data[0]["time"]))):
        frame_inc_name = "frame%04d.inc" % k
        with open(povray_data_folder + "/" + frame_inc_name, "w") as file_inc:
            for rod_i in rod_data[:-1]:
                povray_rod.write_to(
                    file=file_inc,
                    position_data=rod_i["position"][k],
                    radius_data=rod_i["radius"][k],
                    alpha=1.0,
                )
            povray_head.write_to(
                file=file_inc,
                position_data=rod_data[-1]["position"][k],
                radius_data=rod_data[-1]["radius"][k],
                alpha=1.0,
            )
            if with_obstacle:
                povray_target.write_to(
                    file=file_inc,
                    position_data=sphere_data["position"][k],
                    radius_data=sphere_data["radius"][k],
                    alpha=1.0,
                )

        frame_povray_name = "frame%04d.pov" % k
        povray_frame.included_files[1] = povray_data_folder + "/" + frame_inc_name
        if with_mesh:
            povray_frame.included_files.append(povray_data_folder + "/surface.inc")
        with open(povray_data_folder + "/" + frame_povray_name, "w") as file_frame:
            povray_frame.write_included_files_to(file_frame)

        povray_frame_file_name = povray_data_folder + "/" + frame_povray_name
        import subprocess

        # subprocess.run(["povray", "-H400", "-W600", "Quality=11", "Antialias=on", povray_frame_file_name])
        subprocess.run(
            [
                "povray",
                "-H1080",
                "-W1080",
                "Quality=11",
                "Antialias=on",
                povray_frame_file_name,
            ]
        )
        # subprocess.run(["povray", "-H2100", "-W3150", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H480", "-W480", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H960", "-W960", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H1920", "-W1920", "Quality=11", "Antialias=on", povray_frame_file_name])
        # subprocess.run(["povray", "-H3840", "-W3840", "Quality=11", "Antialias=on", povray_frame_file_name])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export simulation result as povray files and frames in filename_povray."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="simulation",
        help="a str: data file name",
    )
    args = parser.parse_args()
    main(filename=args.filename)
