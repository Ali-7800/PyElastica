"""
Created on Dec. 20, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import sys

sys.path.append("../../../../")  # include elastica-python directory
sys.path.append("../../")  # include ActuationModel directory

import pickle
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from elastica import Mesh
from functools import partial
import subprocess

from _rendering_tool import check_folder
from povray import (
    POVRAYFrame,
    POVRAYCamera,
)
from povray.rod import POVRAYRod
from povray.sphere import POVRAYSphere


def render(
    filename, width, height, antialias="on", quality=6, display="Off", pov_thread=4
):
    """Rendering frame

    Generate the povray script file '.pov' and image file '.png'
    The directory must be made before calling this method.

    Parameters
    ----------
    filename : str
        POV filename (without extension)
    width : int
        The width of the output image.
    height : int
        The height of the output image.
    antialias : str ['on', 'off']
        Turns anti-aliasing on/off [default='on']
    quality : int
        Image output quality. [default=11]
    display : str
        Turns display option on/off during POVray rendering. [default='off']
    pov_thread : int
        Number of thread per povray process. [default=4]
        Acceptable range is (4,512).
        Refer 'Symmetric Multiprocessing (SMP)' for further details
        https://www.povray.org/documentation/3.7.0/r3_2.html#r3_2_8_1

    Raises
    ------
    IOError
        If the povray run causes unexpected error, such as parsing error,
        this method will raise IOerror.

    """

    # Define script path and image path
    script_file = filename
    image_file = filename + ".png"

    # Run Povray as subprocess
    cmds = [
        "povray",
        "+I" + script_file,
        "+O" + image_file,
        f"-H{height}",
        f"-W{width}",
        f"Work_Threads={pov_thread}",
        f"Antialias={antialias}",
        f"Quality={quality}",
        f"Display={display}",
    ]
    process = subprocess.Popen(
        cmds, stderr=subprocess.PIPE, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    _, stderr = process.communicate()

    # Check execution error
    if process.returncode:
        print(type(stderr), stderr)
        raise IOError(
            "POVRay rendering failed with the following error: "
            + stderr.decode("ascii")
        )


def main(filename):
    with open(filename + "_data.pickle", "rb") as f:
        data = pickle.load(f)
        rod_data = data["systems"]
        view = int(input("choose view: 1,2,3,4"))
        env_idx = int(
            input(
                "choose env_idx: 0 for flat, 1 for m32_Viekoda_Bay, 2 for mars-landscape"
            )
        )

    povray_data_folder = filename + "_povray"
    check_folder(povray_data_folder)
    scale = 16
    # povray_camera = POVRAYCamera(
    #     position=np.array([-5.0 / 4, 1.5 / 4, 0.6 / 2]) * scale,
    #     # position=[0.5, -5.0, 5.0],
    #     # position=[0.0, 0.0, 20.0],
    #     look_at=[0.0, 0.0, 0.6 / 2],
    #     # look_at=[0.0, 0.0, 0.0],
    #     angle=30.0,
    #     floor=False,
    # )

    FPS = 20.0
    WIDTH = 1080  # 400
    HEIGHT = 1080  # 250
    DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']
    # Multiprocessing Configuration (USER DEFINE)
    MULTIPROCESSING = True
    THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
    NUM_AGENT = 1  # number of parallel rendering.

    if view == 1:
        povray_camera = POVRAYCamera(
            position=[3 / 2, -5.0 / 2, 0.6 / 2],
            # position=[0.5, -5.0, 5.0],
            # position=[0.0, 0.0, 20.0],
            look_at=[0.0, 0.0, 0.6 / 2],
            # look_at=[0.0, 0.0, 0.0],
            angle=40.0,
            floor=False,
        )
    elif view == 2:
        povray_camera = POVRAYCamera(
            position=[0.0, 0.0, 3 * 2.5 / 2],
            # position=[0.5, -5.0, 5.0],
            # position=[0.0, 0.0, 20.0],
            look_at=[0.0, 0.0, 0.6 / 2],
            # look_at=[0.0, 0.0, 0.0],
            angle=40.0,
            floor=False,
        )
    elif view == 3:
        povray_camera = POVRAYCamera(
            position=[0.0, 2.0, 1.0],
            # position=[0.5, -5.0, 5.0],
            # position=[0.0, 0.0, 20.0],
            look_at=[0.0, 0.0, -0.1],
            # look_at=[0.0, 0.0, 0.0],
            # sun_position= [0,0,2000],
            angle=40.0,
            floor=False,
        )
    elif view == 4:
        povray_camera = POVRAYCamera(
            position=[0.5, 0.5, 0.0],
            # position=[0.5, -5.0, 5.0],
            # position=[0.0, 0.0, 20.0],
            look_at=[0.0, 0.0, 0.0],
            # look_at=[0.0, 0.0, 0.0],
            angle=40.0,
            floor=False,
        )

    arm_base_radius = 0.2 / 20 * 0.8
    arm_radius = np.linspace(arm_base_radius, arm_base_radius / 10, 100)

    povray_frame = POVRAYFrame(
        included_files=[
            povray_data_folder + "/camera.inc",
            povray_data_folder + "/frame0000.inc",
            povray_data_folder + "/surface.inc",
        ]
    )

    povray_rod = POVRAYRod(color="<0.45, 0.39, 1.0>", scale=scale)
    povray_head = POVRAYRod(color="<0.45, 0.39, 1.0>", scale=scale)
    povray_target = POVRAYSphere(color=np.array([1.0, 0.498039, 0.0]))

    print("Exporting povray files and frames ...")
    frame_camera_name = "camera.inc"
    with open(povray_data_folder + "/" + frame_camera_name, "w") as file_camera:
        povray_camera.write_to(file_camera)

    radius_base = 0.2 / 20

    if env_idx == 0:
        import os

        if os.path.exists(povray_data_folder + "/surface.inc"):
            os.remove(povray_data_folder + "/surface.inc")
        surface_inc = open(povray_data_folder + "/surface.inc", "w")
        surface_inc_text = """// ground -----------------------------------------------------------------
//---------------------------------<<< settings of squared plane dimensions
#declare RasterScale = 0.50;
#declare RasterHalfLine  = 0.03;
#declare RasterHalfLineZ = 0.03;
//-------------------------------------------------------------------------
#macro Raster(RScale, HLine)
       pigment{ gradient x scale RScale
                color_map{[0.000   color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,0>*0.6]
                          [1.000   color rgbt<1,1,1,0>*0.6]} }
#end// of Raster(RScale, HLine)-macro
//-------------------------------------------------------------------------

plane { <0,0,1>, -0.01    // plane with layered textures
        texture { pigment{color White*1.1}
                  finish {ambient 0.45 diffuse 0.85}
              }
        texture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }
        texture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}
        rotate<0,0,0>
      }
//------------------------------------------------ end of squared plane XZ"""
        surface_inc.write(surface_inc_text)
        surface_inc.close()
    elif env_idx == 1:
        mesh = Mesh(filepath=r"m32_Viekoda_Bay/m32_Viekoda_Bay.obj")
        mesh.translate(-np.array(mesh.mesh_center))
        mesh.translate(
            -np.array([0, 0, 4 * radius_base + np.min(mesh.face_centers[2])])
        )
        mesh.scale(np.ones(3) * scale)
        mesh.povray_mesh_export(
            texture_path=r"m32_Viekoda_Bay/m32_Viekoda_Bay.jpg",
            # color = [188/255,158/255,130/255,0.0],
            export_to=povray_data_folder,
        )
    elif env_idx == 2:
        mesh = Mesh(filepath=r"mars-landscape/model.obj")
        mesh.translate(-np.array(mesh.mesh_center))
        mesh.rotate(axis=np.array([1.0, 0.0, 0.0]), angle=90)
        mesh.scale(np.array([10.0, 10.0, 10.0]) / np.max(mesh.mesh_scale))
        mesh.translate(np.array([0.0, 0.0, 0.05]))
        mesh.scale(np.ones(3) * scale)
        mesh.povray_mesh_export(
            texture_path=r"mars-landscape/texture.jpg",
            normal_path=r"mars-landscape/normal.jpg",
            export_to=povray_data_folder,
        )
    frames = range(len(rod_data[0]["time"]))

    head_pos = []
    arm_tip_pos = [[] for _ in range(len(rod_data[:-1]))]
    ground_height = []

    for k in tqdm(frames):
        frame_inc_name = "frame%04d.inc" % k
        with open(povray_data_folder + "/" + frame_inc_name, "w") as file_inc:
            for i, rod_i in enumerate(rod_data[:-1]):
                povray_rod.write_to(
                    file=file_inc,
                    position_data=rod_i["position"][k],
                    radius_data=arm_radius,  # rod_i["radius"][k],
                    alpha=1.0,
                )

                arm_tip_pos[i].append(np.mean(rod_i["position"][k], axis=-1))
            povray_head.write_to(
                file=file_inc,
                position_data=rod_data[-1]["position"][k],
                radius_data=rod_data[-1]["radius"][k],
                alpha=1.0,
            )
            head_pos.append(np.mean(rod_data[-1]["position"][k], axis=-1))

        frame_povray_name = "frame%04d.pov" % k
        povray_frame.included_files[1] = povray_data_folder + "/" + frame_inc_name
        # povray_frame.included_files.append(povray_data_folder + "/surface.inc")

        with open(povray_data_folder + "/" + frame_povray_name, "w") as file_frame:
            povray_frame.write_included_files_to(file_frame)

        povray_frame_file_name = povray_data_folder + "/" + frame_povray_name
        import subprocess

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
        # batch.append(povray_frame_file_name)

    # pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
    # if MULTIPROCESSING:
    #     func = partial(
    #         render,
    #         width=WIDTH,
    #         height=HEIGHT,
    #         display=DISPLAY_FRAMES,
    #         pov_thread=THREAD_PER_AGENT,
    #     )
    #     with Pool(NUM_AGENT) as p:
    #         for message in p.imap_unordered(func, batch):
    #             # (TODO) POVray error within child process could be an issue
    #             pbar.update()
    # else:
    #     for filename in batch:
    #         render(
    #             filename,
    #             width=WIDTH,
    #             height=HEIGHT,
    #             display=DISPLAY_FRAMES,
    #             pov_thread=multiprocessing.cpu_count(),
    #         )
    #         pbar.update()

    np.savez("traj", head=head_pos, tip=arm_tip_pos)


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
