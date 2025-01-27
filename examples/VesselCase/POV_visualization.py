""" Rendering Script using POVray

This script reads simulated data file to render POVray animation movie.
The data file should contain dictionary of positions vectors and times.

The script supports multiple camera position where a video is generated
for each camera view.

Notes
-----
    The module requires POVray installed.
"""

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm

from _povmacros import Stages, pyelastica_rod, render

# Setup (USER DEFINE)
DATA_PATH = "vessel_2.dat"  # Path to the simulation data
SAVE_PICKLE = True

# Rendering Configuration (USER DEFINE)
OUTPUT_FILENAME = "pov_vessel_2"
model_path = "3D_models/obj/vessel3"
OUTPUT_IMAGES_DIR = "frames_pov"
if os.path.exists(OUTPUT_IMAGES_DIR):
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

FPS = 20.0
WIDTH = 1920  # 400
HEIGHT = 1080  # 250
DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']

# Camera/Light Configuration (USER DEFINE)
stages = Stages()


# Externally Including Files (USER DEFINE)
# If user wants to include other POVray objects such as grid or coordinate axes,
# objects can be defined externally and included separately.
included = ["default.inc"]
included.append(model_path + "/surface.inc")


# Multiprocessing Configuration (USER DEFINE)
MULTIPROCESSING = True
THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
NUM_AGENT = multiprocessing.cpu_count() // 2  # number of parallel rendering.

# Execute
if __name__ == "__main__":
    # Load Data
    assert os.path.exists(DATA_PATH), "File does not exists"
    try:
        if SAVE_PICKLE:
            import pickle as pk

            with open(DATA_PATH, "rb") as fptr:
                data = pk.load(fptr)
        else:
            # (TODO) add importing npz file format
            raise NotImplementedError("Only pickled data is supported")
    except OSError as err:
        print("Cannot open the datafile {}".format(DATA_PATH))
        print(str(err))
        raise

    # Convert data to numpy array
    times = np.array(data["time"])  # shape: (timelength)
    xs = np.array(data["position"])  # shape: (timelength, 3, num_element)

    # Interpolate Data
    # Interpolation step serves two purposes. If simulated frame rate is lower than
    # the video frame rate, the intermediate frames are linearly interpolated to
    # produce smooth video. Otherwise if simulated frame rate is higher than
    # the video frame rate, interpolation reduces the number of frame to reduce
    # the rendering time.
    runtime = times.max()  # Physical run time
    total_frame = int(runtime * FPS)  # Number of frames for the video
    recorded_frame = times.shape[0]  # Number of simulated frames
    times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale

    xs = interpolate.interp1d(times, xs, axis=0)(times_true)
    times = interpolate.interp1d(times, times, axis=0)(times_true)
    base_radius = 5 / 1000 * np.ones_like(xs[:, 0, :])  # wire radial profile
    # Rendering
    # For each frame, a 'pov' script file is generated in OUTPUT_IMAGE_DIR directory.
    batch = []
    output_path = os.path.join(OUTPUT_IMAGES_DIR, "Rod_POV")
    os.makedirs(output_path, exist_ok=True)
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        current_rod = xs[frame_number]
        endpoint_location = np.array(
            [current_rod[0, -1], current_rod[1, -1], current_rod[2, -1]]
        )
        last2endpoint_location = np.array(
            [current_rod[0, -2], current_rod[1, -2], current_rod[2, -2]]
        )
        tangent = endpoint_location - last2endpoint_location
        look_at_location = endpoint_location + 0.1 * tangent
        stages.add_camera(
            location=endpoint_location,
            angle=120,
            look_at=look_at_location,
            sky=[0, -1, 0],
            name=str(frame_number),
        )

        stages.add_light(
            # Flash light for camera
            position=endpoint_location,
            color=[0.5, 0.5, 0.5],
            camera_id=frame_number,
        )
    stage_scripts = stages.generate_scripts()

    for view_name, stage_script in stage_scripts.items():
        # Colect povray scripts
        script = []
        script.extend(['#include "{}"'.format(s) for s in included])
        script.append(stage_script)

        pov_script = "\n".join(script)

        # Write .pov script file
        file_path = os.path.join(output_path, "frame_{:04d}".format(int(view_name)))
        with open(file_path + ".pov", "w+") as f:
            f.write(pov_script)
        batch.append(file_path)

    # Process POVray
    # For each frames, a 'png' image file is generated in OUTPUT_IMAGE_DIR directory.
    pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
    if MULTIPROCESSING:
        func = partial(
            render,
            width=WIDTH,
            height=HEIGHT,
            display=DISPLAY_FRAMES,
            pov_thread=THREAD_PER_AGENT,
        )
        with Pool(NUM_AGENT) as p:
            for message in p.imap_unordered(func, batch):
                # (TODO) POVray error within child process could be an issue
                pbar.update()
    else:
        for filename in batch:
            render(
                filename,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=multiprocessing.cpu_count(),
            )
            pbar.update()

    # Create Video using moviepy
    imageset_path = output_path
    imageset = [
        os.path.join(imageset_path, path)
        for path in os.listdir(imageset_path)
        if path[-3:] == "png"
    ]
    imageset.sort()
    filename = OUTPUT_FILENAME + "_Rod_POV.mp4"
    clip = ImageSequenceClip(imageset, fps=FPS)
    clip.write_videofile(filename, fps=FPS)
