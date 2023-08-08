import numpy as np
import subprocess
import os
import shutil
from collections import OrderedDict

# Quick dirty logging
import logging

logging.basicConfig(level=logging.ERROR)

from p_tqdm import p_map

# from pathos.multiprocessing import ProcessPool


class ExternalProgram:
    def __init__(self, run_command, base_folder_name="simulation_data", **kwargs):
        self._command = run_command
        if kwargs.pop("exist_remove", False):
            if os.path.exists(base_folder_name):
                shutil.rmtree(base_folder_name)
        self._parent_folder = base_folder_name
        self.sep = kwargs.pop("sep", "_")
        self._child_folder = "phase{sep}space{sep}run{sep}".format(sep=self.sep)
        self.base_folder_name = os.path.join(
            base_folder_name, self._child_folder + "{:05d}",
        )
        self.dependencies = []
        # In case run_command is a binary that needs to be copied
        self.add_dependency(run_command)

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, new_command):
        self._command = new_command
        self.dependencies.clear()
        self.add_dependency(new_command)

    def add_dependency(self, dependency):
        prefix_for_removal = "./"
        if dependency.startswith(prefix_for_removal):
            dependency = dependency[len(prefix_for_removal) :]
        dependency_with_abs_path = os.path.join(os.getcwd(), dependency)
        if os.path.isfile(dependency_with_abs_path):
            self.dependencies.append(dependency_with_abs_path)

    def linearize_dict(self, params: dict):
        # s = [None for _ in range(len(params))]
        arg_prefix = "--"
        incremental_str = [None for _ in range(2 * len(params))]
        for idx, (key, val) in enumerate(params.items()):
            incremental_str[2 * idx] = arg_prefix + key
            incremental_str[2 * idx + 1] = str(val)
        return incremental_str

    def run(self, phase_space_param, phase_space_id):
        # Do a blocking run, wait till the simulation finishes before proceeding
        # on to the next
        logging.debug(
            "Executing simulation %d with args %s",
            phase_space_id,
            str(phase_space_param),
        )
        folder_name = self.base_folder_name.format(phase_space_id)
        abspath_folder = os.path.abspath(folder_name)
        logging.debug("Making directory %s", abspath_folder)
        os.makedirs(abspath_folder, exist_ok=True)
        program_args = ["python3"]
        program_args.extend([self._command])
        program_args.extend(self.linearize_dict(phase_space_param))
        # program_args.append("hansgret.txt")
        for dependency in self.dependencies:
            shutil.copy(dependency, abspath_folder)
        old_dir = os.getcwd()
        os.chdir(abspath_folder)
        # return program_args
        # return psutil.Process().pid
        program_status = subprocess.run(program_args, stdout=subprocess.DEVNULL)
        # Change the process to original directory os that it doesn't create and nest additional
        # directories
        os.chdir(old_dir)
        return program_status.returncode


def make_grid_from_params(input_params: dict):
    from itertools import product

    def cartesian_product(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))

    # A list of dictionaries, each containing keywords passed to the
    # phase space
    # can be a generator object,
    return list(cartesian_product(**input_params))


def post_process_data(
    ext_prog: ExternalProgram,
    phase_space_ids: list,
    phase_space_args: list,
    ep_filename: str = "outputs.txt",
    check_folders: bool = True,
):
    """ Crawls through all simulation folders,
    runs through the output.txt file, gathers
    data from it, puts it in a nice csv format

    :return:
    """
    assert os.path.exists(
        ext_prog._parent_folder
    ), "Simulation data folder not found, unable to post-process"

    # Sort, just be aboslutely sure
    dirs = sorted(
        os.scandir(ext_prog._parent_folder), key=lambda x: (x.is_dir(), x.name)
    )
    logging.debug("All directories in data folder: %s", str(dirs))
    run_dirs = [
        os.path.abspath(f.path)
        for f in dirs
        if f.name.startswith(ext_prog._child_folder)
    ]
    logging.debug("Simulation directories in data folder: %s", str(run_dirs))
    run_dirs_tails = [os.path.basename(f) for f in run_dirs]
    logging.debug("Simulation directory ids in data folder: %s", str(run_dirs_tails))

    if check_folders:
        from warnings import warn

        missing_ids = []
        for p_id, dir_tails in zip(phase_space_ids, run_dirs_tails):
            try:
                detected_id = int(
                    dir_tails.rpartition(ext_prog.sep)[-1]
                )  # Fomr phase_space_run_00001 gives ('phase_space_run', '_', '00001')
            except ValueError:
                detected_id = "6199480080"
            if (p_id) != detected_id:
                missing_ids.append(p_id)
        if missing_ids:
            warn(
                " Some simulation folders are missing! The missing ids are : "
                + str(missing_ids),
                RuntimeWarning,
            )
    logging.info("Checked consistency of simulation folders")

    # Open csv file and write to disk
    import csv
    import numpy as np

    id_file_name = "ids.csv"

    with open(id_file_name, "w", newline="") as id_file_handler:
        # Write header first
        csvwriter = csv.writer(id_file_handler, delimiter=",")
        statistics = [
            "effective_vel",
        ]
        header_row = ["id"] + list(phase_space_args[0].keys()) + statistics
        csvwriter.writerow(header_row)
        for p_id, args_dict, output_dir in zip(
            phase_space_ids, phase_space_args, run_dirs
        ):
            logging.debug("Begin processing %s", output_dir)
            temp = [p_id]
            temp.extend(args_dict.values())
            simulation_output_file = os.path.join(output_dir, ep_filename)
            data = np.loadtxt(simulation_output_file, delimiter=" ", ndmin=1)
            temp.extend(data)
            csvwriter.writerow(temp)
    logging.info("Wrote all to csv file")


def main(dry_run=True):
    ext_program = ExternalProgram("continuum_snake.py", exist_remove=(True and not dry_run))

    if True:
        # Good for a parameter sweep, constructs a grid from these
        # parameters and each point is simulated
        phase_space_kwargs = OrderedDict(
            {"wave_length": np.array([0.5]),  
             "timestep": np.linspace(2e-4,5e-4,num=4,endpoint=True),
             "damping": np.linspace(3e-2,3e-1,num=5,endpoint=True),}
        )

        # A list each containing a dict of params to run,
        phase_space_args = make_grid_from_params(phase_space_kwargs)
    else:
        # Here you can specify individual parameters, more useful for a finer run
        # when you know that you only want to run these points
        phase_space_args = [
            {"wave_length": 1.0, "timestep": 1e-4, "damping": 2e-3},
        ]

    phase_space_ids = list(range(1, len(phase_space_args) + 1))

    num_cpus = 4
    logging.info(
        "Begin %d simulations with batch size of %d", len(phase_space_ids), num_cpus
    )
    logging.debug("The ids to be passed are %s", str(phase_space_ids))
    logging.debug("The arguments to be passed are %s", str(phase_space_args))

    # pool = ProcessPool(nodes=4)
    retcodes = (
        p_map(ext_program.run, phase_space_args, phase_space_ids, num_cpus=4)
        if not dry_run
        else [0 for _ in phase_space_args]
    )
    # home = os.path.expanduser("~/code/gravitas/")
    # paths = [ f.path for f in os.scandir(home) if not f.is_dir() ]
    # results = pool.imap(pc.print_path, paths)
    # Check if all codes are fine and that nothing went wrong
    for code, p_args, p_id in zip(retcodes, phase_space_args, phase_space_ids):
        import sys

        if code < 0:
            print(
                "Simulation id {id} with args {args} failed!".format(
                    id=p_id, args=p_args
                ),
                file=sys.stderr,
            )

    post_process_data(ext_program, phase_space_ids, phase_space_args)

    return retcodes


if __name__ == "__main__":
    results = main(dry_run=False)
