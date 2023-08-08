__doc__ = """Timestepping utilities to be used with Rod and RigidBody classes"""
__all__ = [
    "integrate",
    "PositionVerlet",
    "PEFRL",
    "RungeKutta4",
    "EulerForward",
    "extend_stepper_interface",
]

import numpy as np
from tqdm import tqdm
from elastica.timestepper.symplectic_steppers import (
    SymplecticStepperTag,
    PositionVerlet,
    PEFRL,
)
from elastica.timestepper.explicit_steppers import (
    ExplicitStepperTag,
    RungeKutta4,
    EulerForward,
)

from elastica.timestepper.__init__ import(
    extend_stepper_interface
)


class BoundaryError(Exception):
    "Raised when snake leaves surface boundary"
    pass

# TODO Improve interface of this function to take args and kwargs for ease of use
def snake_integrate(
    StatefulStepper,
    System,
    final_time: float,
    n_steps: int = 1000,
    restart_time: float = 0.0,
    progress_bar: bool = True,
    **kwargs,
):
    """

    Parameters
    ----------
    StatefulStepper :
        Stepper algorithm to use.
    System :
        The elastica-system to simulate.
    final_time : float
        Total simulation time. The timestep is determined by final_time / n_steps.
    n_steps : int
        Number of steps for the simulation. (default: 1000)
    restart_time : float
        The timestamp of the first integration step. (default: 0.0)
    progress_bar : bool
        Toggle the tqdm progress bar. (default: True)
    """
    assert final_time > 0.0, "Final time is negative!"
    assert n_steps > 0, "Number of integration steps is negative!"

    # Extend the stepper's interface after introspecting the properties
    # of the system. If system is a collection of small systems (whose
    # states cannot be aggregated), then stepper now loops over the system
    # state
    do_step, stages_and_updates = extend_stepper_interface(StatefulStepper, System)

    dt = np.float64(float(final_time) / n_steps)
    time = restart_time

    for i in tqdm(range(n_steps), disable=(not progress_bar)):
        try:
            time = do_step(StatefulStepper, stages_and_updates, System, time, dt)
        except BoundaryError:
            print("Snake left surface boundary, ending simulation")
            break



    print("Final time of simulation is : ", time)
    return time
