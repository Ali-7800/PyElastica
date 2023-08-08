__doc__ = """
(added in version 0.3.0)

Customized damper module with exponentially decading damping ratio
"""

from elastica.typing import RodType, SystemType
from elastica.dissipation import DamperBase
from numba import njit
import numpy as np


class AnalyticalLinearDamper_Relaxation(DamperBase):
    """
    This damper class is extended from the AnalyticalLinearDamper module in the elastica core.
    The same analytical dissipation scheme is implemented with a time-dependent damping_constant.
    This mimics the effect of the relaxation_nu function in the orignial C++ elsatica (RSOS 2018), which
    initialize the rod with a high initial damping that exponentially decays to a constant value 
    (normal damping ratio). This implementation will be useful in cases where rods are initialized 
    in the air and dropped to the ground, such as Walker and SnakeOn3DTerrain. The high initial 
    damping will help to prevent the bouncing effect, while the quick decay ensures the normal dynamics
    during ground movement.
    
    damping_constant(time) = initial_damping * exp(- time/relaxation_time) + damping_constant 
    
    Attributes
    ----------
    translational_damping_coefficient: numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on translational velocity.
    rotational_damping_coefficient : numpy.ndarray
        1D array containing data with 'float' type.
        Damping coefficient acting on rotational velocity.
    """

    def __init__(self, damping_constant, initial_damping, relaxation_time, time_step, **kwargs):
        """
        Analytical linear damper initializer

        Parameters
        ----------
        damping_constant : float
            Damping constant after the initial transition.
        initial_damping : float 
        	Initial damping constant.
        relaxation_time: float
        	Constant controlling the decaying speed of the initial_damping.
        time_step : float
            Time-step of simulation.
        """
        super().__init__(**kwargs)
        self._damping_constant = damping_constant
        self._initial_damping = initial_damping
        self._relaxation_time = relaxation_time
        self._time_step = time_step

    def dampen_rates(self, rod: SystemType, time: float):
        
        # Compute the damping coefficient for translational velocity
        damping_constant = (self._damping_constant 
        				 + self._initial_damping*np.exp(-time/self._relaxation_time))
        
        nodal_mass = self._system.mass
        translational_damping_coefficient = np.exp(-damping_constant * self._time_step)

        # Compute the damping coefficient for exponential velocity
        element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
        element_mass[0] += 0.5 * nodal_mass[0]
        element_mass[-1] += 0.5 * nodal_mass[-1]
        rotational_damping_coefficient = np.exp(
            -damping_constant
            * self._time_step
            * element_mass
            * np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )
    
    
        rod.velocity_collection[:] = (
            rod.velocity_collection * translational_damping_coefficient
        )

        rod.omega_collection[:] = rod.omega_collection * np.power(
            rotational_damping_coefficient, rod.dilatation
        )