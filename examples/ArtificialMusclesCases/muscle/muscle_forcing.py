import numpy as np
from elastica import *
from elastica._rotations import _inv_rotate, _skew_symmetrize
from numba import njit


class EndpointForcesWithStartTime(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type. Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up for ramp up time.
        start_time: float
            Forces are applied after this start time.
        end_time: float
            Forces are applied until this end time.

    """

    def __init__(
        self,
        start_force,
        end_force,
        ramp_up_time,
        ramp_down_time,
        start_time,
        end_time=np.infty,
    ):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to first node of the system.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        ramp_down_time: float
            Applied forces are ramped down until ramp down time.
        start_time: float
            Forces are applied after this start time.
        end_time: float
            Forces are applied until this end time.

        """
        super(EndpointForcesWithStartTime, self).__init__()
        self.start_force = start_force
        self.end_force = end_force
        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.start_time = start_time
        self.end_time = end_time

    def apply_forces(self, system: SystemType, time=0.0):
        self.compute_end_point_forces_with_start_time(
            system.external_forces,
            self.start_force,
            self.end_force,
            time,
            self.ramp_up_time,
            self.ramp_down_time,
            self.start_time,
            self.end_time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces_with_start_time(
        external_forces,
        start_force,
        end_force,
        time,
        ramp_up_time,
        ramp_down_time,
        start_time,
        end_time,
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        start_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
        end_force: numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Force applied to last node of the system.
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.
        start_time: float
            Forces are applied after this start time.
        end_time: float
            Forces are applied until this end time.

        Returns
        -------

        """
        if (time > start_time) and (time < end_time):
            factor = min(1.0, (time - start_time) / ramp_up_time)
            external_forces[..., 0] += start_force * factor
            external_forces[..., -1] += end_force * factor
        elif time >= end_time:
            factor = max(0.0, 1 - (time - end_time) / ramp_down_time)
            external_forces[..., 0] += start_force * factor
            external_forces[..., -1] += end_force * factor


class ElementwiseForcesAndTorques(NoForces):
    """ """

    def __init__(self, torques, forces):
        super(ElementwiseForcesAndTorques, self).__init__()
        self.torques = torques
        self.forces = forces

    def apply_forces(self, system, time: np.float64 = 0.0):
        system.external_torques -= self.torques
        system.external_forces -= self.forces


class PointSpring(NoForces):
    """ """

    def __init__(self, k, nu, point, index, *args, **kwargs):
        super(PointSpring, self).__init__()
        self.point = point
        self.k = k
        self.index = index
        self.nu = nu

    def apply_forces(self, system, time: np.float64 = 0.0):
        elastic_force = self.k * (
            self.point - system.position_collection[..., self.index]
        )
        damping_force = -self.nu * (system.velocity_collection[..., self.index])
        system.external_forces[..., self.index] += elastic_force + damping_force


class MeshRigidBodyPointSpring(NoForces):
    def __init__(
        self,
        k,
        nu,
        distance_to_point_from_center,
        direction_to_point_from_center,
        point,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k: float
           Stiffness coefficient of the joint.
        nu: float
           Damping coefficient of the joint.
        distance_to_point_from_center : float
            distance of point spring connection on the rigid body with respect to rigid body center.
        direction_to_point_from_center : numpy array (3,)
            direction of point spring connection on the rigid body with respect to rigid body center.
        point : numpy array (3,)
            position of the spring connection point with respect to the world frame



        """
        try:
            self.block_size = len(k)
        except:
            self.block_size = 1
        if self.block_size == 1:
            self.k = [k]
            self.nu = [nu]
            self.distance_to_point_from_center = [distance_to_point_from_center]
            self.direction_to_point_from_center = [direction_to_point_from_center]
            self.point = [point]
        else:
            self.k = k
            self.nu = nu
            self.distance_to_point_from_center = distance_to_point_from_center
            self.direction_to_point_from_center = direction_to_point_from_center
            self.point = point

    def apply_forces(self, system, time: np.float64 = 0.0):
        """
        Apply joint force to the connected rod objects.

        Parameters
        ----------
        system : object
            Mesh rigid-body object
        Returns
        -------

        """

        for block in range(self.block_size):
            current_point_position = np.zeros((3,))
            current_point_velocity = np.zeros((3,))
            system_omega_collection_skew = _skew_symmetrize(system.omega_collection)
            for i in range(3):
                current_point_position[i] += system.position_collection[i, 0]
                current_point_velocity[i] += system.velocity_collection[i, 0]
                for j in range(3):
                    current_point_position[i] += (
                        self.distance_to_point_from_center[block]
                        * system.director_collection[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # rp = rcom + dQN
                    current_point_velocity[i] += (
                        self.distance_to_point_from_center[block]
                        * system_omega_collection_skew[i, j, 0]
                        * ((self.direction_to_point_from_center[block])[j])
                    )  # vp = vcom + d(wxN)

            end_distance_vector = self.point[block] - current_point_position
            elastic_force = self.k[block] * end_distance_vector
            damping_force = self.nu[block] * current_point_velocity

            contact_force = elastic_force + damping_force
            system.external_forces[..., 0] += contact_force
            system.external_torques[..., 0] -= self.distance_to_point_from_center[
                block
            ] * np.cross(
                system.director_collection[..., 0]
                @ self.direction_to_point_from_center[block],
                contact_force,
            )

        return

    def apply_torques(self, system, time: np.float64 = 0.0):
        """
        Apply restoring joint torques to the connected rod objects.

        In FreeJoint class, this routine simply passes.

        Parameters
        ----------
        system : object
            Rod or rigid-body object

        Returns
        -------

        """
        pass
