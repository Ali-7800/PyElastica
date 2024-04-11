import numpy as np
from elastica.contact_forces import NoContact
from elastica.typing import RodType, SystemType, AllowedContactType
from elastica.rod import RodBase
from elastica.surface import MeshSurface
from numpy.testing import assert_allclose
from elastica.utils import Tolerance


from numba import njit
from elastica.contact_utils import (
    _elements_to_nodes_inplace,
    _find_slipping_elements,
    _node_to_element_position,
    _node_to_element_velocity,
)
from elastica._linalg import (
    _batch_matvec,
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_product_k_ik_to_ik,
    _batch_matrix_transpose,
)

from surface_functions import calculate_facet_normals_centers


@njit(cache=True)
def _find_face_neighborhood(face_centers, point, neighborhood_size):
    """
    Finds indices of faces with centers within a sphere of neighborhood_size around the point
    Parameters
    ----------
    face_centers
    point
    neighborhood_size

    Returns
    -------
    neighborhood_idx

    Notes
    -----
    Not benchmarked

    """
    deltas = face_centers - np.array([point[0], point[1], point[2]]).reshape((3, 1))
    distances = _batch_norm(deltas)
    neighborhood_idx = np.where(distances < neighborhood_size)[0]
    return neighborhood_idx


@njit(cache=True)
def _find_closest_faces(faces_centers, positions):
    """
    Finds indices of the closest face to each element
    ----------
    faces_centers
    positions

    Returns
    -------
    closest face

    Notes
    -----
    Not benchmarked

    """
    n_positions = positions.shape[-1]
    closest_faces = np.empty((n_positions,))
    for i in range(n_positions):
        deltas = faces_centers - np.array(
            [positions[0, i], positions[1, i], positions[2, i]]
        ).reshape((3, 1))
        distances = _batch_norm(deltas)
        if distances.size != 0:
            closest_faces[i] = np.argmin(distances)
    return closest_faces


@njit(cache=True)
def _find_closest_face(faces_centers, position):
    """
    Finds indices of the closest face to position
    ----------
    facets_centers
    positions

    Returns
    -------
    closest face

    Notes
    -----
    Not benchmarked

    """
    deltas = faces_centers - np.array([position[0], position[1], position[2]]).reshape(
        (3, 1)
    )
    distances = _batch_norm(deltas)
    closest_face = np.argmin(distances)
    return closest_face


def _find_contact_faces_idx(
    faces_grid,
    x_min,
    y_min,
    grid_size,
    position_collection,
):

    element_position = _node_to_element_position(position_collection)
    n_element = element_position.shape[-1]
    position_idx_array = np.empty((0))
    face_idx_array = np.empty((0))
    grid_position = np.round(
        (element_position[0:2, :] - np.array([x_min, y_min]).reshape((2, 1)))
        / grid_size
    )

    # find facet neighborhood of each element position #WHY is this always correct?
    for i in range(n_element):
        try:
            face_idx_1 = faces_grid[
                (int(grid_position[0, i]), int(grid_position[1, i]))
            ]  # first quadrant
        except:
            face_idx_1 = np.empty((0))
        try:
            face_idx_2 = faces_grid[
                (int(grid_position[0, i] - 1), int(grid_position[1, i]))
            ]  # second quadrant
        except:
            face_idx_2 = np.empty((0))
        try:
            face_idx_3 = faces_grid[
                (int(grid_position[0, i] - 1), int(grid_position[1, i] - 1))
            ]  # third quadrant
        except:
            face_idx_3 = np.empty((0))
        try:
            face_idx_4 = faces_grid[
                (int(grid_position[0, i]), int(grid_position[1, i] - 1))
            ]  # fourth quadrant
        except:
            face_idx_4 = np.empty((0))
        face_idx_element = np.concatenate(
            (face_idx_1, face_idx_2, face_idx_3, face_idx_4)
        )
        face_idx_element_no_duplicates = np.unique(face_idx_element)
        if face_idx_element_no_duplicates.size == 0:
            raise Exception(
                "rod outside surface boundary"
            )  # a rod element is on four grids with no facets

        face_idx_array = np.concatenate(
            (face_idx_array, face_idx_element_no_duplicates)
        )
        n_contacts = face_idx_element_no_duplicates.shape[0]
        position_idx_array = np.concatenate(
            (position_idx_array, i * np.ones((n_contacts,)))
        )

    position_idx_array = position_idx_array.astype(int)
    face_idx_array = face_idx_array.astype(int)
    return position_idx_array, face_idx_array, element_position


############################################## Multiple contacts Grid


@njit(cache=True)
def _calculate_normal_forces_rod_mesh_surface(
    face_normals,
    face_centers,
    element_position,
    position_idx_array,
    face_idx_array,
    surface_tol,
    k,
    nu,
    radius,
    mass,
    velocity_collection,
    external_forces,
):
    """
    This function computes the plane force response on the element, in the
    case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
    is used.

    Parameters
    ----------
    system

    Returns
    -------
    magnitude of the plane response
    """

    # Damping force response due to velocity towards the plane
    element_velocity = _node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

    if len(face_idx_array) > 0:
        element_position_contacts = element_position[:, position_idx_array].astype(
            np.float64
        )
        contact_face_centers = face_centers[:, face_idx_array].astype(np.float64)
        normals_on_elements = face_normals[:, face_idx_array].astype(np.float64)
        radius_contacts = radius[position_idx_array].astype(np.float64)
        element_velocity_contacts = element_velocity[:, position_idx_array].astype(
            np.float64
        )

    else:
        element_position_contacts = element_position.astype(np.float64)
        contact_face_centers = np.zeros_like(element_position).astype(np.float64)
        normals_on_elements = np.zeros_like(element_position).astype(np.float64)
        radius_contacts = radius.astype(np.float64)
        element_velocity_contacts = element_velocity.astype(np.float64)

    # Elastic force response due to penetration
    distance_from_plane = _batch_dot(
        normals_on_elements, element_position_contacts - contact_face_centers
    )
    # print(contact_face_centers)
    plane_penetration = np.minimum(distance_from_plane - radius_contacts, 0.0)
    elastic_force = -k * _batch_product_k_ik_to_ik(
        plane_penetration, normals_on_elements
    )
    normal_component_of_element_velocity = _batch_dot(
        normals_on_elements, element_velocity_contacts
    )
    damping_force = -nu * _batch_product_k_ik_to_ik(
        normal_component_of_element_velocity, normals_on_elements
    )

    # Compute total plane response force
    plane_response_force_contacts = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where(
        (distance_from_plane - radius_contacts) > surface_tol
    )[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_contact_point_idx] = 0.0

    plane_response_forces = np.zeros_like(external_forces)
    for i in range(len(position_idx_array)):
        plane_response_forces[
            :, position_idx_array[i]
        ] += plane_response_force_contacts[:, i]

    # Update the external forces
    _elements_to_nodes_inplace(plane_response_forces, external_forces)
    return (
        _batch_norm(plane_response_force_contacts),
        no_contact_point_idx,
        normals_on_elements,
    )


class RodMeshSurfaceContactWithGrid(NoContact):
    def __init__(
        self,
        k,
        nu,
        slip_velocity_tol,
        static_mu_array,
        kinetic_mu_array,
        faces_grid,
        gamma,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        faces_grid: Dict
            Dictionary that describes the faces withing each grid square
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        """
        self.k = k
        self.nu = nu
        self.faces_grid = faces_grid

        self.surface_tol = 1e-4
        self.grid_size = faces_grid["grid_size"]
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
        ) = static_mu_array
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array
        self.gamma = gamma

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodMeshSurfaceContactWithGrid class first_system should be a rod and second_system should be a mesh surface.

        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
            system_two.__class__, MeshSurface
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a mesh surface".format(
                    system_one.__class__, system_two.__class__
                )
            )
        # check if provided grid is valid
        self.x_min = np.min(system_two.faces[0, :, :])
        self.y_min = np.min(system_two.faces[1, :, :])
        base_length = system_one.rest_lengths.sum()
        # print(2*max(system_one.radius),base_length/system_one.n_elems)
        needed_grid_size = max(
            2 * max(system_one.radius), base_length / system_one.n_elems
        )
        assert_allclose(
            self.grid_size,
            needed_grid_size,
            Tolerance.rtol(),
            Tolerance.atol(),
            err_msg=(
                "Provided grid does not work with given rod. \n"
                "Provided grid size is {0} and needed grid size is {1}".format(
                    self.grid_size, needed_grid_size
                )
            ),
        )

    # kinetic and static friction should separate functions
    # for now putting them together to figure out common variables
    def apply_contact(
        self, system_one: RodType, system_two: AllowedContactType
    ) -> None:
        """
        Apply contact forces and torques between RodType object and mesh surface object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Rod object.

        """
        (
            self.position_idx_array,
            self.face_idx_array,
            self.element_position,
        ) = _find_contact_faces_idx(
            self.faces_grid,
            self.x_min,
            self.y_min,
            self.grid_size,
            system_one.position_collection,
        )

        _calculate_contact_forces_rod_mesh_surface(
            system_two.face_centers,
            system_two.face_normals,
            self.element_position,
            self.position_idx_array,
            self.face_idx_array,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.gamma,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system_one.radius,
            system_one.mass,
            system_one.tangents,
            system_one.position_collection,
            system_one.director_collection,
            system_one.velocity_collection,
            system_one.omega_collection,
            system_one.internal_forces,
            system_one.external_forces,
            system_one.internal_torques,
            system_one.external_torques,
            system_one.lengths,
        )


@njit(cache=True)
def _calculate_contact_forces_rod_mesh_surface(
    face_centers,
    face_normals,
    element_position,
    position_idx_array,
    face_idx_array,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    gamma,
    static_mu_forward,
    static_mu_backward,
    static_mu_sideways,
    radius,
    mass,
    tangents,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
    lengths,
):

    (
        plane_response_force_mag,
        no_contact_point_idx,
        normals_on_elements,
    ) = _calculate_normal_forces_rod_mesh_surface(
        face_normals,
        face_centers,
        element_position,
        position_idx_array,
        face_idx_array,
        surface_tol,
        k,
        nu,
        radius,
        mass,
        velocity_collection,
        external_forces,
    )
    # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
    # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
    # to the plane. So friction forces can only be in plane forces and not out of plane.

    element_velocity = _node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

    if len(position_idx_array) > 0:
        tangents_contacts = tangents[:, position_idx_array]
        element_velocity_contacts = element_velocity[:, position_idx_array]
        radius_contacts = radius[position_idx_array]
        omega_collection_contacts = omega_collection[:, position_idx_array]
        director_collection_contacts = director_collection[:, :, position_idx_array]
    else:
        tangents_contacts = tangents
        element_velocity_contacts = element_velocity
        radius_contacts = radius
        omega_collection_contacts = omega_collection
        director_collection_contacts = director_collection

    tangent_along_normal_direction = _batch_dot(normals_on_elements, tangents_contacts)
    tangent_perpendicular_to_normal_direction = (
        tangents_contacts
        - _batch_product_k_ik_to_ik(tangent_along_normal_direction, normals_on_elements)
    )

    tangent_perpendicular_to_normal_direction_mag = _batch_norm(
        tangent_perpendicular_to_normal_direction
    )

    # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
    # small tolerance (1e-10) for normalization, in order to prevent division by 0.
    axial_direction = _batch_product_k_ik_to_ik(
        1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
        tangent_perpendicular_to_normal_direction,
    )

    # first apply axial kinetic friction
    velocity_mag_along_axial_direction = _batch_dot(
        element_velocity_contacts, axial_direction
    )
    velocity_along_axial_direction = _batch_product_k_ik_to_ik(
        velocity_mag_along_axial_direction, axial_direction
    )

    # Friction forces depends on the direction of velocity, in other words sign
    # of the velocity vector.
    velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
    # Check top for sign convention
    kinetic_mu = 0.5 * (
        kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
        + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
    )
    # Call slip function to check if elements slipping or not
    slip_function_along_axial_direction = _find_slipping_elements(
        velocity_along_axial_direction, slip_velocity_tol
    )

    # Now rolling kinetic friction
    rolling_direction = _batch_cross(axial_direction, normals_on_elements)
    torque_arm = -_batch_product_k_ik_to_ik(radius_contacts, normals_on_elements)
    velocity_along_rolling_direction = _batch_dot(
        element_velocity_contacts, rolling_direction
    )
    velocity_sign_along_rolling_direction = np.sign(velocity_along_rolling_direction)

    directors_transpose_contacts = _batch_matrix_transpose(director_collection_contacts)
    directors_transpose = _batch_matrix_transpose(director_collection)

    # w_rot = Q.T @ omega @ Q @ r
    rotation_velocity = _batch_matvec(
        directors_transpose_contacts,
        _batch_cross(
            omega_collection_contacts,
            _batch_matvec(director_collection_contacts, torque_arm),
        ),
    )
    rotation_velocity_along_rolling_direction = _batch_dot(
        rotation_velocity, rolling_direction
    )
    slip_velocity_mag_along_rolling_direction = (
        velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
    )
    slip_velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
        slip_velocity_mag_along_rolling_direction, rolling_direction
    )
    slip_function_along_rolling_direction = _find_slipping_elements(
        slip_velocity_along_rolling_direction, slip_velocity_tol
    )
    # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
    # and rolling directions.
    unitized_total_velocity = (
        slip_velocity_along_rolling_direction + velocity_along_axial_direction
    )
    unitized_total_velocity /= _batch_norm(unitized_total_velocity + 1e-14)
    # Apply kinetic friction in axial direction.
    kinetic_friction_force_along_axial_direction_contacts = -(
        (1.0 - slip_function_along_axial_direction)
        * kinetic_mu
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, axial_direction)
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_axial_direction_contacts[
        ..., no_contact_point_idx
    ] = 0.0

    # Apply kinetic friction in rolling direction.
    kinetic_friction_force_along_rolling_direction_contacts = -(
        (1.0 - slip_function_along_rolling_direction)
        * kinetic_mu_sideways
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, rolling_direction)
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_rolling_direction_contacts[
        ..., no_contact_point_idx
    ] = 0.0

    # torque = Q @ r @ Fr
    kinetic_rolling_torque_contacts = _batch_matvec(
        director_collection_contacts,
        _batch_cross(
            torque_arm, kinetic_friction_force_along_rolling_direction_contacts
        ),
    )

    # now axial static friction

    # check top for sign convention
    static_mu = 0.5 * (
        static_mu_forward * (1 + velocity_sign_along_axial_direction)
        + static_mu_backward * (1 - velocity_sign_along_axial_direction)
    )
    max_friction_force = (
        slip_function_along_axial_direction * static_mu * plane_response_force_mag
    )
    # friction = min(mu N, gamma v)
    static_friction_force_along_axial_direction_contacts = -(
        np.minimum(np.fabs(gamma * velocity_along_axial_direction), max_friction_force)
        * velocity_sign_along_axial_direction
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
    static_friction_force_along_axial_direction_contacts[
        ..., no_contact_point_idx
    ] = 0.0

    # now rolling static friction
    # there is some normal, tangent and rolling directions inconsitency from Elastica
    # total_torques = _batch_matvec(directors_transpose, (internal_torques + external_torques))[:,position_idx_array]

    # Elastica has opposite defs of tangents in interaction.h and rod.cpp
    # total_torques_along_axial_direction = _batch_dot(total_torques, axial_direction)

    # noslip_force = -(
    #     (
    #         radius[position_idx_array] * force_component_along_rolling_direction
    #         - 2.0 * total_torques_along_axial_direction
    #     )
    #     / 3.0
    #     / radius[position_idx_array]
    # )

    max_friction_force = (
        slip_function_along_rolling_direction
        * static_mu_sideways
        * plane_response_force_mag
    )
    # noslip_force_sign = np.sign(noslip_force)

    static_friction_force_along_rolling_direction_contacts = (
        np.minimum(
            np.fabs(gamma * velocity_along_rolling_direction), max_friction_force
        )
        * velocity_sign_along_rolling_direction
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
    static_friction_force_along_rolling_direction_contacts[
        ..., no_contact_point_idx
    ] = 0.0

    static_rolling_torque_contacts = _batch_matvec(
        director_collection_contacts,
        _batch_cross(
            torque_arm, static_friction_force_along_rolling_direction_contacts
        ),
    )

    # add up contribution from all contacts

    kinetic_friction_force_along_axial_direction = np.zeros_like(element_velocity)
    kinetic_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
    kinetic_rolling_torque = np.zeros_like(element_velocity)
    static_friction_force_along_axial_direction = np.zeros_like(element_velocity)
    static_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
    static_rolling_torque = np.zeros_like(element_velocity)

    for i in range(len(position_idx_array)):
        kinetic_friction_force_along_axial_direction[
            :, position_idx_array[i]
        ] += kinetic_friction_force_along_axial_direction_contacts[:, i]
        kinetic_friction_force_along_rolling_direction[
            :, position_idx_array[i]
        ] += kinetic_friction_force_along_rolling_direction_contacts[:, i]
        kinetic_rolling_torque[
            :, position_idx_array[i]
        ] += kinetic_rolling_torque_contacts[:, i]
        static_friction_force_along_axial_direction[
            :, position_idx_array[i]
        ] += static_friction_force_along_axial_direction_contacts[:, i]
        static_friction_force_along_rolling_direction[
            :, position_idx_array[i]
        ] += static_friction_force_along_rolling_direction_contacts[:, i]
        static_rolling_torque[
            :, position_idx_array[i]
        ] += static_rolling_torque_contacts[:, i]

    # apply all forces and torques
    _elements_to_nodes_inplace(
        kinetic_friction_force_along_axial_direction, external_forces
    )
    _elements_to_nodes_inplace(
        kinetic_friction_force_along_rolling_direction, external_forces
    )
    external_torques += kinetic_rolling_torque
    _elements_to_nodes_inplace(
        static_friction_force_along_axial_direction, external_forces
    )
    _elements_to_nodes_inplace(
        static_friction_force_along_rolling_direction, external_forces
    )
    external_torques += static_rolling_torque


############################################# Multiple contacts search

# class CustomInteractionSurface:
#     """
#     The interaction plane class computes the surface reaction
#     force on a rod-like object.

#         Attributes
#         ----------
#         k: float
#             Stiffness coefficient between the surface and the rod-like object.
#         nu: float
#             Dissipation coefficient between the surface and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet.
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         surface_tol: float
#             Penetration tolerance between the surface and the rod-like object.

#     """

#     def __init__(self, k, nu, facets, up_direction):
#         """

#         Parameters
#         ----------
#         k: float
#             Stiffness coefficient between the plane and the rod-like object.
#         nu: float
#             Dissipation coefficient between the plane and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet (axis,vertix,facet).
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         surface_tol: float
#             Penetration tolerance between the surface and the rod-like object.

#         """

#         self.k = k
#         self.nu = nu
#         self.facets = facets
#         self_up_direction = up_direction
#         self.facets_normals,self.facets_centers = calculate_facet_normals_centers(self.facets,self_up_direction)
#         self.surface_tol = 1e-4

#     def apply_normal_force(self, system):
#         """
#         In the case of contact with the plane, this function computes the plane reaction force on the element.

#         Parameters
#         ----------
#         system: object
#             Rod-like object.

#         Returns
#         -------
#         plane_response_force_mag : numpy.ndarray
#             1D (blocksize) array containing data with 'float' type.
#             Magnitude of plane response force acting on rod-like object.
#         no_contact_point_idx : numpy.ndarray
#             1D (blocksize) array containing data with 'int' type.
#             Index of rod-like object elements that are not in contact with the plane.
#         """
#         return apply_normal_force_numba(
#             self.facets_normals,
#             self.facets_centers,
#             self.surface_tol,
#             self.k,
#             self.nu,
#             system.radius,
#             system.mass,
#             system.position_collection,
#             system.velocity_collection,
#             system.internal_forces,
#             system.external_forces,
#             system.lengths,
#         )


# #@njit(cache=True)
# def apply_normal_force_numba(
#     facets_normals,
#     facets_centers,
#     surface_tol,
#     k,
#     nu,
#     radius,
#     mass,
#     position_collection,
#     velocity_collection,
#     internal_forces,
#     external_forces,
#     lengths,
# ):
#     """
#     This function computes the plane force response on the element, in the
#     case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
#     is used.

#     Parameters
#     ----------
#     system

#     Returns
#     -------
#     magnitude of the plane response
#     """


#     #find center of mass
#     element_position = node_to_element_position(position_collection)
#     mass_times_position = _batch_product_k_ik_to_ik(mass, position_collection)
#     sum_mass_times_position = np.sum(mass_times_position,axis=1)
#     center_of_mass = (sum_mass_times_position / mass.sum())
#     neighborhood_size = np.sum(lengths)/2


#     #find facet neighborhood
#     neighborhood_idx = find_facet_neighborhood(facets_centers,center_of_mass,neighborhood_size)

#     # #if no facets are close to the neighborhood, then expand neighborhood_size
#     # while ~np.any(neighborhood_idx):
#     #     neighborhood_size *= 1.1
#     #     neighborhood_idx = find_facet_neighborhood(facets_centers,center_of_mass,neighborhood_size)

#     neighborhood =  facets_centers[:,neighborhood_idx]
#     neighborhood_normals = facets_normals[:,neighborhood_idx]


#     n_element = element_position.shape[-1]
#     position_idx = []
#     facet_idx = []

#     #find facet neighborhood of each element position
#     for i in range(n_element):
#         neighborhood_idx = find_facet_neighborhood(neighborhood,element_position[:,i],1.1*radius[i])
#         facet_idx += list(neighborhood_idx)
#         for j in range(len(neighborhood_idx)):
#             position_idx += [i]


#     facet_idx_array = facet_idx
#     position_idx_array = position_idx
#     element_position_contacts = element_position[:,position_idx_array]

#     if len(facet_idx_array)>0:
#         contact_facet_centers =  neighborhood[:,facet_idx_array]
#         normals_on_elements = neighborhood_normals[:,facet_idx_array]
#     else:
#         contact_facet_centers = np.zeros_like(element_position_contacts)
#         normals_on_elements = np.zeros_like(element_position_contacts)


#     # Elastic force response due to penetration

#     distance_from_plane = _batch_dot(normals_on_elements,(element_position_contacts - contact_facet_centers))
#     plane_penetration = -np.abs(np.minimum(distance_from_plane - radius[position_idx_array], 0.0))**1.5
#     elastic_force = -k * _batch_product_k_ik_to_ik(plane_penetration,normals_on_elements)

#     # Damping force response due to velocity towards the plane
#     element_velocity = node_to_element_velocity(
#         mass=mass, node_velocity_collection=velocity_collection
#     )

#     element_velocity_contacts = element_velocity[:,position_idx_array]
#     normal_component_of_element_velocity = _batch_dot(normals_on_elements,element_velocity_contacts)
#     damping_force = -nu * _batch_product_k_ik_to_ik(normal_component_of_element_velocity,normals_on_elements)

#     # Compute total plane response force
#     plane_response_force_contacts = elastic_force + damping_force

#     # Check if the rod elements are in contact with plane.
#     no_contact_point_idx = np.where((distance_from_plane - radius[position_idx_array]) > surface_tol)[0]
#     # If rod element does not have any contact with plane, plane cannot apply response
#     # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
#     plane_response_force_contacts[..., no_contact_point_idx] = 0.0


#     plane_response_forces = np.zeros_like(external_forces)
#     for i in range(len(position_idx_array)):
#         plane_response_forces[:,position_idx_array[i]] += plane_response_force_contacts[:,i]


#     # Update the external forces
#     elements_to_nodes_inplace(plane_response_forces, external_forces)

#     return (_batch_norm(plane_response_force_contacts), no_contact_point_idx,normals_on_elements,position_idx_array)


# class CustomFrictionalSurface(NoForces, CustomInteractionSurface):


#     def __init__(
#         self,
#         k,
#         nu,
#         facets,
#         up_direction,
#         slip_velocity_tol,
#         static_mu_array,
#         kinetic_mu_array,
#         gamma,
#     ):
#         """

#         Parameters
#         ----------
#         k: float
#             Stiffness coefficient between the plane and the rod-like object.
#         nu: float
#             Dissipation coefficient between the plane and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet (axis,vertix,facet).
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         slip_velocity_tol: float
#             Velocity tolerance to determine if the element is slipping or not.
#         static_mu_array: numpy.ndarray
#             1D (3,) array containing data with 'float' type.
#             [forward, backward, sideways] static friction coefficients.
#         kinetic_mu_array: numpy.ndarray
#             1D (3,) array containing data with 'float' type.
#             [forward, backward, sideways] kinetic friction coefficients.
#         """
#         CustomInteractionSurface.__init__(self, k, nu, facets, up_direction)
#         self.slip_velocity_tol = slip_velocity_tol
#         (
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#         ) = static_mu_array
#         (
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#         ) = kinetic_mu_array
#         self.gamma = gamma

#     # kinetic and static friction should separate functions
#     # for now putting them together to figure out common variables
#     def apply_forces(self, system, time=0.0):
#         """
#         Call numba implementation to apply friction forces
#         Parameters
#         ----------
#         system
#         time

#         Returns
#         -------

#         """
#         anisotropic_friction(
#             self.facets_centers,
#             self.facets_normals,
#             self.surface_tol,
#             self.slip_velocity_tol,
#             self.k,
#             self.nu,
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#             self.gamma,
#             system.radius,
#             system.mass,
#             system.tangents,
#             system.position_collection,
#             system.director_collection,
#             system.velocity_collection,
#             system.omega_collection,
#             system.internal_forces,
#             system.external_forces,
#             system.internal_torques,
#             system.external_torques,
#             system.lengths,
#         )


# #@njit(cache=True)
# def anisotropic_friction(
#     facets_centers,
#     facets_normals,
#     surface_tol,
#     slip_velocity_tol,
#     k,
#     nu,
#     kinetic_mu_forward,
#     kinetic_mu_backward,
#     kinetic_mu_sideways,
#     gamma,
#     static_mu_forward,
#     static_mu_backward,
#     static_mu_sideways,
#     radius,
#     mass,
#     tangents,
#     position_collection,
#     director_collection,
#     velocity_collection,
#     omega_collection,
#     internal_forces,
#     external_forces,
#     internal_torques,
#     external_torques,
#     lengths,
# ):
#     plane_response_force_mag, no_contact_point_idx,normals_on_elements,position_idx_array = apply_normal_force_numba(
#         facets_normals,
#         facets_centers,
#         surface_tol,
#         k,
#         nu,
#         radius,
#         mass,
#         position_collection,
#         velocity_collection,
#         internal_forces,
#         external_forces,
#         lengths,
#     )
#     # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
#     # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
#     # to the plane. So friction forces can only be in plane forces and not out of plane.

#     tangents_contacts = tangents[:,position_idx_array]
#     tangent_along_normal_direction = _batch_dot(normals_on_elements,tangents_contacts)
#     tangent_perpendicular_to_normal_direction = tangents_contacts -  _batch_product_k_ik_to_ik(tangent_along_normal_direction,normals_on_elements)

#     tangent_perpendicular_to_normal_direction_mag = _batch_norm(
#     tangent_perpendicular_to_normal_direction
#     )

#     # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
#     # small tolerance (1e-10) for normalization, in order to prevent division by 0.
#     axial_direction = _batch_product_k_ik_to_ik(
#         1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
#         tangent_perpendicular_to_normal_direction,
#     )

#     element_velocity = node_to_element_velocity(
#         mass=mass, node_velocity_collection=velocity_collection
#     )

#     element_velocity_contacts = element_velocity[:,position_idx_array]

#     # first apply axial kinetic friction
#     velocity_mag_along_axial_direction = _batch_dot(element_velocity_contacts, axial_direction)
#     velocity_along_axial_direction = _batch_product_k_ik_to_ik(
#         velocity_mag_along_axial_direction, axial_direction
#     )

#     # Friction forces depends on the direction of velocity, in other words sign
#     # of the velocity vector.
#     velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
#     # Check top for sign convention
#     kinetic_mu = 0.5 * (
#         kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
#         + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
#     )
#     # Call slip function to check if elements slipping or not
#     slip_function_along_axial_direction = find_slipping_elements(
#         velocity_along_axial_direction, slip_velocity_tol
#     )

#     # Now rolling kinetic friction
#     rolling_direction = _batch_cross(axial_direction, normals_on_elements)
#     torque_arm = -_batch_product_k_ik_to_ik(radius[position_idx_array],normals_on_elements)
#     velocity_along_rolling_direction = _batch_dot(element_velocity_contacts, rolling_direction)
#     velocity_sign_along_rolling_direction = np.sign(velocity_along_rolling_direction)

#     director_collection_contacts = director_collection[:,:,position_idx_array]
#     directors_transpose_contacts = _batch_matrix_transpose(director_collection_contacts)
#     directors_transpose = _batch_matrix_transpose(director_collection)

#     # w_rot = Q.T @ omega @ Q @ r
#     rotation_velocity = _batch_matvec(
#         directors_transpose_contacts,
#         _batch_cross(omega_collection[:,position_idx_array], _batch_matvec(director_collection_contacts, torque_arm)),
#     )
#     rotation_velocity_along_rolling_direction = _batch_dot(
#         rotation_velocity, rolling_direction
#     )
#     slip_velocity_mag_along_rolling_direction = (
#         velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
#     )
#     slip_velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
#         slip_velocity_mag_along_rolling_direction, rolling_direction
#     )
#     slip_function_along_rolling_direction = find_slipping_elements(
#         slip_velocity_along_rolling_direction, slip_velocity_tol
#     )
#     # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
#     # and rolling directions.
#     unitized_total_velocity = (
#         slip_velocity_along_rolling_direction + velocity_along_axial_direction
#     )
#     unitized_total_velocity /= _batch_norm(unitized_total_velocity + 1e-14)
#     # Apply kinetic friction in axial direction.
#     kinetic_friction_force_along_axial_direction_contacts = -(
#         (1.0 - slip_function_along_axial_direction)
#         * kinetic_mu
#         * plane_response_force_mag
#         * _batch_dot(unitized_total_velocity, axial_direction)
#         * axial_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     kinetic_friction_force_along_axial_direction_contacts[..., no_contact_point_idx] = 0.0


#     # Apply kinetic friction in rolling direction.
#     kinetic_friction_force_along_rolling_direction_contacts = -(
#         (1.0 - slip_function_along_rolling_direction)
#         * kinetic_mu_sideways
#         * plane_response_force_mag
#         * _batch_dot(unitized_total_velocity, rolling_direction)
#         * rolling_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     kinetic_friction_force_along_rolling_direction_contacts[..., no_contact_point_idx] = 0.0


#     # torque = Q @ r @ Fr
#     kinetic_rolling_torque_contacts = _batch_matvec(director_collection_contacts,_batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction_contacts),)


#     # now axial static friction

#     # check top for sign convention
#     static_mu = 0.5 * (
#         static_mu_forward * (1 + velocity_sign_along_axial_direction)
#         + static_mu_backward * (1 - velocity_sign_along_axial_direction)
#     )
#     max_friction_force = (
#         slip_function_along_axial_direction * static_mu * plane_response_force_mag
#     )
#     # friction = min(mu N, gamma v)
#     static_friction_force_along_axial_direction_contacts = -(
#         np.minimum(np.fabs(gamma*velocity_along_axial_direction), max_friction_force)
#         * velocity_sign_along_axial_direction
#         * axial_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
#     static_friction_force_along_axial_direction_contacts[..., no_contact_point_idx] = 0.0


#     # now rolling static friction
#     # there is some normal, tangent and rolling directions inconsitency from Elastica
#     # total_torques = _batch_matvec(directors_transpose, (internal_torques + external_torques))[:,position_idx_array]

#     # Elastica has opposite defs of tangents in interaction.h and rod.cpp
#     # total_torques_along_axial_direction = _batch_dot(total_torques, axial_direction)

#     # noslip_force = -(
#     #     (
#     #         radius[position_idx_array] * force_component_along_rolling_direction
#     #         - 2.0 * total_torques_along_axial_direction
#     #     )
#     #     / 3.0
#     #     / radius[position_idx_array]
#     # )

#     max_friction_force = (
#         slip_function_along_rolling_direction
#         * static_mu_sideways
#         * plane_response_force_mag
#     )
#     # noslip_force_sign = np.sign(noslip_force)

#     static_friction_force_along_rolling_direction_contacts = (
#         np.minimum(np.fabs(gamma*velocity_along_rolling_direction), max_friction_force)
#         * velocity_sign_along_rolling_direction
#         * rolling_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
#     static_friction_force_along_rolling_direction_contacts[..., no_contact_point_idx] = 0.0

#     static_rolling_torque_contacts = _batch_matvec(
#         director_collection_contacts,
#         _batch_cross(torque_arm, static_friction_force_along_rolling_direction_contacts),
#     )

#     #add up contribution from all contacts

#     kinetic_friction_force_along_axial_direction = np.zeros_like(element_velocity)
#     kinetic_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
#     kinetic_rolling_torque = np.zeros_like(element_velocity)
#     static_friction_force_along_axial_direction = np.zeros_like(element_velocity)
#     static_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
#     static_rolling_torque = np.zeros_like(element_velocity)

#     for i in range(len(position_idx_array)):
#         kinetic_friction_force_along_axial_direction[:,position_idx_array[i]] += kinetic_friction_force_along_axial_direction_contacts[:,i]
#         kinetic_friction_force_along_rolling_direction[:,position_idx_array[i]] += kinetic_friction_force_along_rolling_direction_contacts[:,i]
#         kinetic_rolling_torque[:,position_idx_array[i]] += kinetic_rolling_torque_contacts[:,i]
#         static_friction_force_along_axial_direction[:,position_idx_array[i]] += static_friction_force_along_axial_direction_contacts[:,i]
#         static_friction_force_along_rolling_direction[:,position_idx_array[i]] += static_friction_force_along_rolling_direction_contacts[:,i]
#         static_rolling_torque[:,position_idx_array[i]] += static_rolling_torque_contacts[:,i]


#     #apply all forces and torques
#     elements_to_nodes_inplace(
#         kinetic_friction_force_along_axial_direction, external_forces
#     )
#     elements_to_nodes_inplace(
#         kinetic_friction_force_along_rolling_direction, external_forces
#     )
#     external_torques += kinetic_rolling_torque
#     elements_to_nodes_inplace(
#         static_friction_force_along_axial_direction, external_forces
#     )
#     elements_to_nodes_inplace(
#         static_friction_force_along_rolling_direction, external_forces
#     )
#     external_torques += static_rolling_torque


############################################## Single contact Grid


# def find_contact_facets_idx_single(
#     facets_grid,
#     x_min,
#     y_min,
#     grid_size,
#     position_collection,
#     ):

#     element_position = _node_to_element_position(position_collection)
#     n_element = element_position.shape[-1]
#     facet_idx_array = np.array([0])
#     ## Note: Boundary issue needs to be fixed
#     grid_position = np.round((element_position[0:2,:]-np.array([x_min,y_min]).reshape((2,1)))/grid_size)

#     #find facet neighborhood of each element position
#     for i in range(n_element):
#         try:
#             facet_idx_1 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]))] #first quadrant
#         except:
#             facet_idx_1 = np.empty((0))
#         try:
#             facet_idx_2 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]))] #second quadrant
#         except:
#             facet_idx_2 = np.empty((0))
#         try:
#             facet_idx_3 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]-1))] #third quadrant
#         except:
#             facet_idx_3 = np.empty((0))
#         try:
#             facet_idx_4 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]-1))] #fourth quadrant
#         except:
#             facet_idx_4 = np.empty((0))
#         facet_idx_element = np.concatenate((facet_idx_1,facet_idx_2,facet_idx_3,facet_idx_4))
#         facet_idx_element_no_duplicates = np.unique(facet_idx_element)
#         facet_idx_array = np.concatenate((facet_idx_array,facet_idx_element_no_duplicates))

#     facet_idx_array = facet_idx_array.astype(int)

#     return facet_idx_array,element_position


# class CustomInteractionSurfaceGrid:
#     """
#     The interaction plane class computes the surface reaction
#     force on a rod-like object.

#         Attributes
#         ----------
#         k: float
#             Stiffness coefficient between the surface and the rod-like object.
#         nu: float
#             Dissipation coefficient between the surface and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet.
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         surface_tol: float
#             Penetration tolerance between the surface and the rod-like object.

#     """

#     def __init__(self, k, nu, facets, facets_grid, grid_size, up_direction):
#         """

#         Parameters
#         ----------
#         k: float
#             Stiffness coefficient between the plane and the rod-like object.
#         nu: float
#             Dissipation coefficient between the plane and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet (axis,vertix,facet).
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         surface_tol: float
#             Penetration tolerance between the surface and the rod-like object.

#         """

#         self.k = k
#         self.nu = nu
#         self.facets = facets
#         self.x_min = np.min(facets[0,:,:])
#         self.y_min = np.min(facets[1,:,:])
#         self.facets_grid = facets_grid
#         self.up_direction = up_direction
#         self.facets_normals,self.facets_centers = calculate_facet_normals_centers(self.facets,self.up_direction)
#         self.surface_tol = 1e-4
#         self.grid_size = grid_size

#     def apply_normal_force(self, system):
#         """
#         In the case of contact with the plane, this function computes the plane reaction force on the element.

#         Parameters
#         ----------
#         system: object
#             Rod-like object.

#         Returns
#         -------
#         plane_response_force_mag : numpy.ndarray
#             1D (blocksize) array containing data with 'float' type.
#             Magnitude of plane response force acting on rod-like object.
#         no_contact_point_idx : numpy.ndarray
#             1D (blocksize) array containing data with 'int' type.
#             Index of rod-like object elements that are not in contact with the plane.
#         """

#         self.facet_idx_array,self.element_position = find_contact_facets_idx_single(
#         self.facets_grid,
#         self.x_min,
#         self.y_min,
#         self.grid_size,
#         system.position_collection,
#         )

#         return apply_normal_force_numba(
#             self.facets_normals,
#             self.facets_centers,
#             self.element_position,
#             self.facet_idx_array,
#             self.up_direction,
#             self.surface_tol,
#             self.k,
#             self.nu,
#             system.radius,
#             system.mass,
#             system.position_collection,
#             system.velocity_collection,
#             system.internal_forces,
#             system.external_forces,
#             system.lengths,
#         )


# @njit(cache=True)
# def apply_normal_force_numba(
#     facets_normals,
#     facets_centers,
#     element_position,
#     facet_idx_array,
#     up_direction,
#     surface_tol,
#     k,
#     nu,
#     radius,
#     mass,
#     position_collection,
#     velocity_collection,
#     internal_forces,
#     external_forces,
#     lengths,
# ):
#     """
#     This function computes the plane force response on the element, in the
#     case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
#     is used.

#     Parameters
#     ----------
#     system

#     Returns
#     -------
#     magnitude of the plane response
#     """


#     normals_on_elements = np.zeros_like(element_position)
#     closest_facets_to_elements = np.zeros_like(element_position)
#     neighborhood = facets_centers[:,facet_idx_array]
#     neighborhood_normals = facets_normals[:,facet_idx_array]


#     #find which facet each element of the rod lays on
#     closest_facets_to_elements_indices = find_closest_facets(neighborhood,element_position) #gives the index of the closest facet to each element
#     for i in range(element_position.shape[-1]):
#         normals_on_elements[...,i] = neighborhood_normals[...,int(closest_facets_to_elements_indices[i])]
#         closest_facets_to_elements[...,i] = neighborhood[...,int(closest_facets_to_elements_indices[i])]


#     # Elastic force response due to penetration

#     distance_from_plane = _batch_dot(normals_on_elements,(element_position - closest_facets_to_elements))
#     plane_penetration = -np.abs(np.minimum(distance_from_plane - radius, 0.0))**1.5
#     elastic_force = -k * _batch_product_k_ik_to_ik(plane_penetration,normals_on_elements)

#     # Damping force response due to velocity towards the plane
#     element_velocity = node_to_element_velocity(
#         mass=mass, node_velocity_collection=velocity_collection
#     )

#     normal_component_of_element_velocity = _batch_dot(normals_on_elements,element_velocity)
#     damping_force = -nu * _batch_product_k_ik_to_ik(normal_component_of_element_velocity,normals_on_elements)

#     # Compute total plane response force
#     plane_response_forces = elastic_force + damping_force

#     # Check if the rod elements are in contact with plane.
#     no_contact_point_idx = np.where((distance_from_plane - radius) > surface_tol)[0]
#     # If rod element does not have any contact with plane, plane cannot apply response
#     # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
#     plane_response_forces[..., no_contact_point_idx] = 0.0
#     # Update the external forces
#     elements_to_nodes_inplace(plane_response_forces, external_forces)

#     return (_batch_norm(plane_response_forces), no_contact_point_idx,normals_on_elements)


# class CustomFrictionalSurfaceGrid(NoForces, CustomInteractionSurfaceGrid):


#     def __init__(
#         self,
#         k,
#         nu,
#         facets,
#         facets_grid,
#         grid_size,
#         up_direction,
#         slip_velocity_tol,
#         static_mu_array,
#         kinetic_mu_array,
#         gamma,
#     ):
#         """

#         Parameters
#         ----------
#         k: float
#             Stiffness coefficient between the plane and the rod-like object.
#         nu: float
#             Dissipation coefficient between the plane and the rod-like object.
#         facets: numpy.ndarray
#             (3,3,dim) array containing data with 'float' type.
#             The three vertices of each facet (axis,vertix,facet).
#         up_direction: numpy.ndarray
#             (3,1) array containing data with 'float' type.
#             The up direction of the surface.
#         slip_velocity_tol: float
#             Velocity tolerance to determine if the element is slipping or not.
#         static_mu_array: numpy.ndarray
#             1D (3,) array containing data with 'float' type.
#             [forward, backward, sideways] static friction coefficients.
#         kinetic_mu_array: numpy.ndarray
#             1D (3,) array containing data with 'float' type.
#             [forward, backward, sideways] kinetic friction coefficients.
#         """
#         CustomInteractionSurfaceGrid.__init__(self, k, nu, facets, facets_grid,grid_size, up_direction)
#         self.slip_velocity_tol = slip_velocity_tol
#         (
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#         ) = static_mu_array
#         (
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#         ) = kinetic_mu_array
#         self.gamma = gamma

#     # kinetic and static friction should separate functions
#     # for now putting them together to figure out common variables
#     def apply_forces(self, system, time=0.0):
#         """
#         Call numba implementation to apply friction forces
#         Parameters
#         ----------
#         system
#         time

#         Returns
#         -------

#         """

#         self.facet_idx_array,self.element_position = find_contact_facets_idx_single(
#         self.facets_grid,
#         self.x_min,
#         self.y_min,
#         self.grid_size,
#         system.position_collection,
#         )

#         anisotropic_friction(
#             self.facets_centers,
#             self.facets_normals,
#             self.element_position,
#             self.facet_idx_array,
#             self.up_direction,
#             self.surface_tol,
#             self.slip_velocity_tol,
#             self.k,
#             self.nu,
#             self.kinetic_mu_forward,
#             self.kinetic_mu_backward,
#             self.kinetic_mu_sideways,
#             self.static_mu_forward,
#             self.static_mu_backward,
#             self.static_mu_sideways,
#             self.gamma,
#             system.radius,
#             system.mass,
#             system.tangents,
#             system.position_collection,
#             system.director_collection,
#             system.velocity_collection,
#             system.omega_collection,
#             system.internal_forces,
#             system.external_forces,
#             system.internal_torques,
#             system.external_torques,
#             system.lengths,
#         )


# @njit(cache=True)
# def anisotropic_friction(
#     facets_centers,
#     facets_normals,
#     element_position,
#     facet_idx_array,
#     up_direction,
#     surface_tol,
#     slip_velocity_tol,
#     k,
#     nu,
#     kinetic_mu_forward,
#     kinetic_mu_backward,
#     kinetic_mu_sideways,
#     gamma,
#     static_mu_forward,
#     static_mu_backward,
#     static_mu_sideways,
#     radius,
#     mass,
#     tangents,
#     position_collection,
#     director_collection,
#     velocity_collection,
#     omega_collection,
#     internal_forces,
#     external_forces,
#     internal_torques,
#     external_torques,
#     lengths,
# ):
#     plane_response_force_mag, no_contact_point_idx,normals_on_elements = apply_normal_force_numba(
#         facets_normals,
#         facets_centers,
#         element_position,
#         facet_idx_array,
#         up_direction,
#         surface_tol,
#         k,
#         nu,
#         radius,
#         mass,
#         position_collection,
#         velocity_collection,
#         internal_forces,
#         external_forces,
#         lengths,
#     )
#     # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
#     # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
#     # to the plane. So friction forces can only be in plane forces and not out of plane.

#     tangent_along_normal_direction = _batch_dot(normals_on_elements,tangents)
#     tangent_perpendicular_to_normal_direction = tangents -  _batch_product_k_ik_to_ik(tangent_along_normal_direction,normals_on_elements)

#     tangent_perpendicular_to_normal_direction_mag = _batch_norm(
#     tangent_perpendicular_to_normal_direction
#     )

#     # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
#     # small tolerance (1e-10) for normalization, in order to prevent division by 0.
#     axial_direction = _batch_product_k_ik_to_ik(
#         1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
#         tangent_perpendicular_to_normal_direction,
#     )

#     element_velocity = node_to_element_velocity(
#         mass=mass, node_velocity_collection=velocity_collection
#     )


#     # first apply axial kinetic friction
#     velocity_mag_along_axial_direction = _batch_dot(element_velocity, axial_direction)
#     velocity_along_axial_direction = _batch_product_k_ik_to_ik(
#         velocity_mag_along_axial_direction, axial_direction
#     )

#     # Friction forces depends on the direction of velocity, in other words sign
#     # of the velocity vector.
#     velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
#     # Check top for sign convention
#     kinetic_mu = 0.5 * (
#         kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
#         + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
#     )
#     # Call slip function to check if elements slipping or not
#     slip_function_along_axial_direction = find_slipping_elements(
#         velocity_along_axial_direction, slip_velocity_tol
#     )

#     # Now rolling kinetic friction
#     rolling_direction = _batch_cross(axial_direction, normals_on_elements)
#     torque_arm = -_batch_product_k_ik_to_ik(radius,normals_on_elements)
#     velocity_along_rolling_direction = _batch_dot(element_velocity, rolling_direction)
#     velocity_sign_along_rolling_direction = np.sign(velocity_along_rolling_direction)

#     directors_transpose = _batch_matrix_transpose(director_collection)

#     # w_rot = Q.T @ omega @ Q @ r
#     rotation_velocity = _batch_matvec(
#         directors_transpose,
#         _batch_cross(omega_collection, _batch_matvec(director_collection, torque_arm)),
#     )
#     rotation_velocity_along_rolling_direction = _batch_dot(
#         rotation_velocity, rolling_direction
#     )
#     slip_velocity_mag_along_rolling_direction = (
#         velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
#     )
#     slip_velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
#         slip_velocity_mag_along_rolling_direction, rolling_direction
#     )
#     slip_function_along_rolling_direction = find_slipping_elements(
#         slip_velocity_along_rolling_direction, slip_velocity_tol
#     )
#     # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
#     # and rolling directions.
#     unitized_total_velocity = (
#         slip_velocity_along_rolling_direction + velocity_along_axial_direction
#     )
#     unitized_total_velocity /= _batch_norm(unitized_total_velocity + 1e-14)
#     # Apply kinetic friction in axial direction.
#     kinetic_friction_force_along_axial_direction = -(
#         (1.0 - slip_function_along_axial_direction)
#         * kinetic_mu
#         * plane_response_force_mag
#         * _batch_dot(unitized_total_velocity, axial_direction)
#         * axial_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0


#     # Apply kinetic friction in rolling direction.
#     kinetic_friction_force_along_rolling_direction = -(
#         (1.0 - slip_function_along_rolling_direction)
#         * kinetic_mu_sideways
#         * plane_response_force_mag
#         * _batch_dot(unitized_total_velocity, rolling_direction)
#         * rolling_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
#     kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0


#     # torque = Q @ r @ Fr
#     kinetic_rolling_torque = _batch_matvec(director_collection,_batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),)


#     # now axial static friction

#     # check top for sign convention
#     static_mu = 0.5 * (
#         static_mu_forward * (1 + velocity_sign_along_axial_direction)
#         + static_mu_backward * (1 - velocity_sign_along_axial_direction)
#     )
#     max_friction_force = (
#         slip_function_along_axial_direction * static_mu * plane_response_force_mag
#     )
#     # friction = min(mu N, gamma v)
#     static_friction_force_along_axial_direction = -(
#         np.minimum(np.fabs(gamma*velocity_along_axial_direction), max_friction_force)
#         * velocity_sign_along_axial_direction
#         * axial_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
#     static_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0


#     # now rolling static friction
#     # there is some normal, tangent and rolling directions inconsitency from Elastica
#     # total_torques = _batch_matvec(directors_transpose, (internal_torques + external_torques))[:,position_idx_array]

#     # Elastica has opposite defs of tangents in interaction.h and rod.cpp
#     # total_torques_along_axial_direction = _batch_dot(total_torques, axial_direction)

#     # noslip_force = -(
#     #     (
#     #         radius[position_idx_array] * force_component_along_rolling_direction
#     #         - 2.0 * total_torques_along_axial_direction
#     #     )
#     #     / 3.0
#     #     / radius[position_idx_array]
#     # )

#     max_friction_force = (
#         slip_function_along_rolling_direction
#         * static_mu_sideways
#         * plane_response_force_mag
#     )
#     # noslip_force_sign = np.sign(noslip_force)

#     static_friction_force_along_rolling_direction = (
#         np.minimum(np.fabs(gamma*velocity_along_rolling_direction), max_friction_force)
#         * velocity_sign_along_rolling_direction
#         * rolling_direction
#     )
#     # If rod element does not have any contact with plane, plane cannot apply friction
#     # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
#     static_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0

#     static_rolling_torque = _batch_matvec(
#         director_collection,
#         _batch_cross(torque_arm, static_friction_force_along_rolling_direction),
#     )

#     #add up contribution from all contacts


#     #apply all forces and torques
#     elements_to_nodes_inplace(
#         kinetic_friction_force_along_axial_direction, external_forces
#     )
#     elements_to_nodes_inplace(
#         kinetic_friction_force_along_rolling_direction, external_forces
#     )
#     external_torques += kinetic_rolling_torque
#     elements_to_nodes_inplace(
#         static_friction_force_along_axial_direction, external_forces
#     )
#     elements_to_nodes_inplace(
#         static_friction_force_along_rolling_direction, external_forces
#     )
#     external_torques += static_rolling_torque
