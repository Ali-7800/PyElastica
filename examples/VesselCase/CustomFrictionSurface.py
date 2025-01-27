import numpy as np
from elastica.interaction import *
from elastica.external_forces import NoForces
from numba import njit
from elastica.interaction import (
    elements_to_nodes_inplace,
    find_slipping_elements,
    node_to_element_position,
    node_to_element_velocity,
)
from elastica._linalg import (
    _batch_matvec,
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_product_k_ik_to_ik,
    _batch_matrix_transpose,
)

# from surface_functions import (
#     calculate_facet_normals_centers_areas
# )

import sys


@njit(cache=True)
def find_facet_neighborhood(facets_centers, point, neighborhood_size):
    """
    Finds indices of facets with centers within a sphere of neighborhood_size around the point
    Parameters
    ----------
    facets_centers
    point
    neighborhood_size

    Returns
    -------
    neighborhood_idx

    Notes
    -----
    Not benchmarked

    """
    deltas = facets_centers - np.array([point[0], point[1], point[2]]).reshape((3, 1))
    distances = _batch_norm(deltas)
    neighborhood_idx = np.where(distances < neighborhood_size)[0]
    return neighborhood_idx


@njit(cache=True)
def find_closest_facets(facets_centers, positions):
    """
    Finds indices of the closest facet to each element
    ----------
    facets_centers
    positions

    Returns
    -------
    closest facet

    Notes
    -----
    Not benchmarked

    """
    n_positions = positions.shape[-1]
    closest_facets = np.empty((n_positions,))
    for i in range(n_positions):
        deltas = facets_centers - np.array(
            [positions[0, i], positions[1, i], positions[2, i]]
        ).reshape((3, 1))
        distances = _batch_norm(deltas)
        if distances.size != 0:
            closest_facets[i] = np.argmin(distances)
    return closest_facets


@njit(cache=True)
def find_closest_facet(facets_centers, position):
    """
    Finds indices of the closest facet to position
    ----------
    facets_centers
    positions

    Returns
    -------
    closest facet

    Notes
    -----
    Not benchmarked

    """
    deltas = facets_centers - np.array([position[0], position[1], position[2]]).reshape(
        (3, 1)
    )
    distances = _batch_norm(deltas)
    closest_facet = np.argmin(distances)
    return closest_facet


# @njit(cache=True)
def find_contact_facets_idx(
    facets_grid,
    x_min,
    y_min,
    grid_size,
    position_collection,
):

    element_position = node_to_element_position(position_collection)
    n_element = element_position.shape[-1]
    position_idx_array = np.empty((0))
    facet_idx_array = np.empty((0))
    grid_position = np.round(
        (element_position[0:2, :] - np.array([x_min, y_min]).reshape((2, 1)))
        / grid_size
    )
    # here we take the element position subtract the grid left most lower corner position (to get distance from that point)
    # The distance divided by the grid size converts the distance to units of grid size.
    # Rounding gives us the nearest corner to that element center position
    # Since any grid square can contain at most one element any element will have to lie within the four squares around the corner we found above.

    # find facet neighborhood of each element position #WHY is this always correct?
    for i in range(n_element):
        # try:
        #     facet_idx_1 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]))] #first quadrant
        # except:
        #     facet_idx_1 = np.empty((0))
        # try:
        #     facet_idx_2 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]))] #second quadrant
        # except:
        #     facet_idx_2 = np.empty((0))
        # try:
        #     facet_idx_3 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]-1))] #third quadrant
        # except:
        #     facet_idx_3 = np.empty((0))
        # try:
        #     facet_idx_4 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]-1))] #fourth quadrant
        # except:
        #     facet_idx_4 = np.empty((0))
        # facet_idx_element = np.concatenate((facet_idx_1,facet_idx_2,facet_idx_3,facet_idx_4))
        # facet_idx_element_no_duplicates = np.unique(facet_idx_element)
        try:
            facet_idx_element_no_duplicates = facets_grid[
                (int(grid_position[0, i]), int(grid_position[1, i]))
            ]
        except:
            facet_idx_element_no_duplicates = np.empty((0))
        # if facet_idx_element_no_duplicates.size == 0:
        #     raise BoundaryError("Snake outside surface boundary") #a snake element is on four grids with no facets

        facet_idx_array = np.concatenate(
            (facet_idx_array, facet_idx_element_no_duplicates)
        )
        n_contacts = facet_idx_element_no_duplicates.shape[0]
        position_idx_array = np.concatenate(
            (position_idx_array, i * np.ones((n_contacts,)))
        )

    position_idx_array = position_idx_array.astype(int)
    facet_idx_array = facet_idx_array.astype(int)
    return position_idx_array, facet_idx_array, element_position


############################################## Multiple contacts Grid


class CustomInteractionSurfaceGrid:
    """
    The interaction plane class computes the surface reaction
    force on a rod-like object.

        Attributes
        ----------
        k: float
            Stiffness coefficient between the surface and the rod-like object.
        nu: float
            Dissipation coefficient between the surface and the rod-like object.
        facets: numpy.ndarray
            (3,3,dim) array containing data with 'float' type.
            The three vertices of each facet.
        up_direction: numpy.ndarray
            (3,1) array containing data with 'float' type.
            The up direction of the surface.
        surface_tol: float
            Penetration tolerance between the surface and the rod-like object.

    """

    def __init__(
        self,
        k,
        nu,
        facets,
        facets_grid,
        facets_normals,
        facets_centers,
        side_vectors,
        grid_size,
    ):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        facets: numpy.ndarray
            (3,3,dim) array containing data with 'float' type.
            The three vertices of each facet (axis,vertix,facet).
        up_direction: numpy.ndarray
            (3,1) array containing data with 'float' type.
            The up direction of the surface.
        surface_tol: float
            Penetration tolerance between the surface and the rod-like object.

        """
        self.facets = facets
        self.n_facets = facets.shape[-1]

        # assert ("up_direction" in kwargs.keys() or "facet_vertex_normals" in kwargs.keys()),"Please provide valid up_direction or vertices_normals"
        # assert (~("up_direction" in kwargs.keys() and "facet_vertex_normals" in kwargs.keys())),"Please provide one of up_direction or vertices_normals not both"

        self.k = k
        self.nu = nu
        self.x_min = np.min(facets[0, :, :])
        self.y_min = np.min(facets[1, :, :])
        self.facets_grid = facets_grid

        # if "up_direction" in kwargs.keys():
        #     self.up_direction = kwargs["up_direction"]
        #     self.facets_normals,self.facets_centers = calculate_facet_normals_centers(facets = self.facets,up_direction = self.up_direction)
        # else:
        #     self.facet_vertex_normals = kwargs["facet_vertex_normals"]
        #     self.facets_normals,self.facets_centers = calculate_facet_normals_centers(facets = self.facets,facet_vertex_normals = self.facet_vertex_normals)
        self.facets_normals = facets_normals
        self.facets_centers = facets_centers
        self.side_vectors = side_vectors
        self.surface_tol = 1e-4
        self.grid_size = grid_size

    def apply_normal_force(self, system):
        """
        In the case of contact with the plane, this function computes the plane reaction force on the element.

        Parameters
        ----------
        system: object
            Rod-like object.

        Returns
        -------
        plane_response_force_mag : numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Magnitude of plane response force acting on rod-like object.
        no_contact_point_idx : numpy.ndarray
            1D (blocksize) array containing data with 'int' type.
            Index of rod-like object elements that are not in contact with the plane.
        """

        (
            self.position_idx_array,
            self.facet_idx_array,
            self.element_position,
        ) = find_contact_facets_idx(
            self.facets_grid,
            self.x_min,
            self.y_min,
            self.grid_size,
            system.position_collection,
        )

        return apply_normal_force_numba(
            self.facets,
            self.facets_normals,
            self.facets_centers,
            self.element_position,
            self.side_vectors,
            self.position_idx_array,
            self.facet_idx_array,
            self.surface_tol,
            self.k,
            self.nu,
            system.radius,
            system.mass,
            system.velocity_collection,
            system.external_forces,
        )


@njit(cache=True)
def apply_normal_force_numba(
    facets,
    facets_normals,
    facets_centers,
    element_position,
    side_vectors,
    position_idx_array,
    facet_idx_array,
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
    element_velocity = node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

    if len(facet_idx_array) > 0:
        element_position_contacts = element_position[:, position_idx_array]
        contact_facet_centers = facets_centers[:, facet_idx_array]
        normals_on_elements = facets_normals[:, facet_idx_array]
        radius_contacts = radius[position_idx_array]
        element_velocity_contacts = element_velocity[:, position_idx_array]
        contact_facets = facets[:, :, facet_idx_array]
        contact_facets_A = contact_facets[:, 0, :]
        contact_facets_B = contact_facets[:, 1, :]
        contact_facets_C = contact_facets[:, 2, :]
        contact_side_vectors = side_vectors[:, :, facet_idx_array]
        contact_side_AB = contact_side_vectors[:, 0, :]
        contact_side_AC = contact_side_vectors[:, 1, :]
        contact_side_BC = contact_side_vectors[:, 2, :]

    else:
        element_position_contacts = element_position
        contact_facet_centers = np.zeros_like(element_position)
        normals_on_elements = np.zeros_like(element_position)
        radius_contacts = radius
        element_velocity_contacts = element_velocity
        contact_facets = np.zeros_like(element_position)
        contact_side_vectors = np.zeros_like(element_position)
        contact_facets_A = np.zeros_like(element_position)
        contact_facets_B = np.zeros_like(element_position)
        contact_facets_C = np.zeros_like(element_position)
        contact_side_vectors = np.zeros_like(element_position)
        contact_side_AB = np.zeros_like(element_position)
        contact_side_AC = np.zeros_like(element_position)
        contact_side_BC = np.zeros_like(element_position)

    # Elastic force response due to penetration
    center_to_center_vector = element_position_contacts - contact_facet_centers
    distance_from_facet_plane = _batch_dot(normals_on_elements, center_to_center_vector)

    # intersection check (added to fix infinite plane problem)
    projected_element_position = (
        center_to_center_vector - distance_from_facet_plane * normals_on_elements
    )
    plane_intersection_radius = np.sqrt(
        np.maximum(radius_contacts ** 2 - distance_from_facet_plane ** 2, 0)
    )  # using pythagoras, figure out the intersection radius (we can modify this later for cylinder)
    closest_point_on_triangle_to_projected_point = projected_element_position

    projected_element_to_vertex_A = projected_element_position - contact_facets_A
    projected_element_to_vertex_B = projected_element_position - contact_facets_B
    projected_element_to_vertex_C = projected_element_position - contact_facets_C

    d1 = _batch_dot(projected_element_to_vertex_A, contact_side_AB)
    d2 = _batch_dot(projected_element_to_vertex_A, contact_side_AC)
    d3 = _batch_dot(projected_element_to_vertex_B, contact_side_AB)
    d4 = _batch_dot(projected_element_to_vertex_B, contact_side_AC)
    d5 = _batch_dot(projected_element_to_vertex_C, contact_side_AB)
    d6 = _batch_dot(projected_element_to_vertex_C, contact_side_AC)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    region_1 = np.where((d1 <= 0) * (d2 <= 0))[0]  # region 1 (closest to vertex A)
    closest_point_on_triangle_to_projected_point[:, region_1] = contact_facets_A[
        :, region_1
    ]
    region_2 = np.where((d3 >= 0) * (d4 <= d3))[0]  # region 2 (closest to vertex B)
    closest_point_on_triangle_to_projected_point[:, region_2] = contact_facets_B[
        :, region_2
    ]
    region_3 = np.where((d6 >= 0) * (d5 <= d6))[0]  # region 3 (closest to vertex C)
    closest_point_on_triangle_to_projected_point[:, region_3] = contact_facets_C[
        :, region_3
    ]
    region_4 = np.where((vc <= 0) * (d1 >= 0) * (d3 <= 0))[
        0
    ]  # region 4 (closest to some point on AB)
    closest_point_on_triangle_to_projected_point[:, region_4] = contact_facets_A[
        :, region_4
    ] + d1[region_4] * contact_side_AB[:, region_4] / (d1[region_4] - d3[region_4])
    region_5 = np.where((vb <= 0) * (d2 >= 0) * (d6 <= 0))[
        0
    ]  # region 5 (closest to some point on AC)
    closest_point_on_triangle_to_projected_point[:, region_5] = contact_facets_A[
        :, region_5
    ] + d2[region_5] * contact_side_AC[:, region_5] / (d2[region_5] - d6[region_5])
    region_6 = np.where((va <= 0) * (d4 >= d3) * (d5 >= d6))[
        0
    ]  # region 6 (closest to some point on BC)
    closest_point_on_triangle_to_projected_point[:, region_6] = contact_facets_B[
        :, region_6
    ] + (d4[region_6] - d3[region_6]) * contact_side_BC[:, region_6] / (
        (d4[region_6] - d3[region_6]) + (d5[region_6] - d6[region_6])
    )
    # otherwise it is inside the triangle hence closest point will be the same as the projected position
    distance_to_closest_triangle_point = _batch_norm(
        closest_point_on_triangle_to_projected_point - projected_element_position
    )

    plane_penetration = np.minimum(distance_from_facet_plane - radius_contacts, 0.0)
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
    no_penetration_idx = np.where(
        (distance_from_facet_plane - radius_contacts) > surface_tol
    )[0]
    # check if the distance to the closest point on the triangle is smaller than the plane_intersection_radius (we can modify this later for cylinder)
    no_intersection_idx = np.where(
        (distance_to_closest_triangle_point - plane_intersection_radius) > surface_tol
    )[0]
    # If rod element does not have any contact with facet plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_penetration_idx] = 0.0

    # If rod element does not intersect with facet, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_intersection_idx] = 0.0

    plane_response_forces = np.zeros_like(external_forces)
    for i in range(len(position_idx_array)):
        plane_response_forces[
            :, position_idx_array[i]
        ] += plane_response_force_contacts[:, i]

    # Update the external forces
    elements_to_nodes_inplace(plane_response_forces, external_forces)

    return (
        _batch_norm(plane_response_force_contacts),
        no_penetration_idx,
        no_intersection_idx,
        normals_on_elements,
    )


class CustomFrictionalSurfaceGrid(NoForces, CustomInteractionSurfaceGrid):
    def __init__(
        self,
        k,
        nu,
        facets,
        facets_grid,
        facets_normals,
        facets_centers,
        side_vectors,
        grid_size,
        slip_velocity_tol,
        static_mu_array,
        kinetic_mu_array,
        kinetic_mu_sideways_pattern,
        facet_pattern_idx,
        gamma,
        step_skip,
        callback_params=[],
        callback=False,
    ):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        facets: numpy.ndarray
            (3,3,dim) array containing data with 'float' type.
            The three vertices of each facet (axis,vertix,facet).
        up_direction: numpy.ndarray
            (3,1) array containing data with 'float' type.
            The up direction of the surface.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        """
        CustomInteractionSurfaceGrid.__init__(
            self,
            k,
            nu,
            facets,
            facets_grid,
            facets_normals,
            facets_centers,
            side_vectors,
            grid_size,
        )
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
        self.current_step = 0
        self.every = step_skip
        self.callback_params = callback_params
        self.callback = callback
        self.facet_pattern_idx = facet_pattern_idx
        self.kinetic_mu_sideways_pattern = kinetic_mu_sideways_pattern

    # kinetic and static friction should separate functions
    # for now putting them together to figure out common variables
    def apply_forces(self, system, time=0.0):
        """
        Call numba implementation to apply friction forces
        Parameters
        ----------
        system
        time

        Returns
        -------

        """
        (
            self.position_idx_array,
            self.facet_idx_array,
            self.element_position,
        ) = find_contact_facets_idx(
            self.facets_grid,
            self.x_min,
            self.y_min,
            self.grid_size,
            system.position_collection,
        )

        self.no_penetration_idx, self.no_intersection_idx = anisotropic_friction(
            self.facets,
            self.facets_centers,
            self.facets_normals,
            self.element_position,
            self.side_vectors,
            self.position_idx_array,
            self.facet_idx_array,
            self.facet_pattern_idx,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.kinetic_mu_sideways_pattern,
            self.gamma,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system.radius,
            system.mass,
            system.tangents,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.internal_forces,
            system.external_forces,
            system.internal_torques,
            system.external_torques,
        )

        if self.callback:
            self.make_callback()

    def make_callback(self):
        """make_callback."""
        if self.current_step % self.every == 0:
            self.callback_params["facet_idx_array"].append(self.facet_idx_array.copy())
            self.callback_params["no_penetration_idx"].append(
                self.no_penetration_idx.copy()
            )
            self.callback_params["no_intersection_idx"].append(
                self.no_intersection_idx.copy()
            )
        self.current_step += 1


@njit(cache=True)
def anisotropic_friction(
    facets,
    facets_centers,
    facets_normals,
    element_position,
    side_vectors,
    position_idx_array,
    facet_idx_array,
    facet_pattern_idx,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    kinetic_mu_sideways_pattern,
    gamma,
    static_mu_forward,
    static_mu_backward,
    static_mu_sideways,
    radius,
    mass,
    tangents,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):

    (
        plane_response_force_mag,
        no_penetration_idx,
        no_intersection_idx,
        normals_on_elements,
    ) = apply_normal_force_numba(
        facets,
        facets_normals,
        facets_centers,
        element_position,
        side_vectors,
        position_idx_array,
        facet_idx_array,
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

    element_velocity = node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

    if len(position_idx_array) > 0:
        tangents_contacts = tangents[:, position_idx_array]
        element_velocity_contacts = element_velocity[:, position_idx_array]
        radius_contacts = radius[position_idx_array]
        omega_collection_contacts = omega_collection[:, position_idx_array]
        director_collection_contacts = director_collection[:, :, position_idx_array]
        kinetic_mu_sideways_array = kinetic_mu_sideways * np.ones_like(
            position_idx_array, dtype=np.float64
        )
        elements_within_pattern_idx = np.where(facet_pattern_idx[facet_idx_array])[
            0
        ]  # this checks which facets that are within the pattern
        kinetic_mu_sideways_array[
            elements_within_pattern_idx
        ] = kinetic_mu_sideways_pattern
    else:
        tangents_contacts = tangents
        element_velocity_contacts = element_velocity
        radius_contacts = radius
        omega_collection_contacts = omega_collection
        director_collection_contacts = director_collection
        kinetic_mu_sideways_array = kinetic_mu_sideways * np.ones_like(radius)

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
    slip_function_along_axial_direction = find_slipping_elements(
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
    # directors_transpose = _batch_matrix_transpose(director_collection)

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
    slip_function_along_rolling_direction = find_slipping_elements(
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
    kinetic_friction_force_along_axial_direction_contacts[..., no_penetration_idx] = 0.0
    kinetic_friction_force_along_axial_direction_contacts[
        ..., no_intersection_idx
    ] = 0.0

    # Apply kinetic friction in rolling direction.
    kinetic_friction_force_along_rolling_direction_contacts = -(
        (1.0 - slip_function_along_rolling_direction)
        * kinetic_mu_sideways_array
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, rolling_direction)
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_rolling_direction_contacts[
        ..., no_penetration_idx
    ] = 0.0
    kinetic_friction_force_along_rolling_direction_contacts[
        ..., no_intersection_idx
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
    static_friction_force_along_axial_direction_contacts[..., no_penetration_idx] = 0.0
    kinetic_friction_force_along_rolling_direction_contacts[
        ..., no_intersection_idx
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
        ..., no_penetration_idx
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
    elements_to_nodes_inplace(
        kinetic_friction_force_along_axial_direction, external_forces
    )
    elements_to_nodes_inplace(
        kinetic_friction_force_along_rolling_direction, external_forces
    )
    external_torques += kinetic_rolling_torque
    elements_to_nodes_inplace(
        static_friction_force_along_axial_direction, external_forces
    )
    elements_to_nodes_inplace(
        static_friction_force_along_rolling_direction, external_forces
    )
    external_torques += static_rolling_torque

    return no_penetration_idx, no_intersection_idx


def find_contact_facets_idx_single(
    facets_grid,
    x_min,
    y_min,
    grid_size,
    position_collection,
):

    element_position = node_to_element_position(position_collection)
    n_element = element_position.shape[-1]
    facet_idx_array = np.array([0])
    ## Note: Boundary issue needs to be fixed
    grid_position = np.round(
        (element_position[0:2, :] - np.array([x_min, y_min]).reshape((2, 1)))
        / grid_size
    )

    # find facet neighborhood of each element position
    for i in range(n_element):
        try:
            facet_idx_1 = facets_grid[
                (int(grid_position[0, i]), int(grid_position[1, i]))
            ]  # first quadrant
        except:
            facet_idx_1 = np.empty((0))
        try:
            facet_idx_2 = facets_grid[
                (int(grid_position[0, i] - 1), int(grid_position[1, i]))
            ]  # second quadrant
        except:
            facet_idx_2 = np.empty((0))
        try:
            facet_idx_3 = facets_grid[
                (int(grid_position[0, i] - 1), int(grid_position[1, i] - 1))
            ]  # third quadrant
        except:
            facet_idx_3 = np.empty((0))
        try:
            facet_idx_4 = facets_grid[
                (int(grid_position[0, i]), int(grid_position[1, i] - 1))
            ]  # fourth quadrant
        except:
            facet_idx_4 = np.empty((0))
        facet_idx_element = np.concatenate(
            (facet_idx_1, facet_idx_2, facet_idx_3, facet_idx_4)
        )
        facet_idx_element_no_duplicates = np.unique(facet_idx_element)
        facet_idx_array = np.concatenate(
            (facet_idx_array, facet_idx_element_no_duplicates)
        )

    facet_idx_array = facet_idx_array.astype(int)

    return facet_idx_array, element_position
