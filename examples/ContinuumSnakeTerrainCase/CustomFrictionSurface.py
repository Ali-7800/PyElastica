import numpy as np
from elastica.interaction import *
from elastica.external_forces import NoForces

from numba import njit
from elastica.interaction import (
    elements_to_nodes_inplace,
    node_to_element_mass_or_force,
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
    _batch_product_i_ik_to_k,
    _batch_product_i_k_to_ik,
    _batch_vector_sum,
    _batch_matrix_transpose,
    _batch_vec_oneD_vec_cross,
)

from surface_functions import (
    calculate_facet_normals_centers
)

from SnakeIntegrator import (
    BoundaryError
)

@njit(cache=True)
def find_facet_neighborhood(facets_centers,point,neighborhood_size):
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
    deltas = facets_centers - np.array([point[0],point[1],point[2]]).reshape((3,1))
    distances = _batch_norm(deltas)
    neighborhood_idx = np.where(distances<neighborhood_size)[0]
    return neighborhood_idx


@njit(cache=True)
def find_closest_facets(facets_centers,positions):
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
        deltas = facets_centers - np.array([positions[0,i],positions[1,i],positions[2,i]]).reshape((3,1))
        distances = _batch_norm(deltas)
        if distances.size != 0:
            closest_facets[i] = np.argmin(distances)
    return closest_facets


@njit(cache=True)
def find_closest_facet(facets_centers,position):
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
    deltas = facets_centers - np.array([position[0],position[1],position[2]]).reshape((3,1))
    distances = _batch_norm(deltas)
    closest_facet = np.argmin(distances)
    return closest_facet



def find_contact_facets_idx(
    facets_grid,
    x_min,
    y_min,
    grid_size,
    position_collection,
    ):

    element_position = node_to_element_position(position_collection)
    n_element = element_position.shape[-1]
    position_idx_array =  np.empty((0))
    facet_idx_array = np.empty((0))
    grid_position = np.round((element_position[0:2,:]-np.array([x_min,y_min]).reshape((2,1)))/grid_size)

    #find facet neighborhood of each element position
    for i in range(n_element):
        try:
            facet_idx_1 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]))] #first quadrant
        except:
            facet_idx_1 = np.empty((0))
        try:
            facet_idx_2 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]))] #second quadrant
        except:
            facet_idx_2 = np.empty((0))
        try:
            facet_idx_3 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]-1))] #third quadrant
        except:
            facet_idx_3 = np.empty((0))
        try:
            facet_idx_4 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]-1))] #fourth quadrant
        except:
            facet_idx_4 = np.empty((0))
        facet_idx_element = np.concatenate((facet_idx_1,facet_idx_2,facet_idx_3,facet_idx_4))
        facet_idx_element_no_duplicates = np.unique(facet_idx_element)
        if facet_idx_element_no_duplicates.size == 0:
            raise BoundaryError("Snake outside surface boundary") #a snake element is on four grids with no facets

        facet_idx_array = np.concatenate((facet_idx_array,facet_idx_element_no_duplicates))
        n_contacts = facet_idx_element_no_duplicates.shape[0]
        position_idx_array = np.concatenate((position_idx_array,i*np.ones((n_contacts,))))

    position_idx_array = position_idx_array.astype(int)
    facet_idx_array = facet_idx_array.astype(int)
    return position_idx_array,facet_idx_array,element_position




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

    def __init__(self, k, nu, facets,facets_grid, grid_size, **kwargs):
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
        n_facets = facets.shape[-1]

        assert ("up_direction" in kwargs.keys() or "facet_vertex_normals" in kwargs.keys()),"Please provide valid up_direction or vertices_normals"
        assert (~("up_direction" in kwargs.keys() and "facet_vertex_normals" in kwargs.keys())),"Please provide one of up_direction or vertices_normals not both"

        self.k = k
        self.nu = nu
        self.x_min = np.min(facets[0,:,:])
        self.y_min = np.min(facets[1,:,:])
        self.facets_grid = facets_grid
        
        
        if "up_direction" in kwargs.keys():
            self.up_direction = kwargs["up_direction"]
            self.facets_normals,self.facets_centers = calculate_facet_normals_centers(facets = self.facets,up_direction = self.up_direction)
        else:
            self.facet_vertex_normals = kwargs["facet_vertex_normals"]
            self.facets_normals,self.facets_centers = calculate_facet_normals_centers(facets = self.facets,facet_vertex_normals = self.facet_vertex_normals)
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

        self.position_idx_array,self.facet_idx_array,self.element_position = find_contact_facets_idx(
        self.facets_grid,
        self.x_min,
        self.y_min,
        self.grid_size,
        system.position_collection,
        )


        return apply_normal_force_numba(
            self.facets_normals,
            self.facets_centers,
            self.element_position,
            self.position_idx_array,
            self.facet_idx_array,
            self.surface_tol,
            self.k,
            self.nu,
            system.radius,
            system.mass,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
            system.lengths,
        )


@njit(cache=True)
def apply_normal_force_numba(
    facets_normals,
    facets_centers,
    element_position,
    position_idx_array,
    facet_idx_array,
    surface_tol,
    k,
    nu,
    radius,
    mass,
    position_collection,
    velocity_collection,
    internal_forces,
    external_forces,
    lengths,
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

    if len(facet_idx_array)>0:
        element_position_contacts = element_position[:,position_idx_array]
        contact_facet_centers =  facets_centers[:,facet_idx_array]
        normals_on_elements = facets_normals[:,facet_idx_array]
        radius_contacts = radius[position_idx_array]
        element_velocity_contacts = element_velocity[:,position_idx_array]

    else:
        element_position_contacts = element_position
        contact_facet_centers = np.zeros_like(element_position)
        normals_on_elements = np.zeros_like(element_position)
        radius_contacts = radius
        element_velocity_contacts = element_velocity

    

    # Elastic force response due to penetration

    distance_from_plane = _batch_dot(normals_on_elements,(element_position_contacts - contact_facet_centers))
    plane_penetration = -np.abs(np.minimum(distance_from_plane - radius_contacts, 0.0))**1.5
    elastic_force = -k * _batch_product_k_ik_to_ik(plane_penetration,normals_on_elements)


    normal_component_of_element_velocity = _batch_dot(normals_on_elements,element_velocity_contacts)
    damping_force = -nu * _batch_product_k_ik_to_ik(normal_component_of_element_velocity,normals_on_elements)

    # Compute total plane response force
    plane_response_force_contacts = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane - radius_contacts) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_contact_point_idx] = 0.0


    plane_response_forces = np.zeros_like(external_forces)
    for i in range(len(position_idx_array)):
        plane_response_forces[:,position_idx_array[i]] += plane_response_force_contacts[:,i]


    # Update the external forces
    elements_to_nodes_inplace(plane_response_forces, external_forces)
    return (_batch_norm(plane_response_force_contacts), no_contact_point_idx,normals_on_elements)








class CustomFrictionalSurfaceGrid(NoForces, CustomInteractionSurfaceGrid):


    def __init__(
        self,
        k,
        nu,
        facets,
        facets_grid,
        grid_size,
        slip_velocity_tol,
        static_mu_array,
        kinetic_mu_array,
        gamma,
        ** kwargs
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
        CustomInteractionSurfaceGrid.__init__(self, k, nu, facets, facets_grid,grid_size, **kwargs)
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
        self.position_idx_array,self.facet_idx_array,self.element_position = find_contact_facets_idx(
        self.facets_grid,
        self.x_min,
        self.y_min,
        self.grid_size,
        system.position_collection,
        )

        anisotropic_friction(
            self.facets_centers,
            self.facets_normals,
            self.element_position,
            self.position_idx_array,
            self.facet_idx_array,
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
            system.radius,
            system.mass,
            system.tangents,
            system.position_collection,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.internal_forces,
            system.external_forces,
            system.internal_torques,
            system.external_torques,
            system.lengths,
        )


@njit(cache=True)
def anisotropic_friction(
    facets_centers,
    facets_normals,
    element_position,
    position_idx_array,
    facet_idx_array,
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


    plane_response_force_mag, no_contact_point_idx,normals_on_elements = apply_normal_force_numba(
        facets_normals,
        facets_centers,
        element_position,
        position_idx_array,
        facet_idx_array,
        surface_tol,
        k,
        nu,
        radius,
        mass,
        position_collection,
        velocity_collection,
        internal_forces,
        external_forces,
        lengths,
    )
    # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
    # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
    # to the plane. So friction forces can only be in plane forces and not out of plane.

    element_velocity = node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )

    if len(position_idx_array)>0:
        tangents_contacts = tangents[:,position_idx_array]
        element_velocity_contacts = element_velocity[:,position_idx_array]
        radius_contacts = radius[position_idx_array]
        omega_collection_contacts = omega_collection[:,position_idx_array]
        director_collection_contacts = director_collection[:,:,position_idx_array]
    else:
        tangents_contacts = tangents
        element_velocity_contacts = element_velocity
        radius_contacts = radius
        omega_collection_contacts = omega_collection
        director_collection_contacts = director_collection




    tangent_along_normal_direction = _batch_dot(normals_on_elements,tangents_contacts)
    tangent_perpendicular_to_normal_direction = tangents_contacts -  _batch_product_k_ik_to_ik(tangent_along_normal_direction,normals_on_elements)

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
    velocity_mag_along_axial_direction = _batch_dot(element_velocity_contacts, axial_direction)
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
    torque_arm = -_batch_product_k_ik_to_ik(radius_contacts,normals_on_elements)
    velocity_along_rolling_direction = _batch_dot(element_velocity_contacts, rolling_direction)
    velocity_sign_along_rolling_direction = np.sign(velocity_along_rolling_direction)

    directors_transpose_contacts = _batch_matrix_transpose(director_collection_contacts)
    directors_transpose = _batch_matrix_transpose(director_collection)

    # w_rot = Q.T @ omega @ Q @ r
    rotation_velocity = _batch_matvec(
        directors_transpose_contacts,
        _batch_cross(omega_collection_contacts, _batch_matvec(director_collection_contacts, torque_arm)),
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
    kinetic_friction_force_along_axial_direction_contacts[..., no_contact_point_idx] = 0.0

    
    
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
    kinetic_friction_force_along_rolling_direction_contacts[..., no_contact_point_idx] = 0.0


    # torque = Q @ r @ Fr
    kinetic_rolling_torque_contacts = _batch_matvec(director_collection_contacts,_batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction_contacts),)
    

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
        np.minimum(np.fabs(gamma*velocity_along_axial_direction), max_friction_force)
        * velocity_sign_along_axial_direction
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
    static_friction_force_along_axial_direction_contacts[..., no_contact_point_idx] = 0.0
    

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
        np.minimum(np.fabs(gamma*velocity_along_rolling_direction), max_friction_force)
        * velocity_sign_along_rolling_direction
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
    static_friction_force_along_rolling_direction_contacts[..., no_contact_point_idx] = 0.0
    
    static_rolling_torque_contacts = _batch_matvec(
        director_collection_contacts,
        _batch_cross(torque_arm, static_friction_force_along_rolling_direction_contacts),
    )

    #add up contribution from all contacts
    
    kinetic_friction_force_along_axial_direction = np.zeros_like(element_velocity)
    kinetic_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
    kinetic_rolling_torque = np.zeros_like(element_velocity)
    static_friction_force_along_axial_direction = np.zeros_like(element_velocity)
    static_friction_force_along_rolling_direction = np.zeros_like(element_velocity)
    static_rolling_torque = np.zeros_like(element_velocity)

    for i in range(len(position_idx_array)):
        kinetic_friction_force_along_axial_direction[:,position_idx_array[i]] += kinetic_friction_force_along_axial_direction_contacts[:,i]
        kinetic_friction_force_along_rolling_direction[:,position_idx_array[i]] += kinetic_friction_force_along_rolling_direction_contacts[:,i]
        kinetic_rolling_torque[:,position_idx_array[i]] += kinetic_rolling_torque_contacts[:,i]
        static_friction_force_along_axial_direction[:,position_idx_array[i]] += static_friction_force_along_axial_direction_contacts[:,i]
        static_friction_force_along_rolling_direction[:,position_idx_array[i]] += static_friction_force_along_rolling_direction_contacts[:,i]
        static_rolling_torque[:,position_idx_array[i]] += static_rolling_torque_contacts[:,i]


    #apply all forces and torques
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
    grid_position = np.round((element_position[0:2,:]-np.array([x_min,y_min]).reshape((2,1)))/grid_size)

    #find facet neighborhood of each element position
    for i in range(n_element):
        try:
            facet_idx_1 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]))] #first quadrant
        except:
            facet_idx_1 = np.empty((0))
        try:
            facet_idx_2 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]))] #second quadrant
        except:
            facet_idx_2 = np.empty((0))
        try:
            facet_idx_3 = facets_grid[(int(grid_position[0,i]-1),int(grid_position[1,i]-1))] #third quadrant
        except:
            facet_idx_3 = np.empty((0))
        try:
            facet_idx_4 = facets_grid[(int(grid_position[0,i]),int(grid_position[1,i]-1))] #fourth quadrant
        except:
            facet_idx_4 = np.empty((0))
        facet_idx_element = np.concatenate((facet_idx_1,facet_idx_2,facet_idx_3,facet_idx_4))
        facet_idx_element_no_duplicates = np.unique(facet_idx_element)
        facet_idx_array = np.concatenate((facet_idx_array,facet_idx_element_no_duplicates))

    facet_idx_array = facet_idx_array.astype(int)

    return facet_idx_array,element_position







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



# Anisotrpic Coulomb friction -- Kinetic friction ONLY. No static, No rolling!!
# Developed based on and validated against to D. HU and M Shelley PNAS 2009
# NOTE: friction coefficients are passed as arrays in the order
# mu_forward : mu_backward : mu_sideways
# head is at x[0] and forward means head to tail
# NOTE: HERE, OVERALL FORCE IN OPPOSITE DERICTION IS REMOVED FROM 
# GROUND RESPONCE FORCE AND FRICTION, to make the model generic for interacting 
# with 3-D terrains and objects.
# (same friction model is implemented in C elastica)

class AnisotropicKineticCoulomb_3DTerrain_Interaction(NoForces, InteractionPlane):
    """
    This anisotropic kinetic Coulomb friction plane class for building 3D terrains
    and objects. Rod interacts with this plane through only elastic and damping contacts.  

        Attributes
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            The normal vector of the plane.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
    """

    def __init__(
        self,
        k,
        nu,
        plane_origin,
        plane_normal,
        slip_velocity_tol,
        kinetic_mu_array,
        list_of_grounds,
        write_object,
        randomize=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        plane_origin: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Origin of the plane.
        plane_normal: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            The normal vector of the plane.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        list_of_grounds : list
            selection of ground types and obstacles.
        """
        InteractionPlane.__init__(self, k, nu, plane_origin, plane_normal)
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array
        
        self.list_of_grounds = list_of_grounds
        
        if write_object:
            file_object = open("object.inc","w")
            file_object.close()
            
        if "bumpy" in self.list_of_grounds:
            #matrix of spheres - infinite space 
            self.block_size=0.015        #size of each block containing one sphere at the center
            self.sphere_radius=0.0075    #less equal to 0.5*blockWidth
            self.sphere_z_plane=0.0      #z position of all spheres
            
            #these are just for rendering - render enough sphere to occupy the target space
            numSphere_x = 150;
            numSphere_y = 150;
            
            #write povray file
            if write_object:
                file_object = open("object.inc","a")
                for i in range(numSphere_x):
                    position_x = self.block_size * (0.5 * numSphere_x - 0.5 - i)
                    for j in range(numSphere_y):
                        position_y = self.block_size * (0.5 * numSphere_y - 0.5 - j)
                        file_object.writelines("sphere{ <%f,%f,%f>,%f \n"% (position_x, position_y,0.0,self.sphere_radius))
                        file_object.writelines("pigment{ color rgbt<0.7,0.4,0.1,0.2>}\n")
                        file_object.writelines("scale<4,4,4>\n")
                        file_object.writelines("rotate<0,90,90>\n")
                        file_object.writelines("translate<2,0,4>\n")
                        file_object.writelines("}\n")
                file_object.close()

        if "rock" in self.list_of_grounds:
            
            #single spheres
            sphere_location = []
            sphere_radius = []
            
            sphere_location.append(np.array([0.5,-0.4,-1.96]))
            sphere_radius.append(1.99)
            sphere_location.append(np.array([-0.4,-0.1,-1.95]))
            sphere_radius.append(1.98)
            sphere_location.append(np.array([0.6,0.5,-0.025]))
            sphere_radius.append(0.25)                       
        
            #write povray file
            if write_object:
                            
                file_object = open("object.inc","a")
                for i in range(len(sphere_radius)):
                    position = sphere_location[i]
                    radius = sphere_radius[i]
                    
                    file_object.writelines("sphere{ <%f,%f,%f>,%f \n"% (position[0],position[1],position[2],radius))
                    file_object.writelines("pigment{ color rgbt<1.0,0.65,0,0>}\n")
                    file_object.writelines("scale<4,4,4>\n")
                    file_object.writelines("rotate<0,90,90>\n")
                    file_object.writelines("translate<2,0,4>\n")
                    file_object.writelines("}\n")
                file_object.close()
            
            self.pass_sphere_location = np.array(sphere_location)
            self.pass_sphere_radius = np.array(sphere_radius)


        if "pebble" in self.list_of_grounds:
            #get parser args
            block_size = kwargs.get("block_size", 0.0) 

            #matrix of pebbles
            self.pattern_center = np.array([1.5,0.0,0.0])
            self.block_pbs_sizes = np.array([block_size,0.009])   #block size and pebble size 
            num_pbs_x = 100
            num_pbs_y = 100
            self.num_pbs = np.array([num_pbs_x,num_pbs_y])   #number of posts in x and y directions - even numbers only for now
            self.perturb_x = np.zeros((num_pbs_x,num_pbs_y))
            self.perturb_y = np.zeros((num_pbs_x,num_pbs_y))
            self.position_z = 0.0;
            radius = self.block_pbs_sizes[1]*np.ones((num_pbs_x,num_pbs_y))
            self.perturb_radius = np.zeros((num_pbs_x,num_pbs_y))
            
            
            #perturb x and y position to introduce randomness
            if randomize:
                max_perturb_r = 0.8*self.block_pbs_sizes[1]     #allow maximum perturbation of 80% of radius
                self.perturb_radius = np.random.uniform(size = (num_pbs_x,num_pbs_y), low = -max_perturb_r, high = max_perturb_r)
                radius += self.perturb_radius
                #set the range of the randomization according to the local radius
                perturb_range = 0.5*self.block_pbs_sizes[0]*np.ones((num_pbs_x,num_pbs_y)) - radius
                #distribution of the randomness - linearly increasing probability for larger offsets 
                probability_x = np.random.power(size = (num_pbs_x,num_pbs_y), a = 2)
                probability_y = np.random.power(size = (num_pbs_x,num_pbs_y), a = 2) 
                sign_x = np.random.choice([-1.0,1.0], size = (num_pbs_x,num_pbs_y))
                sign_y = np.random.choice([-1.0,1.0], size = (num_pbs_x,num_pbs_y))
                self.perturb_x = sign_x * perturb_range * probability_x 
                self.perturb_y = sign_y * perturb_range * probability_y
                               
            #write povray file
            if write_object:
                file_object = open("object.inc","a")
                for i in range(self.num_pbs[0]):
                    for j in range(self.num_pbs[1]):
                        position_x = self.pattern_center[0] + self.perturb_x[i][j] + self.block_pbs_sizes[0] * (-self.num_pbs[0]/2.0+0.5 + i)
                        position_y = self.pattern_center[1] + self.perturb_y[i][j] + self.block_pbs_sizes[0] * (-self.num_pbs[1]/2.0+0.5 + j)
                        file_object.writelines("sphere{ <%f,%f,%f>,%f \n"% (position_x,position_y,self.position_z,radius[i][j]))
                        file_object.writelines("pigment{ color rgbt<1.0,0.65,0,0>}\n")
                        file_object.writelines("scale<4,4,4>\n")
                        file_object.writelines("rotate<0,90,90>\n")
                        file_object.writelines("translate<2,0,4>\n")
                        file_object.writelines("}\n")
                file_object.close()
        
        if "post" in self.list_of_grounds:         
            #matrix of posts
            self.pattern_center = np.array([1.5,0.0,0.0])
            self.block_post_sizes = np.array([0.1,0.005])   #block size and post size 
            num_posts_x = 20
            num_posts_y = 20
            self.num_posts = np.array([num_posts_x,num_posts_y])   #number of posts in x and y directions - even numbers only for now
            self.perturb_x = np.zeros((num_posts_x,num_posts_y))
            self.perturb_y = np.zeros((num_posts_x,num_posts_y))
            
            #perturb x and y position to introduce randomness
            if randomize:
                max_perturb = 0.8*self.block_post_sizes[0]/2.0      #allow maximum perturbation of 80% of block size
                self.perturb_x = np.random.uniform(size = (num_posts_x,num_posts_y), low = -max_perturb, high = max_perturb)
                self.perturb_y = np.random.uniform(size = (num_posts_x,num_posts_y), low = -max_perturb, high = max_perturb)
                               
            #write povray file
            if write_object:
                file_object = open("object.inc","a")
                for i in range(self.num_posts[0]):
                    for j in range(self.num_posts[1]):
                        position_x = self.pattern_center[0] + self.perturb_x[i][j] + self.block_post_sizes[0] * (-self.num_posts[0]/2.0+0.5 + i)
                        position_y = self.pattern_center[1] + self.perturb_y[i][j] + self.block_post_sizes[0] * (-self.num_posts[1]/2.0+0.5 + j)
                        file_object.writelines("cylinder{ <%f,%f,%f>,\n"% (position_x, position_y,0.0))
                        file_object.writelines("<%f,%f,%f>,%f \n"% (position_x, position_y,0.1,self.block_post_sizes[1]))
                        file_object.writelines("pigment{ color rgbt<0.7,0.8,0.1,0.2>}\n")
                        file_object.writelines("scale<4,4,4>\n")
                        file_object.writelines("rotate<0,90,90>\n")
                        file_object.writelines("translate<2,0,4>\n")
                        file_object.writelines("}\n")
                file_object.close()

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
        #Apply all selected interaction modules --
        #the default plain ground should always be inculded.
        if "bumpy" in self.list_of_grounds:
            #interaction with matrix of sphere
            matrix_sphere_interaction(
                self.slip_velocity_tol,
                self.k,
                self.nu,
                self.kinetic_mu_forward,
                self.kinetic_mu_backward,
                self.kinetic_mu_sideways,
                self.block_size,
                self.sphere_radius,
                self.sphere_z_plane,
                system.radius,
                system.tangents,
                system.position_collection,
                system.director_collection,
                system.velocity_collection,
                system.omega_collection,
                system.internal_forces,
                system.external_forces,
                system.internal_torques,
                system.external_torques,
            )
        
        if "rock" in self.list_of_grounds:
            #interaction with single sphere
            single_sphere_interaction(
                self.slip_velocity_tol,
                self.k,
                self.nu,
                self.kinetic_mu_forward,
                self.kinetic_mu_backward,
                self.kinetic_mu_sideways,
                self.pass_sphere_location,
                self.pass_sphere_radius,
                system.radius,
                system.tangents,
                system.position_collection,
                system.compute_position_center_of_mass(),
                system.director_collection,
                system.velocity_collection,
                system.omega_collection,
                system.internal_forces,
                system.external_forces,
                system.internal_torques,
                system.external_torques,
            )
            
        if "pebble" in self.list_of_grounds:
            #interaction with single sphere
            matrix_pebble_interaction(
                self.slip_velocity_tol,
                self.k,
                self.nu,
                self.kinetic_mu_forward,
                self.kinetic_mu_backward,
                self.kinetic_mu_sideways,
                self.pattern_center,
                self.block_pbs_sizes,
                self.num_pbs,
                self.perturb_x,
                self.perturb_y,
                self.position_z,
                self.perturb_radius,
                system.radius,
                system.tangents,
                system.position_collection,
                system.compute_position_center_of_mass(),
                system.director_collection,
                system.velocity_collection,
                system.omega_collection,
                system.internal_forces,
                system.external_forces,
                system.internal_torques,
                system.external_torques,
            )

            
        if "post" in self.list_of_grounds:
            #interaction with single sphere
            matrix_post_interaction(
                self.slip_velocity_tol,
                self.k,
                self.nu,
                self.kinetic_mu_forward,
                self.kinetic_mu_backward,
                self.kinetic_mu_sideways,
                self.pattern_center,
                self.block_post_sizes,
                self.num_posts,
                self.perturb_x,
                self.perturb_y,
                system.radius,
                system.tangents,
                system.position_collection,
                system.compute_position_center_of_mass(),
                system.director_collection,
                system.velocity_collection,
                system.omega_collection,
                system.internal_forces,
                system.external_forces,
                system.internal_torques,
                system.external_torques,
            )
        #baseline interaction - plain ground
        plane_interaction(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            system.radius,
            system.mass,
            system.tangents,
            system.position_collection,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.internal_forces,
            system.external_forces,
            system.internal_torques,
            system.external_torques,
        )
        
# baseline interaction - plain ground
# called by rod routine - determine how rod interacts with the plane
@njit(cache=True)
def plane_interaction(
    plane_origin,
    plane_normal,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
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
):
    n_elements = radius.shape[0]
    surface_normal = _batch_product_i_k_to_ik(plane_normal, np.ones(n_elements))
    # Compute element wise ground penetration
    element_position = node_to_element_position(position_collection)
    distance_from_plane = _batch_dot(surface_normal, (element_position - plane_origin)) 
    plane_penetration = np.minimum(distance_from_plane - radius, 0.0)
    
    plane_response_force_mag, no_contact_point_idx = apply_response(
        surface_normal,
        plane_penetration,
        k,
        nu,
        mass,
        velocity_collection,
        external_forces,
    )
    apply_kinematic_friction(
        surface_normal,
        slip_velocity_tol,
        kinetic_mu_forward,
        kinetic_mu_backward,
        kinetic_mu_sideways,
        plane_response_force_mag,
        no_contact_point_idx,
        mass,
        tangents,
        velocity_collection,
        external_forces,
    )


# interaction with matrix of sphere
# called by rod routine - determine how rod interacts with the bumpy ground
@njit(cache=True)
def matrix_sphere_interaction(
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    block_size,
    sphere_radius,
    sphere_z_plane,
    radius,
    tangents,
    position_collection,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):
    n_elements = radius.shape[0]
    # Compute element wise ground penetration
    element_position = node_to_element_position(position_collection)
    # floor division - take x and y index, and make z to be zeros
    block_index = element_position // np.array([block_size,block_size,100.0]).reshape(3,1)
    local_center = ((block_index + np.array([0.5,0.5,sphere_z_plane]).reshape(3,1))
                    * np.array([block_size,block_size,1.0]).reshape(3,1))
    center_to_element = element_position - local_center
    distance_from_center = _batch_norm(center_to_element)
    surface_normal = _batch_product_k_ik_to_ik(
        1.0 / ( distance_from_center + 1e-14), center_to_element
    )
    plane_penetration = np.minimum(distance_from_center - (radius+sphere_radius), 0.0)
    
    plane_response_force_mag, no_contact_point_idx = apply_response(
        surface_normal,
        plane_penetration,
        k,
        nu,
        velocity_collection,
        external_forces,
    )
    apply_kinematic_friction(
        surface_normal,
        slip_velocity_tol,
        kinetic_mu_forward,
        kinetic_mu_backward,
        kinetic_mu_sideways,
        plane_response_force_mag,
        no_contact_point_idx,
        tangents,
        velocity_collection,
        external_forces,
    )

# interaction with single spheres
# called by rod routine - determine how rod interacts with big rocks
@njit(cache=True)
def single_sphere_interaction(
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    sphere_location,
    sphere_radius,
    radius,
    tangents,
    position_collection,
    center_of_mass,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):
    n_elements = radius.shape[0]
    # compute element wise ground penetration
    element_position = node_to_element_position(position_collection)
                
    for i in range(len(sphere_radius)):
        single_position = sphere_location[i,...]
        single_radius = sphere_radius[i] 
        surface_radius = np.sqrt(single_radius*single_radius - single_position[2]*single_position[2])
        surface_center = np.array([single_position[0],single_position[1],0.0])
        center_to_surface_center = np.linalg.norm(center_of_mass - surface_center)
         
        if ((center_to_surface_center-surface_radius)<0.175):
            center_to_element = element_position - _batch_product_i_k_to_ik(single_position, np.ones(n_elements))
            distance_from_center = _batch_norm(center_to_element)
            surface_normal = _batch_product_k_ik_to_ik(
                1.0 / ( distance_from_center + 1e-14), center_to_element
            )
            plane_penetration = np.minimum(distance_from_center - (radius+single_radius), 0.0)
            
            plane_response_force_mag, no_contact_point_idx = apply_response(
                surface_normal,
                plane_penetration,
                k,
                nu,
                velocity_collection,
                external_forces,
            )
            apply_kinematic_friction(
                surface_normal,
                slip_velocity_tol,
                kinetic_mu_forward,
                kinetic_mu_backward,
                kinetic_mu_sideways,
                plane_response_force_mag,
                no_contact_point_idx,
                tangents,
                velocity_collection,
                external_forces,
            )
 

# interaction with matrix of spheres
# called by rod routine - determine how rod interacts with the pebble region
@njit(cache=True)
def matrix_pebble_interaction(
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    pattern_center,
    block_pbs_sizes,
    num_pbs,
    perturb_x,
    perturb_y,
    position_z,
    perturb_radius,
    radius,
    tangents,
    position_collection,
    center_of_mass,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):
    center_to_center = np.linalg.norm(center_of_mass - pattern_center)
    pattern_radius = block_pbs_sizes[0]*num_pbs[0]*np.sqrt(2.0)/2.0
    
    if ((center_to_center-pattern_radius)<0.175):
        n_elements = radius.shape[0]
        # compute element wise position
        element_position = node_to_element_position(position_collection)
        # compute relative postion to pattern center
        center_to_element = element_position - _batch_product_i_k_to_ik(pattern_center, np.ones(n_elements))
        # floor division - take x and y index, and make z to be zeros
        block_index = center_to_element // np.array([block_pbs_sizes[0],block_pbs_sizes[0],100.0]).reshape(3,1)
        local_center = ((block_index + np.array([0.5,0.5,0.0]).reshape(3,1))
                      * np.array([block_pbs_sizes[0],block_pbs_sizes[0],0.0]).reshape(3,1))
        local_radius = block_pbs_sizes[1]*np.ones(n_elements)
        
        for i in range(n_elements):
            # make z of the post center to be same with the snake position
            local_center[2][i] = position_z
            # Up to HERE, we assume posts are created on standard grid in infinite space.
            # However we only want to create a finite number of posts - [num_pbs]
            # So we discard the post positions that are not in the [num_pbs] limit,
            # by making their local_center very large, so they are never encountered by the snake.
            # If the position is what we want, add perturbation in x and y directions
            for j in range(2):
                if abs(block_index[j][i]+0.5)>(0.5*num_pbs[j]):
                    local_center[j][i] = 100.0
                    block_index[j][i] = np.minimum(block_index[j][i],num_pbs[j]/2.0-1.0)
                    block_index[j][i] = np.maximum(block_index[j][i],-num_pbs[j]/2.0)
            
            local_perturb_x = perturb_x[int(block_index[0][i]+0.5*num_pbs[0])][int(block_index[1][i]+0.5*num_pbs[1])]
            local_perturb_y = perturb_y[int(block_index[0][i]+0.5*num_pbs[0])][int(block_index[1][i]+0.5*num_pbs[1])]
            local_center[:,i] += np.array([local_perturb_x,local_perturb_y,0.0])
            
            local_perturb_radius = perturb_radius[int(block_index[0][i]+0.5*num_pbs[0])][int(block_index[1][i]+0.5*num_pbs[1])]
            local_radius[i] += local_perturb_radius

        
        # compute local_center in global coordinate   
        local_center += _batch_product_i_k_to_ik(pattern_center, np.ones(n_elements))
        
        center_to_element = element_position - local_center
        distance_from_center = _batch_norm(center_to_element)
        surface_normal = _batch_product_k_ik_to_ik(
            1.0 / ( distance_from_center + 1e-14), center_to_element
        )
        plane_penetration = np.minimum(distance_from_center - local_radius - radius, 0.0)
    
        plane_response_force_mag, no_contact_point_idx = apply_response(
            surface_normal,
            plane_penetration,
            k,
            nu,
            velocity_collection,
            external_forces,
        )
        apply_kinematic_friction(
            surface_normal,
            slip_velocity_tol,
            kinetic_mu_forward,
            kinetic_mu_backward,
            kinetic_mu_sideways,
            plane_response_force_mag,
            no_contact_point_idx,
            tangents,
            velocity_collection,
            external_forces,
        )     


# interaction with matrix of cylinder
# called by rod routine - determine how rod interacts with the grassy region
@njit(cache=True)
def matrix_post_interaction(
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    pattern_center,
    block_post_sizes,
    num_posts,
    perturb_x,
    perturb_y,
    radius,
    tangents,
    position_collection,
    center_of_mass,
    director_collection,
    velocity_collection,
    omega_collection,
    internal_forces,
    external_forces,
    internal_torques,
    external_torques,
):
    center_to_center = np.linalg.norm(center_of_mass - pattern_center)
    pattern_radius = block_post_sizes[0]*num_posts[0]*np.sqrt(2.0)/2.0
    
    if ((center_to_center-pattern_radius)<0.175):
        n_elements = radius.shape[0]
        # compute element wise position
        element_position = node_to_element_position(position_collection)
        # compute relative postion to pattern center
        center_to_element = element_position - _batch_product_i_k_to_ik(pattern_center, np.ones(n_elements))
        # floor division - take x and y index, and make z to be zeros
        block_index = center_to_element // np.array([block_post_sizes[0],block_post_sizes[0],100.0]).reshape(3,1)
        local_center = ((block_index + np.array([0.5,0.5,0.0]).reshape(3,1))
                      * np.array([block_post_sizes[0],block_post_sizes[0],0.0]).reshape(3,1))
        
        for i in range(n_elements):
            # make z of the post center to be same with the snake position
            local_center[2][i] = element_position[2][i]
            # Up to HERE, we assume posts are created on standard grid in infinite space.
            # However we only want to create a finite number of posts - [num_posts]
            # So we discard the post positions that are not in the [num_posts] limit,
            # by making their local_center very large, so they are never encountered by the snake.
            # If the position is what we want, add perturbation in x and y directions
            for j in range(2):
                if abs(block_index[j][i]+0.5)>(0.5*num_posts[j]):
                    local_center[j][i] = 100.0
                    block_index[j][i] = np.minimum(block_index[j][i],num_posts[j]/2.0-1.0)
                    block_index[j][i] = np.maximum(block_index[j][i],-num_posts[j]/2.0)
            
            local_perturb_x = perturb_x[int(block_index[0][i]+0.5*num_posts[0])][int(block_index[1][i]+0.5*num_posts[1])];
            local_perturb_y = perturb_y[int(block_index[0][i]+0.5*num_posts[0])][int(block_index[1][i]+0.5*num_posts[1])];
            local_center[:,i] += np.array([local_perturb_x,local_perturb_y,0.0])
        
        # compute local_center in global coordinate   
        local_center += _batch_product_i_k_to_ik(pattern_center, np.ones(n_elements))
        
        center_to_element = element_position - local_center
        distance_from_center = _batch_norm(center_to_element)
        surface_normal = _batch_product_k_ik_to_ik(
            1.0 / ( distance_from_center + 1e-14), center_to_element
        )
        plane_penetration = np.minimum(distance_from_center - (radius+block_post_sizes[1]), 0.0)
    
        plane_response_force_mag, no_contact_point_idx = apply_response(
            surface_normal,
            plane_penetration,
            k,
            nu,
            velocity_collection,
            external_forces,
        )
        apply_kinematic_friction(
            surface_normal,
            slip_velocity_tol,
            kinetic_mu_forward,
            kinetic_mu_backward,
            kinetic_mu_sideways,
            plane_response_force_mag,
            no_contact_point_idx,
            tangents,
            velocity_collection,
            external_forces,
        )    

# apply normal - called by apply_forces
# Here, we modify the original ground reaction -- removing the overall force in
# normal direction. Now, response force has only elastic and damping force, which
# friction force is scaled with. 
@njit(cache=True)
def apply_response(    
    surface_normal,
    plane_penetration,
    k,
    nu,
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

    # Elastic force response due to penetration
    elastic_force = -k * _batch_product_k_ik_to_ik(plane_penetration,surface_normal)

    # Damping force response due to velocity towards the plane
    element_velocity = node_to_element_velocity(mass=mass, node_velocity_collection=velocity_collection)
    normal_component_of_element_velocity = _batch_dot(surface_normal, element_velocity)
    damping_force = -nu * _batch_product_k_ik_to_ik(
        normal_component_of_element_velocity, surface_normal)

    # Compute total plane response force
    plane_response_force_total = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where(plane_penetration == 0.0)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_total[..., no_contact_point_idx] = 0.0

    # Update the external forces
    elements_to_nodes_inplace(plane_response_force_total, external_forces)

    return (_batch_norm(plane_response_force_total), no_contact_point_idx)


# apply kinematic friction forces - no static frictions, no rolling friction
# called by plane_interaction
@njit(cache=True)
def apply_kinematic_friction(    
    surface_normal,
    slip_velocity_tol,
    kinetic_mu_forward,
    kinetic_mu_backward,
    kinetic_mu_sideways,
    plane_response_force_mag,
    no_contact_point_idx,
    mass,
    tangents,
    velocity_collection,
    external_forces,
):
    # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
    # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
    # to the plane. So friction forces can only be in plane forces and not out of plane.
    tangent_along_normal_direction = _batch_dot(surface_normal, tangents)
    tangent_perpendicular_to_normal_direction = tangents - _batch_product_k_ik_to_ik(
        tangent_along_normal_direction, surface_normal
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
    element_velocity = node_to_element_velocity(mass=mass, node_velocity_collection=velocity_collection)
    # first apply axial kinetic friction
    velocity_mag_along_axial_direction = _batch_dot(element_velocity, axial_direction)
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
    rolling_direction = _batch_cross(axial_direction, surface_normal)
    velocity_mag_along_rolling_direction = _batch_dot(element_velocity, rolling_direction)
    velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
        velocity_mag_along_rolling_direction, rolling_direction
    )
    slip_function_along_rolling_direction = find_slipping_elements(
        velocity_along_rolling_direction, slip_velocity_tol
    )
    # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
    # and rolling directions. 
    unitized_total_velocity = element_velocity/_batch_norm(element_velocity + 1e-14)
    # Apply kinetic friction in axial direction.
    kinetic_friction_force_along_axial_direction = -(
        (1.0 - slip_function_along_axial_direction)
        * kinetic_mu
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, axial_direction)
        * axial_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_axial_direction[..., no_contact_point_idx] = 0.0
    elements_to_nodes_inplace(
        kinetic_friction_force_along_axial_direction, external_forces
    )
    # Apply kinetic friction in rolling direction.
    kinetic_friction_force_along_rolling_direction = -(
        (1.0 - slip_function_along_rolling_direction)
        * kinetic_mu_sideways
        * plane_response_force_mag
        * _batch_dot(unitized_total_velocity, rolling_direction)
        * rolling_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_along_rolling_direction[..., no_contact_point_idx] = 0.0
    elements_to_nodes_inplace(
        kinetic_friction_force_along_rolling_direction, external_forces
    )
