__doc__ = """ Numba implementation module containing contact between rods and rigid bodies and other rods rigid bodies or surfaces."""


import numpy as np
from elastica.typing import SystemType, RodType, AllowedContactType
from elastica.rod import RodBase
from elastica.rigidbody import RigidBodyBase, Sphere, Cylinder
from elastica.surface import SurfaceBase, Plane
from elastica.contact_forces_calc import (
    _calculate_contact_forces_self_rod_numba,
    _prune_using_aabbs_rod_cylinder,
    _calculate_contact_forces_rod_cylinder,
    _calculate_contact_forces_rod_plane,
    _prune_using_aabbs_rod_rod,
    _calculate_contact_forces_rod_rod,
    _calculate_contact_forces_sphere_plane,
)


class NoContact:
    """
    This is the base class for contact applied between rod-like objects and allowed contact objects.

    Notes
    -----
    Every new contact class must be derived
    from NoContact class.

    """

    def __init__(self):
        """

        Parameters
        ----------

        """

    def _generate_contact_function(self, system_one, system_two):
        self._apply_contact = NotImplementedError

    def apply_contact(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ):
        """
        Apply contact forces and torques between SystemType object and AllowedContactType object.

        In NoContact class, this routine simply passes.

        Parameters
        ----------
        system_one : SystemType
            Rod or rigid-body object
        system_two : AllowedContactType
            Rod, rigid-body or surface object
        Returns
        -------

        """
        self._apply_contact(system_one, system_two)

        return


class ExternalContact(NoContact):
    """
    This class is for applying contact forces between rod-rigid body, rod-rod, rod-surface, rigid body-rigid body, and rigid body-surface.

    If you are want to apply contact forces between rod and cylinder, first system is always rod and second system
    is always cylinder.
    In addition to the contact forces, user can define apply friction forces between rod and cylinder that
    are in contact. For details on friction model refer to this [1]_.
    TODO: Currently friction force is between rod-cylinder, in future implement friction forces between rod-rod.

    Notes
    -----
    The `velocity_damping_coefficient` is set to a high value (e.g. 1e4) to minimize slip and simulate stiction
    (static friction), while friction_coefficient corresponds to the Coulombic friction coefficient.

    Examples
    --------
    How to define contact between rod and cylinder.

    >>> simulator.add_contact_to(rod, cylinder).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ...    velocity_damping_coefficient=10,
    ...    kinetic_friction_coefficient=10,
    ... )

    How to define contact between rod and rod.

    >>> simulator.add_contact_to(rod, rod).using(
    ...    ExternalContact,
    ...    k=1e4,
    ...    nu=10,
    ... )

    .. [1] Preclik T., Popa Constantin., Rude U., Regularizing a Time-Stepping Method for Rigid Multibody Dynamics, Multibody Dynamics 2011, ECCOMAS. URL: https://www10.cs.fau.de/publications/papers/2011/Preclik_Multibody_Ext_Abstr.pdf
    """

    # Dev note:
    # Most of the cylinder-cylinder contact SHOULD be implemented
    # as given in this `paper <http://larochelle.sdsmt.edu/publications/2005-2009/Collision%20Detection%20of%20Cylindrical%20Rigid%20Bodies%20Using%20Line%20Geometry.pdf>`,
    # but the elastica-cpp kernels are implemented.
    # This is maybe to speed-up the kernel, but it's
    # potentially dangerous as it does not deal with "end" conditions
    # correctly.

    def __init__(self, k, nu, velocity_damping_coefficient=0, friction_coefficient=0):
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        velocity_damping_coefficient : float
            Velocity damping coefficient between rigid-body and rod contact is used to apply friction force in the
            slip direction.
        friction_coefficient : float
            For Coulombic friction coefficient for rigid-body and rod contact.
        """
        super(ExternalContact, self).__init__()
        self.velocity_damping_coefficient = velocity_damping_coefficient
        self.friction_coefficient = friction_coefficient
        self.k = k
        self.nu = nu
        self.surface_tol = 1e-4

    def _apply_contact_forces_rod_rigid_body(self, rod, rigid_body):
        # system 1 is always rod and system 2 is always rigid body

        if issubclass(rigid_body.__class__, Cylinder):
            cylinder = rigid_body
            # check if cylinder is in contact
            if _prune_using_aabbs_rod_cylinder(
                rod.position_collection,
                rod.radius,
                rod.lengths,
                cylinder.position_collection,
                cylinder.director_collection,
                cylinder.radius[0],
                cylinder.length[0],
            ):
                return

            x_cyl = (
                cylinder.position_collection[..., 0]
                - 0.5 * cylinder.length * cylinder.director_collection[2, :, 0]
            )

            rod_element_position = 0.5 * (
                rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
            )
            _calculate_contact_forces_rod_cylinder(
                rod_element_position,
                rod.lengths * rod.tangents,
                cylinder.position_collection[..., 0],
                x_cyl,
                cylinder.length * cylinder.director_collection[2, :, 0],
                rod.radius + cylinder.radius,
                rod.lengths + cylinder.length,
                rod.internal_forces,
                rod.external_forces,
                cylinder.external_forces,
                cylinder.external_torques,
                cylinder.director_collection[:, :, 0],
                rod.velocity_collection,
                cylinder.velocity_collection,
                self.k,
                self.nu,
                self.velocity_damping_coefficient,
                self.friction_coefficient,
            )
        elif issubclass(rigid_body.__class__, Sphere):
            # sphere-rod contact
            pass
        else:
            # mesh rigid body-rod contact
            pass

    def _apply_contact_forces_rod_rod(self, rod_one, rod_two):
        # # First, check for a global AABB bounding box, and see whether that
        # # intersects
        if _prune_using_aabbs_rod_rod(
            rod_one.position_collection,
            rod_one.radius,
            rod_one.lengths,
            rod_two.position_collection,
            rod_two.radius,
            rod_two.lengths,
        ):
            return

        _calculate_contact_forces_rod_rod(
            rod_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            rod_one.radius,
            rod_one.lengths,
            rod_one.tangents,
            rod_one.velocity_collection,
            rod_one.internal_forces,
            rod_one.external_forces,
            rod_two.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            rod_two.radius,
            rod_two.lengths,
            rod_two.tangents,
            rod_two.velocity_collection,
            rod_two.internal_forces,
            rod_two.external_forces,
            self.k,
            self.nu,
        )

    def _apply_contact_forces_rod_surface(self, rod, surface):
        if issubclass(surface.__class__, Plane):
            _calculate_contact_forces_rod_plane(
                surface.plane_origin,
                surface.plane_normal,
                surface.surface_tol,
                self.k,
                self.nu,
                rod.radius,
                rod.mass,
                rod.position_collection,
                rod.velocity_collection,
                rod.internal_forces,
                rod.external_forces,
            )
        else:
            # rod-mesh surface contact
            pass

    def _apply_contact_forces_rigid_body_surface(self, rigid_body, surface):
        if issubclass(rigid_body.__class__, Sphere):
            sphere = rigid_body
            if issubclass(surface.__class__, Plane):
                plane = surface
                _calculate_contact_forces_sphere_plane(
                    plane.origin,
                    plane.normal,
                    self.surface_tol,
                    self.k,
                    self.nu,
                    sphere.length,
                    sphere.position_collection,
                    sphere.velocity_collection,
                    sphere.external_forces,
                )
            else:
                # mesh surface-sphere contact
                pass

        elif issubclass(rigid_body.__class__, Cylinder):
            # cylinder = rigid_body
            if issubclass(surface.__class__, Plane):
                plane = surface
                # plane-cylinder contact
            else:
                # mesh surface-cylinder contact
                pass

        else:
            if issubclass(surface.__class__, Plane):
                plane = surface
                # plane mesh-rigid body contact
            else:
                # mesh surface-mesh rigid body contact
                pass
            pass

    def _apply_contact_forces_rigid_body_rigid_body(
        self, rigid_body_one, rigid_body_two
    ):
        pass

    def _generate_contact_function(self, system_one, system_two):
        system_one_class = system_one.__class__
        system_two_class = system_two.__class__

        _apply_contact = NotImplementedError

        if issubclass(system_one_class, RodBase) and issubclass(
            system_two_class, RodBase
        ):
            # rod-rod contact
            _apply_contact = self._apply_contact_forces_rod_rod

        elif issubclass(system_one_class, RodBase) and issubclass(
            system_two_class, RigidBodyBase
        ):
            # rod-rigid body contact
            _apply_contact = self._apply_contact_forces_rod_rigid_body

        elif issubclass(system_one_class, RigidBodyBase) and issubclass(
            system_two_class, RodBase
        ):
            # rigid body-rod contact
            raise TypeError(
                r"To add contact the systems must follow this order: rod, rigidbody, surface"
            )
        elif issubclass(system_one_class, RodBase) and issubclass(
            system_two_class, SurfaceBase
        ):
            # rod-surface contact
            _apply_contact = self._apply_contact_forces_rod_surface

        elif issubclass(system_one_class, SurfaceBase) and issubclass(
            system_two_class, RodBase
        ):
            # surface-rod contact
            raise TypeError(
                r"To add contact the systems must follow this order: rod, rigidbody, surface"
            )

        elif issubclass(system_one_class, RigidBodyBase) and issubclass(
            system_two_class, RigidBodyBase
        ):
            # rigid body-rigid body contact
            _apply_contact = self._apply_contact_forces_rigid_body_rigid_body

        elif issubclass(system_one_class, RigidBodyBase) and issubclass(
            system_two_class, SurfaceBase
        ):
            # rigid body-surface contact
            _apply_contact = self._apply_contact_forces_rigid_body_surface

        elif issubclass(system_one_class, SurfaceBase) and issubclass(
            system_two_class, RigidBodyBase
        ):
            # surface contact-rigid body contact
            raise TypeError(
                r"To add contact the systems must follow this order: rod, rigidbody, surface"
            )

        self._apply_contact = _apply_contact

    def apply_contact(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
    ):
        self._apply_contact(system_one, system_two)


class SelfContact(NoContact):
    """
    This class is modeling self contact of rod.

    """

    def __init__(self, k, nu):
        super(SelfContact, self).__init__()
        self.k = k
        self.nu = nu

    def _calculate_contact_forces_self_rod(self, system_one, system_two):
        _calculate_contact_forces_self_rod_numba(
            system_one.position_collection[
                ..., :-1
            ],  # Discount last node, we want element start position
            system_one.radius,
            system_one.lengths,
            system_one.tangents,
            system_one.velocity_collection,
            system_one.external_forces,
            self.k,
            self.nu,
        )

    def _generate_contact_function(self, system_one, system_two):
        self._apply_contact = self._calculate_contact_forces_self_rod

    def apply_contact(self, system_one: RodType, system_two: SystemType):
        # del index_one, index_two

        self._apply_contact(system_one, system_two)
