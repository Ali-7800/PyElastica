__doc__ = """ surface classes and implementation details """


import numpy as np
from elastica.surface.surface_base import SurfaceBase
from numpy.testing import assert_allclose
from elastica.utils import Tolerance
from numba import njit


class MeshSurface(SurfaceBase):
    """
    Mesh surface initializer.
        Attributes
        ----------
        mesh: mesh class object
    """

    def __init__(
        self,
        mesh,
    ):
        self.n_faces = mesh.face_normals.shape[-1]
        for face in range(self.n_faces):
            try:
                assert_allclose(
                    np.linalg.norm(mesh.face_normals[:, face]),
                    1.0,
                    atol=Tolerance.atol(),
                    err_msg="face " + str(face) + "'s normal is not a unit vector",
                )
            except AssertionError:
                mesh.face_normals[:, face] = mesh.face_normals[:, face - 1]
        self.face_normals = mesh.face_normals
        self.face_centers = mesh.face_centers
        self.faces = mesh.faces
