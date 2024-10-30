import numpy as np
import sympy as sp
import sympy.vector as vc
from tqdm import tqdm
from examples.ArtificialMusclesCases.muscle.muscle_utils import *


from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
    _batch_matmul,
)

from elastica.rod.knot_theory import _compute_writhe, _compute_twist


def _get_rotation_matrix2(scale, axis_collection):
    blocksize = axis_collection.shape[1]
    rot_mat = np.empty((3, 3, blocksize))

    for k in range(blocksize):
        v0 = axis_collection[0, k]
        v1 = axis_collection[1, k]
        v2 = axis_collection[2, k]

        theta = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)

        v0 /= theta + 1e-14
        v1 /= theta + 1e-14
        v2 /= theta + 1e-14

        theta *= scale[k]
        u_prefix = np.sin(theta)
        u_sq_prefix = 1.0 - np.cos(theta)

        rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
        rot_mat[1, 1, k] = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2)
        rot_mat[2, 2, k] = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1)

        rot_mat[0, 1, k] = u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[1, 0, k] = -u_prefix * v2 + u_sq_prefix * v0 * v1
        rot_mat[0, 2, k] = -u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[2, 0, k] = u_prefix * v1 + u_sq_prefix * v0 * v2
        rot_mat[1, 2, k] = u_prefix * v0 + u_sq_prefix * v1 * v2
        rot_mat[2, 1, k] = -u_prefix * v0 + u_sq_prefix * v1 * v2

    return rot_mat


def _rotate2(director_collection, scale, axis_collection):
    """
    Does alibi rotations
    https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

    Parameters
    ----------
    director_collection
    scale
    axis_collection

    Returns
    -------

    TODO Finish documentation
    """
    return _batch_matmul(
        director_collection, _get_rotation_matrix2(scale, axis_collection)
    )
    # return _batch_matmul(
    #     _get_rotation_matrix2(scale, axis_collection), director_collection
    # )


def _find_normal_vector(direction, additive):
    vector_other_than_direction_vector = direction + additive
    return (
        vector_other_than_direction_vector
        - vector_other_than_direction_vector.dot(direction) * direction
    )


def np_lambdify(varname, func):
    lamb = sp.lambdify(varname, func, modules=["numpy"])
    if func.is_constant():
        return lambda t: np.full_like(t, lamb(t))
    else:
        return lambda t: lamb(*t)


def get_fiber_geometry_symbolic_representation(
    start_radius_list,
    taper_slope_list,
    direction,
    offset_list,
    turns_per_length_list,
    CCW_list,
):

    # checks
    assert len(start_radius_list) == len(
        taper_slope_list
    ), "slope list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        offset_list
    ), "offset list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        turns_per_length_list
    ), "turns per length list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        CCW_list
    ), "CCW list must be the same size as start radius list"

    s = sp.symbols("s")
    N = vc.CoordSys3D("N")
    start_radius_syms_list = [
        sp.symbols("r_{0}".format(i)) for i in range(len(start_radius_list))
    ]
    k_syms_list = [
        sp.symbols("k_{0}".format(i)) for i in range(len(turns_per_length_list))
    ]
    taper_slope_syms_list = [
        sp.symbols("m_{0}".format(i)) for i in range(len(taper_slope_list))
    ]
    CCW_syms_list = [sp.symbols("CCW_{0}".format(i)) for i in range(len(CCW_list))]
    radius_dict = {}
    k_dict = {}
    direction_dict = {}
    normal_dict = {}
    binormal_dict = {}
    position_dict = {}
    for i in range(len(start_radius_list)):
        radius_dict[i] = start_radius_syms_list[i] + taper_slope_syms_list[i] * s
        k_dict[i] = 2 * np.pi * (2 * CCW_syms_list[i] - 1) * k_syms_list[i]

    position_dict[0] = s * (
        direction[0] * N.i + direction[1] * N.j + direction[2] * N.k
    )  # centerline_f(s,N)
    direction_dict[0] = sp.diff(position_dict[0], s).normalize()
    normal_temp = sp.diff(direction_dict[0], s).normalize()

    # do gram-schmidt to find a non zero normal vector
    if normal_temp.dot(normal_temp) == 0:
        normal_temp = _find_normal_vector(direction_dict[0], N.i).normalize()
    # do again in case the direction vector was a multiple of N.i
    if normal_temp.dot(normal_temp) == 0:
        normal_temp = _find_normal_vector(direction_dict[0], N.j).normalize()

    normal_dict[0] = normal_temp
    binormal_dict[0] = vc.cross((direction_dict[0]), normal_dict[0])

    for i in tqdm(range(len(radius_dict))):
        position_dict[i + 1] = position_dict[i] + (
            radius_dict[i] * sp.cos(k_dict[i] * s + offset_list[i]) * normal_dict[i]
            + radius_dict[i] * sp.sin(k_dict[i] * s + offset_list[i]) * binormal_dict[i]
        )
        direction_dict[i + 1] = sp.diff(position_dict[i + 1], s).normalize()
        normal_dict[i + 1] = sp.diff(direction_dict[i + 1], s).normalize()
        binormal_dict[i + 1] = vc.cross((direction_dict[i + 1]), normal_dict[i + 1])

    final_coil_position_i = sp.lambdify(
        [s, start_radius_syms_list, k_syms_list, taper_slope_syms_list, CCW_syms_list],
        position_dict[i + 1] & N.i,
        "numpy",
    )
    final_coil_position_j = sp.lambdify(
        [s, start_radius_syms_list, k_syms_list, taper_slope_syms_list, CCW_syms_list],
        position_dict[i + 1] & N.j,
        "numpy",
    )
    final_coil_position_k = sp.lambdify(
        [s, start_radius_syms_list, k_syms_list, taper_slope_syms_list, CCW_syms_list],
        position_dict[i + 1] & N.k,
        "numpy",
    )
    position_symbolic = [
        final_coil_position_i,
        final_coil_position_j,
        final_coil_position_k,
    ]
    # centerline_numpy = [sp.lambdify(s,position_dict[0] & N.i,"numpy"),sp.lambdify(s,position_dict[0] & N.j,"numpy"),sp.lambdify(s,position_dict[0] & N.k,"numpy")]

    return position_symbolic


def get_fiber_geometry_based_on_symbolic_representation(
    position_symbolic,
    n_elem,
    start_radius_list,
    taper_slope_list,
    start_position,
    offset_list,
    length,
    turns_per_length_list,
    initial_link_per_fiber_length,
    CCW_list,
    check_twist_difference=True,
):
    # Compute normal, binormal and director collection
    # We need to compute directors because helix is not a straight rod and it has a complex shape.
    position_collection = np.zeros((3, n_elem + 1))
    # centerline_position = np.zeros((3, n_elem + 1))
    director_collection = np.zeros((3, 3, n_elem))
    normal_collection = np.zeros((3, n_elem))
    binormal_collection = np.zeros((3, n_elem))

    # checks
    assert len(start_radius_list) == len(
        taper_slope_list
    ), "slope list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        offset_list
    ), "offset list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        turns_per_length_list
    ), "turns per length list must be the same size as start radius list"
    assert len(start_radius_list) == len(
        CCW_list
    ), "CCW list must be the same size as start radius list"
    final_coil_position_i = position_symbolic[0]
    final_coil_position_j = position_symbolic[1]
    final_coil_position_k = position_symbolic[2]
    # centerline_position_i = centerline_numpy[0]
    # centerline_position_j = centerline_numpy[1]
    # centerline_position_k = centerline_numpy[2]

    curve_angle = np.arange(0, (n_elem + 1)) * length / (n_elem + 1)
    # centerline_position[0, ...] = start_position[0] + centerline_position_i(curve_angle)
    # centerline_position[1, ...] = start_position[1] + centerline_position_j(curve_angle)
    # centerline_position[2, ...] = start_position[2] + centerline_position_k(curve_angle)
    # distance_cumsum = np.cumsum(_batch_norm(centerline_position[..., 1:] - centerline_position[..., :-1]))
    # distance_variable = np.zeros_like(curve_angle)
    # distance_variable[1:] = distance_cumsum
    # distance_variable *= length/distance_cumsum[-1]
    input = (
        curve_angle,
        start_radius_list,
        turns_per_length_list,
        taper_slope_list,
        CCW_list,
    )

    position_collection[0, ...] = start_position[0] + final_coil_position_i(*input)
    position_collection[1, ...] = start_position[1] + final_coil_position_j(*input)
    position_collection[2, ...] = start_position[2] + final_coil_position_k(*input)

    start = position_collection[..., 0]

    # Compute rod tangents using positions
    position_for_difference = position_collection
    position_diff = position_for_difference[..., 1:] - position_for_difference[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths

    for i in range(n_elem):
        # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
        normal_temp = _find_normal_vector(tangents[..., i], np.array([1.0, 0.0, 0.0]))
        # do again in case the direction vector was a multiple of N.i
        if normal_temp.dot(normal_temp) == 0:
            normal_temp = _find_normal_vector(
                tangents[..., i], np.array([0.0, 1.0, 0.0])
            )

        normal_collection[..., i] = normal_temp / np.linalg.norm(normal_temp)
        binormal_collection[..., i] = np.cross(
            tangents[..., i], normal_collection[..., i]
        )
        director_collection[..., i] = np.vstack(
            (
                normal_collection[..., i],
                binormal_collection[..., i],
                tangents[..., i],
            )
        )

    fiber_length = rest_lengths.sum()
    centerline = position_collection.reshape(
        (1, position_collection.shape[0], position_collection.shape[-1])
    )
    coil_writhe = _compute_writhe(centerline)
    normal_collection_twist_calc = normal_collection.reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    total_intrinsic_twist, local_intrinsic_twist = _compute_twist(
        centerline, normal_collection_twist_calc
    )
    intrinsic_link = coil_writhe[0] + total_intrinsic_twist[0]
    print("Intrinsic_link_per_length:" + str((intrinsic_link) / fiber_length))
    injected_twist = (initial_link_per_fiber_length * fiber_length) - intrinsic_link
    # print("Writhe = "+str(coil_writhe[0]))
    # print("Link = "+str(initial_link_per_length*length))
    # print("Intrinsic Twist = "+str(total_intrinsic_twist[0]))
    print("Coil Twist = " + str(injected_twist))

    rotation_angle = (injected_twist * 2 * np.pi) * (curve_angle / curve_angle[-1])

    director_collection[:] = _rotate2(
        director_collection, scale=rotation_angle, axis_collection=tangents
    )
    normal_collection_twist_calc2 = director_collection[0, :, :].reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    post_rotation_twist, local_new_twist = _compute_twist(
        centerline, normal_collection_twist_calc2
    )
    # print("New Twist = "+str(post_rotation_twist[0]))
    print("Overall Twist/length = " + str(post_rotation_twist[0] / fiber_length))
    twist_difference = abs(
        post_rotation_twist[0] - injected_twist - total_intrinsic_twist[0]
    )

    if check_twist_difference:
        assert (
            twist_difference < 1.0
        ), "Not Enough Elements To Capture Twist. Twist Difference = " + str(
            twist_difference
        )

    return (
        fiber_length,
        start,
        position_collection,
        director_collection,
        intrinsic_link,
        injected_twist,
    )


def get_fiber_geometry(
    n_elem,
    # start_radius_list,
    # taper_slope_list,
    radial_profile_list,
    start_position,
    direction,
    normal,
    offset_list,
    length,
    turns_per_length_list,
    initial_link_per_fiber_length,
    CCW_list,
    check_twist_difference=True,
    position_noise_scale=0.0,
):

    # Compute normal, binormal and director collection
    # We need to compute directors because helix is not a straight rod and it has a complex shape.
    position_collection = np.zeros((3, n_elem + 1))
    director_collection = np.zeros((3, 3, n_elem))
    normal_collection = np.zeros((3, n_elem))
    binormal_collection = np.zeros((3, n_elem))

    # # checks
    # assert len(start_radius_list) == len(
    #     taper_slope_list
    # ), "slope list must be the same size as start radius list"
    # assert len(start_radius_list) == len(
    #     offset_list
    # ), "offset list must be the same size as start radius list"
    # assert len(start_radius_list) == len(
    #     turns_per_length_list
    # ), "turns per length list must be the same size as start radius list"
    # assert len(start_radius_list) == len(
    #     CCW_list
    # ), "CCW list must be the same size as start radius list"

    # create symbolic variables
    s = sp.symbols("s")
    N = vc.CoordSys3D("N")
    radius_dict = {}
    k_dict = {}
    direction_dict = {}
    normal_dict = {}
    binormal_dict = {}
    position_dict = {}
    binormal = np.cross(direction, normal)
    # for i in range(len(start_radius_list)):
    for i in range(len(radial_profile_list)):
        # radius_dict[i] = start_radius_list[i] + taper_slope_list[i] * s
        radius_dict[i] = radial_profile_list[i](s)
        k_dict[i] = 2 * np.pi * (2 * int(CCW_list[i]) - 1) * turns_per_length_list[i]

    direction_dict[0] = direction[0] * N.i + direction[1] * N.j + direction[2] * N.k
    normal_dict[0] = normal[0] * N.i + normal[1] * N.j + normal[2] * N.k
    binormal_dict[0] = binormal[0] * N.i + binormal[1] * N.j + binormal[2] * N.k
    position_dict[0] = s * direction_dict[0]

    curve_angle = np.arange(0, (n_elem + 1)) * length / (n_elem + 1)
    # for i in range(n_elem+1):
    #     print(curve_angle[i]/10e-3,sp.N(radial_profile_list[0](curve_angle[i]))/0.000793925)

    for i in tqdm(range(len(radius_dict))):
        position_dict[i + 1] = position_dict[i] + (
            radius_dict[i] * sp.cos(k_dict[i] * s + offset_list[i]) * normal_dict[i]
            + radius_dict[i] * sp.sin(k_dict[i] * s + offset_list[i]) * binormal_dict[i]
        )
        direction_dict[i + 1] = sp.diff(position_dict[i + 1], s)
        normal_dict[i + 1] = sp.diff(direction_dict[i + 1], s).normalize()
        binormal_dict[i + 1] = vc.cross(
            (direction_dict[i + 1]).normalize(), normal_dict[i + 1]
        )

    final_coil_position_i = sp.lambdify(s, position_dict[i + 1] & N.i, "numpy")
    final_coil_position_j = sp.lambdify(s, position_dict[i + 1] & N.j, "numpy")
    final_coil_position_k = sp.lambdify(s, position_dict[i + 1] & N.k, "numpy")

    np.random.seed(1)
    position_collection[0, ...] = (
        start_position[0]
        + final_coil_position_i(curve_angle)
        + position_noise_scale * np.random.uniform(-1.0, 1.0, n_elem + 1)
    )
    position_collection[1, ...] = (
        start_position[1]
        + final_coil_position_j(curve_angle)
        + position_noise_scale * np.random.uniform(-1.0, 1.0, n_elem + 1)
    )
    position_collection[2, ...] = (
        start_position[2]
        + final_coil_position_k(curve_angle)
        + position_noise_scale * np.random.uniform(-1.0, 1.0, n_elem + 1)
    )

    start = position_collection[..., 0]

    # Compute rod tangents using positions
    position_for_difference = position_collection
    position_diff = position_for_difference[..., 1:] - position_for_difference[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths

    for i in range(n_elem):
        # Compute the normal vector at each element. Since we allow helix radius to vary, we need to compute
        # vectors creating normal for each element.
        normal_collection[0, i] = -tangents[1, i]
        normal_collection[1, i] = tangents[0, i]
        normal_collection[..., i] /= np.linalg.norm(normal_collection[..., i])

        binormal_collection[..., i] = np.cross(
            tangents[..., i], normal_collection[..., i]
        )
        director_collection[..., i] = np.vstack(
            (
                normal_collection[..., i],
                binormal_collection[..., i],
                tangents[..., i],
            )
        )

    fiber_length = rest_lengths.sum()
    centerline = position_collection.reshape(
        (1, position_collection.shape[0], position_collection.shape[-1])
    )
    coil_writhe = _compute_writhe(centerline)
    normal_collection_twist_calc = normal_collection.reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    total_intrinsic_twist, local_intrinsic_twist = _compute_twist(
        centerline, normal_collection_twist_calc
    )
    intrinsic_link = coil_writhe[0] + total_intrinsic_twist[0]
    print("Intrinsic_link_per_length:" + str((intrinsic_link) / fiber_length))
    injected_twist = (initial_link_per_fiber_length * fiber_length) - intrinsic_link
    # print("Writhe = "+str(coil_writhe[0]))
    # print("Link = "+str(initial_link_per_length*length))
    # print("Intrinsic Twist = "+str(total_intrinsic_twist[0]))
    print("Coil Twist = " + str(injected_twist))

    rotation_angle = (injected_twist * 2 * np.pi) * (curve_angle / curve_angle[-1])

    director_collection[:] = _rotate2(
        director_collection, scale=rotation_angle, axis_collection=tangents
    )
    normal_collection_twist_calc2 = director_collection[0, :, :].reshape(
        (1, normal_collection.shape[0], normal_collection.shape[-1])
    )
    post_rotation_twist, local_new_twist = _compute_twist(
        centerline, normal_collection_twist_calc2
    )
    # print("New Twist = "+str(post_rotation_twist[0]))
    print("Overall Twist/length = " + str(post_rotation_twist[0] / fiber_length))
    twist_difference = abs(
        post_rotation_twist[0] - injected_twist - total_intrinsic_twist[0]
    )

    if check_twist_difference:
        assert (
            twist_difference < 1.0
        ), "Not Enough Elements To Capture Twist. Twist Difference = " + str(
            twist_difference
        )

    return (
        fiber_length,
        start,
        position_collection,
        director_collection,
        intrinsic_link,
        injected_twist,
    )
