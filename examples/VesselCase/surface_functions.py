import numpy as np
from numba import njit
from tqdm import tqdm
import os
import sys

sys.path.append("../../")
from matplotlib.tri.triangulation import Triangulation
from elastica._linalg import (
    _batch_norm,
)


def calculate_facet_normals_centers_areas(
    facets,
    **kwargs,
):
    """
    This function the normal vector pointing upwards for each facet in the surface
    Parameters
    ----------
    facets (3,3,dim)
    up_direction (3,1)

    Returns
    -------
    facet_normals (3,dim)

    Notes
    -----
    """
    n_facets = facets.shape[-1]
    facets_normals = np.zeros((3, n_facets))
    facets_centers = np.zeros((3, n_facets))
    face_areas = np.zeros((n_facets,))

    assert (
        "up_direction" in kwargs.keys() or "facet_vertex_normals" in kwargs.keys()
    ), "Please provide valid up_direction or vertices_normals"
    assert ~(
        "up_direction" in kwargs.keys() and "facet_vertex_normals" in kwargs.keys()
    ), "Please provide one of up_direction or vertices_normals not both"

    if "up_direction" in kwargs.keys():
        for i in range(n_facets):
            up_direction = kwargs["up_direction"]
            facets_centers[:, i] = np.mean(
                facets[:, :, i], axis=1
            )  # center of the facet vertices
            # find the normal vector by taking the cross product using two facet sides
            facets_normals[:, i] = np.cross(
                facets[:, 2, i] - facets[:, 1, i], facets[:, 2, i] - facets[:, 0, i]
            )
            face_areas[i] = np.linalg.norm(facets_normals[:, i])
            if face_areas[i] < 1e-16:  # if normal has zero magintude set it to point up
                facets_normals[:, i] = up_direction
            else:
                facets_normals[:, i] /= face_areas[i]
            dotproduct = np.dot(
                up_direction, facets_normals[:, i]
            )  # check normal vector orientation with respect to the up_direction
            if dotproduct < 0:
                facets_normals[:, i] = -facets_normals[:, i]
    else:
        facet_vertex_normals = kwargs["facet_vertex_normals"]
        assert (
            facet_vertex_normals.shape[-1] == n_facets
        ), "Number of facets in facet_vertex_normals is not equal to the number of facets"
        for i in range(n_facets):
            facets_centers[:, i] = np.mean(
                facets[:, :, i], axis=1
            )  # center of the facet vertices
            # find the normal vector by taking the cross product using two facet sides
            facets_normals[:, i] = np.cross(
                facets[:, 2, i] - facets[:, 1, i], facets[:, 2, i] - facets[:, 0, i]
            )
            face_areas[i] = np.linalg.norm(facets_normals[:, i])
            if (
                np.linalg.norm(facets_normals[:, i]) < 1e-16
            ):  # if normal has zero magintude set it to point up
                facets_normals[:, i] = facet_vertex_normals[:, 0, i]
            else:
                facets_normals[:, i] /= face_areas[i]
            dotproduct = np.dot(
                facet_vertex_normals[:, 0, i], facets_normals[:, i]
            )  # check normal vector orientation with respect to one of the vertex normals
            if dotproduct < 0:
                facets_normals[:, i] = -facets_normals[:, i]

    return facets_normals, facets_centers, face_areas


def import_surface_from_obj(
    model_path,
    max_extent=3.0,
    max_z=0.0,
    surface_reorient=[[1, 2], [2, 1]],
    normals_invert=False,
    povray_viz=True,
    with_texture=False,
):
    """
    This function imports obj, returns facets, and exports povray surface mesh
    Parameters
    ----------
    model_filename
    texture_img_filename
    normal_img_filename
    max_extent
    max_z
    surface_reorient
    povray_viz


    Returns
    -------
    facets
    facet_centers
    surface.inc

    Notes
    -----
    """

    print("Importing Surface ...")
    surface_model = open(model_path + "/model.obj", "r")
    surface_model_text = surface_model.read()
    surface_model.close()
    vertices = []
    texture_vertices = []
    vertices_normals = []
    n_texture_file = 0
    mtl_dict = {}
    faces = []
    facet_texture_index = []

    for line in surface_model_text.splitlines():
        if len(line) > 0:
            if line[0] == "v" and line[1] == " ":
                vertex = []
                for number in line[3:].split(" "):
                    vertex.append(float(number))

                vertices.append(vertex)
            elif line[0] == "v" and line[1] == "t":
                vertex = []
                for number in line[3:].split(" "):
                    vertex.append(float(number))
                texture_vertices.append(vertex)
            elif line[0] == "v" and line[1] == "n":
                vertex = []
                for number in line[3:].split(" "):
                    vertex.append(float(number))
                vertices_normals.append(vertex)
            elif line[0] == "u":  # new material
                n_texture_file += 1
                mtl_dict[n_texture_file] = line[7:]  # store material name for later
            elif line[0] == "f":
                block_line = []
                for block in line[2:].split(" "):
                    vertex = []
                    for number in block.split("/"):
                        try:
                            vertex.append(int(number) - 1)
                        except:
                            vertex.append(0)
                    if len(vertex) == 3:
                        block_line.append(vertex)
                faces.append(block_line)
                facet_texture_index.append(
                    n_texture_file - 1
                )  # include this to figure out where each texture this face is mapped to

    vertices = np.array(vertices)
    scale_value = np.maximum(np.max(vertices), -np.min(vertices)) / max_extent
    vertices /= scale_value  # this scales the model such that the coordinates of all points is less than or equal to max_extent
    texture_vertices = np.array(texture_vertices)
    vertices_normals = np.array(vertices_normals)
    if normals_invert:
        vertices_normals *= -1  # some meshes have opposite normal directions
    faces = np.array(faces)
    n_faces = faces.shape[0]
    print(1 / scale_value)

    # reorient surface
    vertices[:, surface_reorient[0]] = vertices[:, surface_reorient[1]]
    vertices_normals[:, surface_reorient[0]] = vertices_normals[:, surface_reorient[1]]

    # center mesh at 0,0,0
    mesh_center = (
        np.array(
            [
                np.max(vertices[:, 0]) + np.min(vertices[:, 0]),
                np.max(vertices[:, 1]) + np.min(vertices[:, 1]),
                np.max(vertices[:, 2]) + np.min(vertices[:, 2]),
            ]
        )
        / 2
    )
    vertices -= mesh_center
    # set maximum z value
    vertices[:, 2] += -np.max(vertices[:, 2]) + max_z
    print(
        np.array(
            [
                np.max(vertices[:, 0]) + np.min(vertices[:, 0]),
                np.max(vertices[:, 1]) + np.min(vertices[:, 1]),
                np.max(vertices[:, 2]) + np.min(vertices[:, 2]),
            ]
        )
        / 2
    )

    facets = np.zeros((3, 3, n_faces))  # n_coordinates #n_vertices #n_faces
    facet_vertex_normal = np.zeros((3, 3, n_faces))
    uv_vectors = np.zeros((2, 3, n_faces))

    for i in range(n_faces):
        for j in range(3):
            facets[:, j, i] = vertices[faces[i, j, 0]]
            if len(texture_vertices) > 0:
                texture_vertex = texture_vertices[faces[i, j, 1]]
                uv_vectors[:, j, i] = texture_vertex
            facet_vertex_normal[:, j, i] = vertices_normals[faces[i, j, 2]]

    print("Surface imported!")

    if povray_viz:
        print("Exporting POVray Surface Mesh ...")
        mesh_list = []
        mesh_list.append("mesh2 {")
        n_vertices = vertices.shape[0]
        n_vertices_normal = vertices_normals.shape[0]
        n_texture_vertices = texture_vertices.shape[0]

        mesh_list.append("\nvertex_vectors {")
        mesh_list.append("\n" + str(n_vertices))
        for i in range(n_vertices):
            mesh_list.append(
                ",<"
                + str(vertices[i, 0])
                + ","
                + str(vertices[i, 1])
                + ","
                + str(vertices[i, 2])
                + ">"
            )
        mesh_list.append("\n}")

        mesh_list.append("\nnormal_vectors {")
        mesh_list.append("\n" + str(n_vertices_normal))
        for i in range(n_vertices_normal):
            mesh_list.append(
                ",<"
                + str(vertices_normals[i, 0])
                + ","
                + str(vertices_normals[i, 1])
                + ","
                + str(vertices_normals[i, 2])
                + ">"
            )
        mesh_list.append("\n}")

        mesh_list.append("\nuv_vectors {")
        mesh_list.append("\n" + str(n_texture_vertices))
        for i in range(n_texture_vertices):
            mesh_list.append(
                ",<"
                + str(texture_vertices[i, 0])
                + ","
                + str(texture_vertices[i, 1])
                + ">"
            )
        mesh_list.append("\n}")

        mesh_list.append("\ntexture_list {")
        mesh_list.append("\n" + str(n_texture_file) + ",")
        if ~with_texture:
            mesh_list.append(
                "\ntexture {\npigment { color rgbt <1,1,0,0.75>}\n}"
            )  # add color
        else:
            for i in range(n_texture_file):
                mesh_list.append(
                    '\ntexture {\npigment {\nuv_mapping \nimage_map {\njpeg "'
                    + model_path
                    + "/"
                    + str(mtl_dict[i + 1])
                    + '_diffuse.jpg" \nmap_type 0\n}\n}'
                )  # add color
                if os.path.exists(
                    model_path + "/" + str(mtl_dict[i + 1]) + "_normal.jpg"
                ):
                    mesh_list.append(
                        '\nnormal {\nuv_mapping \nbump_map { \njpeg "'
                        + model_path
                        + "/"
                        + str(mtl_dict[i + 1])
                        + '_normal.jpg" \nmap_type 0\nbump_size 50\n}\n}'
                    )  # add normals
                mesh_list.append("\n}")
        mesh_list.append("\n}")

        mesh_list.append("\nface_indices {")
        mesh_list.append("\n" + str(n_faces))
        for i in range(n_faces):
            mesh_list.append(
                ",<"
                + str(faces[i, 0, 0])
                + ","
                + str(faces[i, 1, 0])
                + ","
                + str(faces[i, 2, 0])
                + ">,"
                + str(facet_texture_index[i])
            )
        mesh_list.append("\n}")

        mesh_list.append("\nuv_indices {")
        mesh_list.append("\n" + str(n_faces))
        for i in range(n_faces):
            mesh_list.append(
                ",<"
                + str(faces[i, 0, 1])
                + ","
                + str(faces[i, 1, 1])
                + ","
                + str(faces[i, 2, 1])
                + ">"
            )
        mesh_list.append("\n}")

        mesh_list.append("\nnormal_indices {")
        mesh_list.append("\n" + str(n_faces))
        for i in range(n_faces):
            mesh_list.append(
                ",<"
                + str(faces[i, 0, 2])
                + ","
                + str(faces[i, 1, 2])
                + ","
                + str(faces[i, 2, 2])
                + ">"
            )
        mesh_list.append("\n}")

        mesh_list.append("\n}")

        mesh_str = "".join(mesh_list)

        if os.path.exists(model_path + "/surface.inc"):
            os.remove(model_path + "/surface.inc")

        surface_inc = open(model_path + "/surface.inc", "w")
        surface_inc_text = "\n" + mesh_str
        a = surface_inc.write(surface_inc_text)
        surface_inc.close()
        print("POVray Surface Mesh Exported!")

    return facets, facet_vertex_normal


def create_surface_from_parameterization(
    x,
    y,
    z,
    color_map,
    povray_viz=True,
):
    """
    This function returns facets and exports povray surface mesh from surface parameterization
    Parameters
    ----------
    x
    y
    z
    povray_viz

    Returns
    -------
    facets
    surface.inc

    Notes
    -----
    """

    print("Creating Facets ...")

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(x, y, z)
    triangles = tri.get_masked_triangles()
    xt = tri.x[triangles][..., np.newaxis]
    yt = tri.y[triangles][..., np.newaxis]
    zt = z[triangles][..., np.newaxis]
    facets = np.moveaxis(np.concatenate((xt, yt, zt), axis=2), [0, 2], [-1, 0])
    n_facets = facets.shape[-1]

    print("Facets Created!")

    if povray_viz:
        print("Exporting POVray Surface Mesh ...")
        mesh_list = []
        color_list = []
        for i in range(n_facets):
            color = color_map(np.mean(facets[:, :, i], axis=1))
            mesh_list.append("triangle {")
            for j in range(3):
                if j < 2:
                    mesh_list.append(
                        "<"
                        + str(facets[0, j, i])
                        + ","
                        + str(facets[1, j, i])
                        + ","
                        + str(facets[2, j, i])
                        + ">,"
                    )
                else:
                    mesh_list.append(
                        "<"
                        + str(facets[0, j, i])
                        + ","
                        + str(facets[1, j, i])
                        + ","
                        + str(facets[2, j, i])
                        + ">"
                    )

            mesh_list.append(" texture { Color" + str(i) + " } }\n")
            color_list.append(
                "#declare Color"
                + str(i)
                + " = texture {\n pigment { color rgb<"
                + str(color[0])
                + ","
                + str(color[1])
                + ","
                + str(color[2])
                + "> }\nfinish { phong 0.2 }}\n"
            )

        mesh_str = "".join(mesh_list)
        color_str = "".join(color_list)

        if os.path.exists("surface.inc"):
            os.remove("surface.inc")

        surface_inc = open("surface.inc", "w")
        surface_inc_text = color_str + "\n" + mesh_str
        a = surface_inc.write(surface_inc_text)
        surface_inc.close()
        print("POVray Surface Mesh Exported!")

    return facets


# @njit(cache=True)
# def surface_grid_numba(
#     faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up
# ):
#     x_min = np.min(faces[0, :, :])  # grid x zero position
#     y_min = np.min(faces[1, :, :])  # grid y zero position
#     n_x_positions = int(
#         np.ceil((np.max(faces[0, :, :]) - x_min) / grid_size)
#     )  # number of grid sizes that fit in x direction
#     n_y_positions = int(
#         np.ceil((np.max(faces[1, :, :]) - y_min) / grid_size)
#     )  # number of grid sizes that fit in y direction
#     faces_grid = dict()
#     for i in range(n_x_positions):
#         x_left = x_min + (i * grid_size)  # current grid square left x coordinate
#         x_right = x_min + (
#             (i + 1) * grid_size
#         )  # current grid square right x coordinate
#         for j in range(n_y_positions):
#             y_down = y_min + (j * grid_size)  # current grid square down y coordinate
#             y_up = y_min + ((j + 1) * grid_size)  # current grid square up y coordinate
#             if (
#                 len(
#                     np.where(
#                         (
#                             (
#                                 face_y_down > y_up
#                             )  # if face_y_down coordinate is greater than grid square up position then face is above grid square
#                             + (
#                                 face_y_up < y_down
#                             )  # if face_y_up coordinate is lower than grid square down position then face is below grid square
#                             + (
#                                 face_x_right < x_left
#                             )  # if face_x_right coordinate is lower than grid square left position then face is to the left of the grid square
#                             + (
#                                 face_x_left > x_right
#                             )  # if face_x_left coordinate is greater than grid square right position then face is to the right of the grid square
#                         )
#                         == 0  # if the face is not below, above, to the right of or, to the left of the grid then they must intersect
#                     )[0]
#                 )
#                 > 0
#             ):
#                 faces_grid[(i, j)] = np.where(
#                     (
#                         (face_y_down > y_up)
#                         + (face_y_up < y_down)
#                         + (face_x_right < x_left)
#                         + (face_x_left > x_right)
#                     )
#                     == 0
#                 )[0]
#     return faces_grid


@njit(cache=True)
def surface_grid_numba(
    faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up
):
    x_min = np.min(faces[0, :, :])  # grid x zero position
    y_min = np.min(faces[1, :, :])  # grid y zero position
    n_x_positions = int(
        np.ceil((np.max(faces[0, :, :]) - x_min) / grid_size)
    )  # number of grid sizes that fit in x direction
    n_y_positions = int(
        np.ceil((np.max(faces[1, :, :]) - y_min) / grid_size)
    )  # number of grid sizes that fit in y direction
    faces_grid = dict()
    for i in range(n_x_positions):
        x_left = x_min + (
            max(0, i - 1) * grid_size
        )  # current grid square left x coordinate
        x_right = x_min + (
            min(n_x_positions - 1, i + 1) * grid_size
        )  # current grid square right x coordinate
        for j in range(n_y_positions):
            y_down = y_min + (
                max(0, j - 1) * grid_size
            )  # current grid square down y coordinate
            y_up = y_min + (
                min(n_y_positions - 1, j + 1) * grid_size
            )  # current grid square up y coordinate
            if (
                len(
                    np.where(
                        (
                            (
                                face_y_down > y_up
                            )  # if face_y_down coordinate is greater than grid square up position then face is above grid square
                            + (
                                face_y_up < y_down
                            )  # if face_y_up coordinate is lower than grid square down position then face is below grid square
                            + (
                                face_x_right < x_left
                            )  # if face_x_right coordinate is lower than grid square left position then face is to the left of the grid square
                            + (
                                face_x_left > x_right
                            )  # if face_x_left coordinate is greater than grid square right position then face is to the right of the grid square
                        )
                        == 0  # if the face is not below, above, to the right of or, to the left of the grid then they must intersect
                    )[0]
                )
                > 0
            ):
                faces_grid[(i, j)] = np.where(
                    (
                        (face_y_down > y_up)
                        + (face_y_up < y_down)
                        + (face_x_right < x_left)
                        + (face_x_left > x_right)
                    )
                    == 0
                )[0]
    return faces_grid


def surface_grid(faces, grid_size):
    face_x_left = np.min(faces[0, :, :], axis=0)
    face_y_down = np.min(faces[1, :, :], axis=0)
    face_x_right = np.max(faces[0, :, :], axis=0)
    face_y_up = np.max(faces[1, :, :], axis=0)
    print("Creating Grid ...")
    facets_grid = dict(
        surface_grid_numba(
            faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up
        )
    )
    print("Grid Created!")

    return facets_grid
