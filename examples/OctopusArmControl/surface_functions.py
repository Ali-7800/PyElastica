import numpy as np
from numba import njit
import os
import sys

sys.path.append("../../")
from matplotlib.tri.triangulation import Triangulation


def calculate_facet_normals_centers(
    facets,
    **kwargs,
):
    """
    This function finds the normal vector pointing upwards for each facet in the surface
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

    assert (
        "up_direction" in kwargs.keys() or "facet_vertex_normals" in kwargs.keys()
    ), "Please provide valid up_direction or vertices_normals"
    assert ~(
        "up_direction" in kwargs.keys() and "facet_vertex_normals" in kwargs.keys()
    ), "Please provide one of up_direction or vertices_normals not both"

    if "up_direction" in kwargs.keys():
        for i in range(1, n_facets):
            up_direction = kwargs["up_direction"]
            facets_centers[:, i] = np.mean(
                facets[:, :, i], axis=1
            )  # center of the facet vertices
            # find the normal vector by taking the cross product using two facet sides
            facets_normals[:, i] = np.cross(
                facets[:, 2, i] - facets[:, 1, i], facets[:, 2, i] - facets[:, 0, i]
            )
            if (
                np.linalg.norm(facets_normals[:, i]) < 1e-16
            ):  # if normal has zero magintude set it to point up
                facets_normals[:, i] = up_direction
            else:
                facets_normals[:, i] /= np.linalg.norm(facets_normals[:, i])
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
        for i in range(1, n_facets):
            facets_centers[:, i] = np.mean(
                facets[:, :, i], axis=1
            )  # center of the facet vertices
            # find the normal vector by taking the cross product using two facet sides
            facets_normals[:, i] = np.cross(
                facets[:, 2, i] - facets[:, 1, i], facets[:, 2, i] - facets[:, 0, i]
            )
            if (
                np.linalg.norm(facets_normals[:, i]) < 1e-16
            ):  # if normal has zero magintude set it to point up
                facets_normals[:, i] = facet_vertex_normals[:, 0, i]
            else:
                facets_normals[:, i] /= np.linalg.norm(facets_normals[:, i])
            dotproduct = np.dot(
                facet_vertex_normals[:, 0, i], facets_normals[:, i]
            )  # check normal vector orientation with respect to one of the vertex normals
            if dotproduct < 0:
                facets_normals[:, i] = -facets_normals[:, i]

    return facets_normals, facets_centers


def import_surface_from_obj(
    model_path,
    max_extent=3.0,
    max_z=0.0,
    surface_reorient=[[1, 2], [2, 1]],
    povray_viz=True,
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
    faces = []

    for line in surface_model_text.splitlines():
        if len(line) > 0:
            if line[0] == "v" and line[1] == " ":
                vertex = []
                for number in line[2:].split(" "):
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
            elif line[0] == "f":
                block_line = []
                for block in line[2:].split(" "):
                    vertex = []
                    for number in block.split("/"):
                        vertex.append(int(number) - 1)
                    block_line.append(vertex)
                faces.append(block_line)

    vertices = np.array(vertices)
    scale_value = np.maximum(np.max(vertices), -np.min(vertices)) / max_extent
    vertices /= scale_value  # this scales the model such that the coordinates of all points is less than or equal to max_extent
    texture_vertices = np.array(texture_vertices)
    vertices_normals = np.array(vertices_normals)
    faces = np.array(faces)
    n_faces = faces.shape[0]

    # reorient surface
    vertices[:, surface_reorient[0]] = vertices[:, surface_reorient[1]]
    vertices_normals[:, surface_reorient[0]] = vertices_normals[:, surface_reorient[1]]

    # set maximum z value
    vertices[:, 2] += -np.max(vertices[:, 2]) + max_z

    facets = np.zeros((3, 3, n_faces))
    facet_vertex_normal = np.zeros((3, 3, n_faces))
    uv_vectors = np.zeros((2, 3, n_faces))

    for i in range(n_faces):
        for j in range(3):
            facets[:, j, i] = vertices[faces[i, j, 0]]
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
                + ">"
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

        mesh_list.append(
            '\ntexture {\npigment {\nuv_mapping \nimage_map {\njpeg "'
            + model_path
            + '/texture.jpg" \nmap_type 0\n}\n}'
        )  # add color
        if os.path.exists(model_path + "/normal.jpg"):
            mesh_list.append(
                '\nnormal {\nuv_mapping \nbump_map { \njpeg "'
                + model_path
                + '/normal.jpg" \nmap_type 0\nbump_size 50\n}\n}'
            )  # add normals
        mesh_list.append("\n}\n}")

        mesh_str = "".join(mesh_list)

        if os.path.exists("surface.inc"):
            os.remove("surface.inc")

        surface_inc = open("surface.inc", "w")
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


@njit(cache=True)
def surface_grid_numba(
    facets, grid_size, facet_x_left, facet_x_right, facet_y_down, facet_y_up
):
    x_min = np.min(facets[0, :, :])
    y_min = np.min(facets[1, :, :])
    n_x_positions = int(np.ceil((np.max(facets[0, :, :]) - x_min) / grid_size))
    n_y_positions = int(np.ceil((np.max(facets[1, :, :]) - y_min) / grid_size))
    facets_grid = dict()
    for i in range(n_x_positions):
        x_left = x_min + (i * grid_size)
        x_right = x_min + ((i + 1) * grid_size)
        for j in range(n_y_positions):
            y_down = y_min + (j * grid_size)
            y_up = y_min + ((j + 1) * grid_size)
            if np.any(
                np.where(
                    (
                        (facet_y_down > y_up)
                        + (facet_y_up < y_down)
                        + (facet_x_right < x_left)
                        + (facet_x_left > x_right)
                    )
                    == 0
                )[0]
            ):
                facets_grid[(i, j)] = np.where(
                    (
                        (facet_y_down > y_up)
                        + (facet_y_up < y_down)
                        + (facet_x_right < x_left)
                        + (facet_x_left > x_right)
                    )
                    == 0
                )[0]
    return facets_grid


def surface_grid(facets, grid_size):
    facet_x_left = np.min(facets[0, :, :], axis=0)
    facet_y_down = np.min(facets[1, :, :], axis=0)
    facet_x_right = np.max(facets[0, :, :], axis=0)
    facet_y_up = np.max(facets[1, :, :], axis=0)
    print("Creating Grid ...")
    facets_grid = dict(
        surface_grid_numba(
            facets, grid_size, facet_x_left, facet_x_right, facet_y_down, facet_y_up
        )
    )
    print("Grid Created!")

    return facets_grid