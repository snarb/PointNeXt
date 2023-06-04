import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
from openpoints.utils.utils_3shape import *

# Plane parameters
width = 10
height = 10
thickness = 1

# Number of points and planes
n_points = 10000
n_planes = 100
MAX_ROT_ANGLE = 35


def create_plane_mesh(origin, normal, size=1.0, resolution=10):
    # Create a grid of points in the xy plane
    x = np.linspace(-size / 2, size / 2, resolution)
    y = np.linspace(-size / 2, size / 2, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Stack the coordinates into a (N, 3) array of 3D positions
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Compute rotation from [0, 0, 1] to the given normal
    default_normal = np.array([0, 0, 1])
    rotation_vector = np.cross(default_normal, normal)
    rotation_angle = np.arccos(np.dot(default_normal, normal))
    rotation = R.from_rotvec(rotation_angle * rotation_vector)

    # Rotate the positions so they lie in the defined plane
    positions = rotation.apply(positions)

    # Translate the positions so they are centered at the origin
    positions += origin

    # Create the Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Create triangles
    triangles = [[i + j * resolution, i + j * resolution + 1, (i + 1) + (j + 1) * resolution] for j in
                 range(resolution - 1) for i in range(resolution - 1)]
    triangles += [[i + j * resolution + 1, (i + 1) + j * resolution + 1, (i + 1) + (j + 1) * resolution] for j in
                  range(resolution - 1) for i in range(resolution - 1)]

    # Create the Open3D triangle mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh
def draw_plane_with_normal(points, origin, normal, length=2, target_plane = None):
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create line set for normal
    line_points = [origin, origin + length * normal]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # Visualize
    if target_plane is None:
        o3d.visualization.draw_geometries([pcd, line_set])
    else:
        o3d.visualization.draw_geometries([target_plane, pcd, line_set])


def gen_plane(dx, dy, delta = 0.01):
    angle_x = np.random.uniform(dx - delta, dx + delta)
    rotation_x = R.from_euler('x', angle_x, degrees=True)
    # Generate a random rotation within a range around the y-axis
    angle_y = np.random.uniform(dy- delta, dy + delta)
    rotation_y = R.from_euler('y', angle_y, degrees=True)
    # Combine the rotations
    rotation = rotation_x * rotation_y
    # Apply the rotation to the initial normal
    normal = rotation.apply([1, 0, 0])
    # Generate a random point within a bounding box
    origin = np.random.uniform(-10, 10, size=3)
    return normal, origin

def create_homography(rotation_matrix, translation_vector):
    homography = np.eye(4)
    homography[:3, :3] = rotation_matrix
    homography[:3, 3] = translation_vector
    return homography

for i in range(n_planes):
    normal, origin = gen_plane(1, 1, 0.01)

    # Generate two orthogonal vectors in the plane
    v1 = np.random.randn(3)
    v1 -= v1.dot(normal) * normal  # make it orthogonal to normal
    v1 /= np.linalg.norm(v1)  # normalize it
    v2 = np.cross(normal, v1)  # second vector in the plane

    # Generate random coefficients for v1 and v2 within the plane bounds
    c1 = np.random.uniform(-width / 2, width / 2, n_points)
    c2 = np.random.uniform(-height / 2, height / 2, n_points)
    c3 = np.random.uniform(-thickness / 2, thickness / 2, n_points)

    # Generate points in the plane with some thickness
    points = (origin + c1[:, np.newaxis] * v1 + c2[:, np.newaxis] * v2
              + c3[:, np.newaxis] * normal)




    rot_matr = normal_to_rotation_matrix(normal)
    ret_norm = rotation_matrix_to_normal(torch.tensor(rot_matr))
    assert torch.allclose(ret_norm, torch.tensor(normal))

    hom1 = create_homography(rot_matr, origin)

    normal2, origin2 = gen_plane(1, 20, 0.01)
    rot_matr2 = normal_to_rotation_matrix(normal2)

    hom2 = create_homography(rot_matr2, origin)
    rotation_mae, translation_mae = homography_mae_robust(torch.tensor(hom1)[None, ...], torch.tensor(hom2)[None, ...])

    rotation_mae2, translation_mae = homography_mae(torch.tensor(hom1)[None, ...], torch.tensor(hom2)[None, ...])
    print(rotation_mae, rotation_mae2)
    #mesh = create_plane_mesh(origin, normal, size=5.0)

    # Display it
    #o3d.visualization.draw_geometries([mesh])
    #draw_plane_with_normal(points, origin, normal, target_plane = mesh)
    g = 2
