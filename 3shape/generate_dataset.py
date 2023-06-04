import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os

# Plane parameters
width = 10
height = 10
thickness = 1

# Number of points and planes
n_points = 2000
n_planes = 700
MAX_ROT_ANGLE = 35

# Directories for saving the files
ply_dir = '/media/brans/data/datasets/toy/val/meshes'
txt_dir = '/media/brans/data/datasets/toy/val/planes'

for i in range(n_planes):
    # Generate a random rotation within a range around the x-axis
    angle_x = np.random.uniform(-MAX_ROT_ANGLE, MAX_ROT_ANGLE)
    rotation_x = R.from_euler('x', angle_x, degrees=True)

    # Generate a random rotation within a range around the y-axis
    angle_y = np.random.uniform(-MAX_ROT_ANGLE, MAX_ROT_ANGLE)
    rotation_y = R.from_euler('y', angle_y, degrees=True)

    # Combine the rotations
    rotation = rotation_x * rotation_y

    # Apply the rotation to the initial normal
    normal = rotation.apply([1, 0, 0])

    # Generate a random point within a bounding box
    origin = np.random.uniform(-10, 10, size=3)

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

    # Create a point cloud and save it as a .ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_dir + f'/plane_{i}.ply', pcd)

    # Save the normal and origin as a .txt file
    plane_data = np.vstack((normal, origin))
    plane_data_flat = plane_data.flatten()
    plane_data_str = ','.join(map(str, plane_data_flat))
    with open(txt_dir + f'/plane_{i}.txt', 'w') as f:
        f.write(plane_data_str)