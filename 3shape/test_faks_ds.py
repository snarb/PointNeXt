import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

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

for i in range(n_planes):
    # euler_angle = np.random.uniform(-5, 5, size=3)
    #
    # # Convert to quaternion
    # rotation = R.from_euler('xyz', euler_angle, degrees=True)
    # quaternion = rotation.as_quat()
    #
    # # Convert to normal
    # normal = R.from_quat(quaternion).apply([1, 0, 0])  # assuming initial normal was [1, 0, 0]
    #
    # #normal = np.random.randn(3)
    # normal /= np.linalg.norm(normal)
    # # If the z-component is negative, flip the normal
    # if normal[2] < 0:
    #     normal = -normal
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

    #mesh = create_plane_mesh(origin, normal, size=5.0)

    # Display it
    #o3d.visualization.draw_geometries([mesh])
    #draw_plane_with_normal(points, origin, normal, target_plane = mesh)
    g = 2
    # Save the point cloud, the origin, and the normal
    np.savez(f'plane_{i}.npz', points=points, origin=origin, normal=normal)
