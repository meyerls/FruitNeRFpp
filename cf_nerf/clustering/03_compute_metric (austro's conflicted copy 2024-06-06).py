import open3d as o3d
import numpy as np
from hausdorff import hausdorff_distance
import torch
#from chamferdist import ChamferDistance

def load_obj_file(filename):
    vertices = []
    faces = []

    clusters = []
    init_flag = True

    with open(filename, 'r') as obj_file:
        for line in obj_file:
            # Split the line into components
            parts = line.strip().split()

            if len(parts) > 0:
                # new object data starts with 'o'
                if parts[0] == 'o':
                    if init_flag:
                        init_flag = False
                    else:
                        clusters.append(vertices_cluster)
                    vertices_cluster = []
                # Vertex data starts with 'v'
                elif parts[0] == 'v':
                    vertex = list(map(float, parts[1:]))
                    vertices.append(vertex)
                    vertices_cluster.append(vertex)
                # Face data starts with 'f'
                elif parts[0] == 'f':
                    face = [int(index.split('/')[0]) - 1 for index in
                            parts[1:]]  # Subtract 1 because OBJ indices start from 1
                    faces.append(face)
        clusters.append(vertices_cluster)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    cluster_center = []
    cluster_pcd = []
    for c in clusters:
        cluster_center.append(np.asarray(c).mean(axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(c)
        cluster_pcd.append(pcd)

    return mesh, cluster_center, cluster_pcd


def find_color_indices(array, color):
    """
    Find the indices of all elements in a 2D array that match a specific color.

    Parameters:
    array (numpy.ndarray): 2D array with shape (N, 3), where each row is a color.
    color (numpy.ndarray): 1D array with shape (3,) representing the color to find.

    Returns:
    numpy.ndarray: 1D array of indices where the rows of `array` match `color`.
    """
    # Ensure the color is a NumPy array
    color = np.array(color)

    # Compare each row of the array with the color
    matching_indices = np.all(array == color, axis=1)

    # Get the indices of rows that match the color
    indices = np.where(matching_indices)[0]

    return indices


def compute_median_of_pcd(points):
    """
    Compute the median of a list of 3D points.

    Parameters:
    points (numpy.ndarray): 2D array with shape (N, 3), where each row is a 3D point (x, y, z).

    Returns:
    numpy.ndarray: 1D array with shape (3,) representing the median point (x, y, z).
    """
    # Compute the median along each dimension
    median = np.median(points, axis=0)

    return median


pc_path = "/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/debug/outputs/01_apple_tree_1024x1024_#300/cf-nerf/2024-06-01_113540/cf-nerf/instance_clustered.ply"
pcd = o3d.io.read_point_cloud(pc_path)
#pcd.scale(0.5, center=(0, 0, 0))

mesh_path = "/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/clustering/fruits.obj"
# mesh = o3d.io.read_triangle_mesh(mesh_path)

mesh, mesh_cluster_center, mesh_pcd_list = load_obj_file(mesh_path)

pcd_colors = np.asarray(pcd.colors)
unique_colors = np.unique(pcd_colors, axis=0)

pcd_color_center = []
pcd_color_list = []

for color in unique_colors:
    if color.sum() == 0:
        continue
    indices = find_color_indices(array=color, color=pcd_colors)
    color_pcd = pcd.select_by_index(indices=indices)
    color_pcd_mean = compute_median_of_pcd(points=np.asarray(color_pcd.points))
    pcd_color_center.append(color_pcd_mean)
    pcd_color_list.append(color_pcd)

estimated_center = np.vstack(pcd_color_center)
gt_center = np.vstack(mesh_cluster_center)

#chamferDist = ChamferDistance(torch.from_numpy(estimated_center), torch.from_numpy(gt_center))

print(f"Hausdorff distance test: {hausdorff_distance(XA=gt_center, XB=estimated_center, distance='euclidean')}")
#print(f"Chamfer distance test: {chamferDist}")

gt_center_copy = gt_center.copy()

distance_list = []

number_of_correctly_counted_objects = 0
false_positive = 0

for color in unique_colors:
    if color.sum() == 0:
        continue
    indices = find_color_indices(array=color, color=pcd_colors)
    color_pcd = pcd.select_by_index(indices=indices)
    color_pcd_mean = compute_median_of_pcd(points=np.asarray(color_pcd.points))

    distance2centers = np.linalg.norm(gt_center_copy - color_pcd_mean, axis=1)
    nearest_center_id = np.argmin(distance2centers)
    distance2nearest_center = distance2centers[nearest_center_id]

    if distance2nearest_center < 0.05:
        distance_list.append(distance2nearest_center)
        gt_center_copy = np.delete(gt_center_copy, nearest_center_id, axis=0)
        number_of_correctly_counted_objects += 1
    else:
        false_positive += 1

false_negative = gt_center_copy.shape[0]
true_positive = number_of_correctly_counted_objects

number_of_gt_objects = gt_center.shape[0]
Precision = true_positive / (true_positive + false_positive)
Recall = true_positive / (true_positive + false_negative)
F1 = 2 * Precision * Recall / (Precision + Recall)

print("Counting result: {}/{}".format(number_of_correctly_counted_objects, number_of_gt_objects))
print("Precision: {}".format(Precision))
print("Recall: {}".format(Recall))
print("F1: {}".format(F1))

o3d.visualization.draw_geometries([pcd, mesh])
