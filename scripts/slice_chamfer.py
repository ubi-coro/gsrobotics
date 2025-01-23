""" 
slice_chamfer.py

Description:
This script applies a chamfer-like slicing approach to an existing point cloud, based on corner/edge geometry obtained from a prior analysis (stored in results_{index}.pkl). 

Core Functionality:
	By interpreting an identified plane (with normal and offset) from the results dictionary, it can:

	    Build a simple 3D “chamfer” point cloud (optional):
		Using build_deburred_edge_point_cloud, generate a synthetic wedge (flat 45° surface) near the original edge location.

	    Segment / Filter the real point cloud above the identified plane:
		plane_from_results computes a plane normal and offset d from the given edge parameters (angle, midpoint, etc.).
		keep_points_above_plane then retains all points for which n^T * x + d ≥ 0, effectively slicing away material below the chamfer plane.

	    Transform the sliced result so that the plane is mapped to z=0:
		transform_plane_to_xy(n, d) builds a 4×4 homogeneous transform that brings the plane n^T x + d=0 into the XY-plane with normal = (0,0,1).
		This step lets you view the remaining cloud in a “flattened” orientation that can simplify further measurement or inspection.

	    Visualization & Saving:
		Plots the partially sliced and transformed cloud for quick verification.
		Saves the filtered cloud to segmented_pcd_{index}.pcd for downstream use.

Usage: 
	python slice_chamfer.py --index 0 --distance 6.0

	Where “--index” determines which results and point cloud to load, and “--distance” sets how far along the identified plane’s normal to offset the slice. The script then outputs the final segmented cloud and displays it in a Matplotlib 3D plot. 
	
Known Bugs:
	For unknown reason, o3d.io.write_point_cloud cannot handle the point cloud after applying the homogenous transformation, so the point cloud before (directly after segmenting) is stored instead.
	
Author/Contact:
    - Jannick Stranghöner (Universität Bielefeld, jannick.stranghoener@uni-bielefeld.de), 2024
    - (No warranties; adapted for demonstration of GelSight 3D capabilities)
"""

#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

def build_deburred_edge_point_cloud(results_dict, d_chamfer=0.5, num_long=50, num_thick=10):
    """
    Build a simple 3D point cloud representing a flat 45°-chamfered surface
    near the original 'sharp' corner, preserving orientation from results_dict.

    Args:
      results_dict: dict with keys:
         - 'edge_location' = (x1, y1, x2, y2) in some 2D coordinate system
         - 'alpha_deg', 'beta_deg', etc. (optional usage here)
      d_chamfer: distance from the sharp edge to the new 45° face
                (in your units, e.g. mm).
      num_long: number of samples along the edge direction
      num_thick: number of samples across the chamfer thickness
    Returns:
      pcd_points: (N,3) float array of 3D points
    """

    (x1, y1, x2, y2) = results_dict['edge_location']
    alpha_deg = results_dict.get('alpha_deg', 90.0)
    beta_deg = results_dict.get('beta_deg', 0.0)
    beta_deg = results_dict.get('beta_deg', 0.0)
    beta_deg = results_dict.get('beta_deg', 0.0)
    beta_deg = results_dict.get('beta_deg', 0.0)

    # 1) Define the edge direction in 2D
    ex_2d = (x2 - x1)
    ey_2d = (y2 - y1)
    length_2d = np.hypot(ex_2d, ey_2d)
    if length_2d < 1e-9:
        raise ValueError("Edge is degenerate (x1,y1) ~ (x2,y2).")

    ex_2d /= length_2d
    ey_2d /= length_2d

    # Midpoint
    mx_2d = 0.5*(x1 + x2)
    my_2d = 0.5*(y1 + y2)

    # In-plane perpendicular direction
    # e_hat = (ex_2d, ey_2d)
    # n_hat = (-ey_2d, ex_2d)
    def to_global_3D(u, v):
        """
        Convert local 2D coords (u along edge, v orthonormal) into 3D.
        Here we treat the 'image plane' as X,Y with Z=0 initially,
        and assign a 45° slope in Z based on v.
        """
        x_img = mx_2d + u*ex_2d + v*(-ey_2d)
        y_img = my_2d + u*ey_2d + v*( ex_2d)
        # Simple 45° slope => Z = v (assuming v is in mm)
        z_val = v  # or v * np.tan(np.radians(45))

        return np.array([x_img, y_img, z_val], dtype=float)

    pcd_points = []
    for i in range(num_long):
        # param s in [0..1] along the edge
        s = i/(num_long - 1)
        u = s * length_2d  # in the same units as the edge_location

        for j in range(num_thick):
            # param w in [0..d_chamfer] across the chamfer
            w = j/(num_thick - 1)*d_chamfer
            pt = to_global_3D(u, w)
            pcd_points.append(pt)

    return np.array(pcd_points, dtype=float)


def plane_from_results(results_dict: dict[str, float], d_chamfer: float):
    (x1, y1, x2, y2) = results_dict['edge_location']  # the plane is the same for all points, so we choose the first of the line
    beta_deg = results_dict.get('beta_deg', 90.0)
    alpha_deg = results_dict.get('alpha_deg', 90.0)
    avg_depth = results_dict.get('avg_depth', 15.0)

    edge_vec = np.array([x2 - x1, y2 - y1, 0.0])
    normal = Rotation.from_rotvec((beta_deg - 90.0) * edge_vec / np.linalg.norm(edge_vec), degrees=True)
    n = normal.as_matrix() @ np.array([0, 0, 1])
    n /= np.linalg.norm(n)  # ensure unit length

    # Next, define point on the plane
    # For example, the midpoint of the edge is (mx, my, 0).
    mx = 0.5*(x1 + x2)
    my = 0.5*(y1 + y2)
    p0 = np.array([mx, my, avg_depth], dtype=float)

    # We want the plane to be offset by +dChamfer along n from that line.
    # So define P = p0 + dChamfer*n.
    # Then the plane equation is n^T x + d = 0, with d = -n^T P.
    P = p0 - d_chamfer * n
    d = -np.dot(n, P)

    return n, d



def plot_plane_and_cloud3d(pcd_points, n, d, plane_center=None, plane_size=100.0):
    """
    Plot a 3D point cloud and a finite patch of the plane n^T x + d = 0 in a matplotlib 3D figure.

    Args:
      pcd_points: (N, 3) array of 3D points.
      n: (3,) plane normal vector (unit or not).
      d: plane offset, so the plane eqn is n^T x + d = 0.
      plane_center: optional, a 3D point on the plane to center the patch
                    (if None, we use the plane's closest point to the origin).
      plane_size: half-size of the plane patch in the two directions orthonormal to n.

    Returns:
      None (displays a matplotlib figure).
    """
    pcd_points = np.asarray(pcd_points)
    if pcd_points.shape[1] != 3:
        raise ValueError("pcd_points must be of shape (N, 3)")

    # 1) Figure out a point on the plane if not provided.
    n = np.array(n, dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        raise ValueError("Plane normal n is zero or invalid.")
    n_unit = n / norm_n

    if plane_center is None:
        # Closest point on plane to origin: X0 = -(d / ||n||^2)*n
        plane_center = -(d / (norm_n**2)) * n  # shape (3,)
        plane_center = np.array([0, 0, 0])

    # 2) Build two orthonormal directions u,v that are perpendicular to n.
    #    A common approach is to take any vector that isn't parallel to n, cross once, cross again.
    #    We'll just do a quick approach using e.g. [1,0,0] or [0,1,0].
    ref = np.array([1, 0, 0], dtype=float)
    if np.abs(n_unit.dot(ref)) > 0.99:
        ref = np.array([0, 1, 0], dtype=float)

    u = np.cross(n_unit, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n_unit, u)

    # 3) Generate a grid of points in [−plane_size..+plane_size] for (s,t).
    num_grid = 20
    min_xyz = pcd_points.min(axis=0)
    max_xyz = pcd_points.max(axis=0)

    s_vals = np.linspace(min_xyz[0], max_xyz[0], num_grid)
    t_vals = np.linspace(min_xyz[1], max_xyz[1], num_grid)
    S, T = np.meshgrid(s_vals, t_vals)  # shape (num_grid, num_grid)

    # 4) For each (s,t), compute the 3D point: plane_center + s*u + t*v
    Xp = plane_center[0] + S*u[0] + T*v[0]
    Yp = plane_center[1] + S*u[1] + T*v[1]
    Zp = plane_center[2] + S*u[2] + T*v[2]

    # 5) Make a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # -- Plot the point cloud
    ax.scatter3D(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2],
                 c='k', marker='.', s=5, label="Point Cloud")

    # -- Plot the plane patch as a surface
    ax.plot_surface(Xp, Yp, Zp, color='r',  # a light green with alpha
                    edgecolor='none')

    # 6) Set axis labels, aspect, etc.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Optionally, compute bounding box from the cloud to set the view nicely
    #ax.set_xlim(min_xyz[0], max_xyz[0])
    #ax.set_ylim(min_xyz[1], max_xyz[1])
    #ax.set_zlim(min_xyz[2], max_xyz[2])

    ax.legend()
    plt.title("Plane & Point Cloud")
    plt.show()

def keep_points_above_plane(pcd_points, n, d):
    """
    Given pcd_points shape (N,3), plane n^T x + d = 0,
    keep only points for which n^T x + d >= 0.
    """
    # n^T x_i + d >= 0
    pcd_points = np.asarray(pcd_points)
    distances = pcd_points @ n + d
    mask = (distances >= 0.0)
    return pcd_points[mask], distances[mask]

def transform_plane_to_xy(n, d):
    """
    Compute a 4x4 transform that maps the plane n^T x + d=0 into z=0
    with plane normal = (0,0,1).
    We'll do it as:
      1) Translate by +d/norm(n)^2 if needed,
      2) Rotate so that n -> (0,0,1).

    Return the 4x4 homogeneous transform T, so new_x = T * old_x.
    """
    # 1) The plane passes through any point X0 s.t. n^T X0 + d=0.
    #    We can pick X0 = - (d/||n||^2)*n (the closest point on plane to origin).
    #    Then we define a translation that moves X0 to origin.
    #    Then we define a rotation that aligns n with the z-axis.

    norm_n = np.linalg.norm(n)
    if norm_n < 1e-9:
        raise ValueError("Plane normal is zero or invalid")

    # The point on the plane closest to origin:
    X0 = -(d / (norm_n**2)) * n

    # Build translation to move X0 -> origin:
    T_trans = np.eye(4)
    T_trans[0:3, 3] = -X0  # translation by -X0

    # Next, rotate n to (0,0,1).
    # We can do this with a small helper function.
    # One typical approach is the Rodrigues formula or aligning two vectors.
    n_unit = n / norm_n

    # If n is already (0,0,1), no rotation needed
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    dot_nz = n_unit.dot(z_axis)

    if np.isclose(dot_nz, 1.0, atol=1e-6):
        # Already aligned
        R = np.eye(3)
    elif np.isclose(dot_nz, -1.0, atol=1e-6):
        # n is (0,0,-1), 180 deg rotation around X or Y
        R = np.eye(3)
        R[0,0] = -1
        R[1,1] = -1
    else:
        # general case: axis = n x z_axis, angle = arccos(n dot z_axis)
        axis = np.cross(n_unit, z_axis)
        axis_len = np.linalg.norm(axis)
        axis /= axis_len
        angle = np.arccos(dot_nz)
        # Rodrigues rotation formula
        K = np.array([
            [0,      -axis[2], axis[1]],
            [axis[2], 0,       -axis[0]],
            [-axis[1], axis[0], 0     ]
        ], dtype=float)
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

    T_rot = np.eye(4)
    T_rot[0:3, 0:3] = R

    # Combine: T = T_rot * T_trans
    T = T_rot @ T_trans
    return T


def main():
    parser = argparse.ArgumentParser(
        description="Build a deburred edge point cloud from results_{index}.pkl."
    )
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="Index of the results file to load, e.g. results_{index}.pkl.")
    parser.add_argument("-d", "--distance", type=float, default=6.0,
                        help="Distance from the original edge to the chamfer (d_chamfer).")
    args = parser.parse_args()

    # 1) Load the pickle file results_{index}.pkl and the original point cloud
    pickle_filename = f'/home/jstranghoener/PycharmProjects/gsrobotics/results/results_{args.index}.pkl'
    print(f"Loading corner parameters from: {pickle_filename}")
    with open(pickle_filename, 'rb') as f:
        results_dict = pickle.load(f)
    pcd = o3d.io.read_point_cloud(f'/home/jstranghoener/PycharmProjects/gsrobotics/captures/pc_{args.index}.pcd')

    # 2) Define the chamfer plane from these results
    n, d = plane_from_results(results_dict, d_chamfer=args.distance)
    print(f"Plane normal = {n}, d = {d}")

    #plot_plane_and_cloud3d(pcd.points, n, d)

    # 3) Plot the cloud
    #pcd_points = build_deburred_edge_point_cloud(results_dict, d_chamfer=args.distance)

    # 4) Keep only points "above" the plane (n^T x + d >= 0)
    pcd_points_filtered, dist_vals = keep_points_above_plane(pcd.points, n, d)
    print(f"Points before filtering: {np.asarray(pcd.points).shape[0]}")
    print(f"Points after filtering:  {pcd_points_filtered.shape[0]}")

    # 5) Compute transformation that maps plane to z=0 with normal=(0,0,1)
    T = transform_plane_to_xy(n, d)

    # 6) Apply T to the filtered points
    #    We'll convert each point to homogeneous coords, multiply, then extract [x,y,z].
    Nf = pcd_points_filtered.shape[0]
    ones_col = np.ones((Nf, 1), dtype=float)
    homog_pts = np.hstack((pcd_points_filtered, ones_col))  # shape (Nf, 4)

    transformed_homog = (T @ homog_pts.T).T  # shape (Nf, 4)
    transformed_xyz = transformed_homog[:, 0:3]

     # 5) Make a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # -- Plot the point cloud
    ax.scatter3D(transformed_xyz[:, 0], transformed_xyz[:, 1], transformed_xyz[:, 2],
                 c='k', marker='.', s=5, label="Point Cloud")

    # 6) Set axis labels, aspect, etc.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Optionally, compute bounding box from the cloud to set the view nicely
    #ax.set_xlim(min_xyz[0], max_xyz[0])
    #ax.set_ylim(min_xyz[1], max_xyz[1])
    #ax.set_zlim(min_xyz[2], max_xyz[2])

    ax.legend()
    plt.title("Plane & Point Cloud")
    plt.show()

    # 7) Save the resulting point cloud
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(np.asarray(pcd_points_filtered))
    output_path = f'/home/jstranghoener/PycharmProjects/gsrobotics/results/segmented_pcd_{args.index}.pcd'
    o3d.io.write_point_cloud(output_path, pcd_out)
    print(f"Saved transformed partial cloud to {output_path}")

if __name__ == "__main__":
    main()
