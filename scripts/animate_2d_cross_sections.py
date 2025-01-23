""" 
animate_2d_cross_sections.py

Description:
This script creates an animated 3D visualization of a “slab” plane sliding along a detected edge in the GelSight-measured point cloud. At each step, it identifies the points lying within a defined thickness on either side of a plane normal to the edge direction, then projects them onto that plane in 3D. The animation updates the positions of:

    The “slab” points in the original cloud (blue),
    The projected points (green),

allowing you to see, frame by frame, how the point cloud cross-section changes as the plane traverses the edge.

Core Functionality:

    Edge & Slab: The script retrieves an (x1,y1,x2,y2) edge from a results dictionary, samples multiple steps along that 2D line, and defines a plane around each step (with user-defined thickness).
    Animation: Uses matplotlib’s FuncAnimation to step through each plane position and update two dynamic scatter plots (slab and projected).
    Point Cloud: A subset of the entire cloud is shown in gray for reference, while the red points indicate the edge itself.

Typical usage: 
	python animate_2d_cross_sections.py --index 0 --thickness_factor 1.0 --num_steps 50

	(where “--index” chooses which point cloud/depth map/results to load, “--thickness_factor” controls the slab thickness, and “--num_steps” how many positions along the edge to animate).

See main() for argument parsing and data loading, and animate_slabs_over_edge(...) for the core animation logic.
	
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
from matplotlib.animation import FuncAnimation

def animate_slabs_over_edge(results_dict, depth_map, pcd,
                            thickness_factor=1.0, num_steps=100):
    """
    Demonstration of an animation that moves a "slab" plane along a 2D edge,
    showing how the points in that slab are projected in 3D.

    Args:
      results_dict (dict): contains 'edge_location' = (x1,y1,x2,y2), possibly angles, etc.
      depth_map (ndarray): 2D array (H,W) with depth values.
      pcd (o3d.geometry.PointCloud): 3D point cloud of the part.
      thickness_factor (float): how thick the plane slab is, relative to edge length/steps.
      num_steps (int): how many steps to sample along the edge from (x1,y1)->(x2,y2).
    """
    # -------------------------------
    # 1) Basic Setup
    # -------------------------------
    # Convert Open3D pcd to a NumPy array
    pcd_points = np.asarray(pcd.points)

    # Edge from results_dict
    (x1, y1, x2, y2) = results_dict['edge_location']
    edge_vec2d = np.array([x2 - x1, y2 - y1])
    edge_length = np.linalg.norm(edge_vec2d)

    # We'll sample 'num_steps' points along that 2D edge
    xs = np.linspace(x1, x2, num_steps, dtype=int)
    ys = np.linspace(y1, y2, num_steps, dtype=int)

    # Decide slab thickness
    thickness = thickness_factor * edge_length / num_steps

    # 3D bounding box for reference
    xyz_min = pcd_points.min(axis=0)
    xyz_max = pcd_points.max(axis=0)

    # -------------------------------
    # 2) Set up a Matplotlib 3D figure for animation
    # -------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    # Plot the entire point cloud once (static)
    ax.scatter(pcd_points[::25, 0],  # subsample if large
               pcd_points[::25, 1],
               pcd_points[::25, 2],
               c='gray', s=2, alpha=0.5, label='Point Cloud')

    # We'll create scatter plots for the dynamic slab and projection
    # but we won't fill them yet. We'll update them in the animation.
    slab_scatter = ax.scatter([], [], [], c='blue', s=20, label='Slab Points')
    proj_scatter = ax.scatter([], [], [], c='green', s=20, label='Projected')

    # We also might show the edge points in red for reference
    edge_x, edge_y, edge_z = [], [], []
    for i in range(num_steps):
        xx = xs[i]
        yy = ys[i]
        zz = depth_map[yy, xx]
        edge_x.append(xx)
        edge_y.append(yy)
        edge_z.append(zz)
    ax.scatter(edge_x, edge_y, edge_z, c='red', s=20, label='Edge')

    # Set the bounding box
    ax.set_xlim([xyz_min[0], xyz_max[0]])
    ax.set_ylim([xyz_min[1], xyz_max[1]])
    ax.set_zlim([xyz_min[2], xyz_max[2]])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title("Sliding Slab Animation")
    ax.legend()

    # -------------------------------
    # 3) Define update function
    # -------------------------------
    def update(frame_index):
        """
        This function is called at each frame of the animation (0..num_steps-1).
        We'll compute the slab for this step, project the points, then update the
        slab_scatter and proj_scatter objects.
        """

        # (x, y, z) at this edge step
        xi = xs[frame_index]
        yi = ys[frame_index]
        zi = depth_map[yi, xi]

        # Plane normal in XY direction only (no slope in z)
        # If you want to incorporate local slope in z, you can do so
        ni = np.array([edge_vec2d[0], edge_vec2d[1], 0.0])
        ni /= np.linalg.norm(ni)

        pi = np.array([xi, yi, zi])

        # 1) Signed distance for each point in pcd to this plane
        diffs = pcd_points - pi
        dist_vals = np.einsum('ij,j->i', diffs, ni)

        # 2) Keep points within +/- thickness
        slab_mask = (np.abs(dist_vals) <= thickness)
        slab_pts = pcd_points[slab_mask]

        # 3) Project slab_pts onto the plane
        diffs_slab = slab_pts - pi
        dot_slab = np.einsum('ij,j->i', diffs_slab, ni)
        x_proj = slab_pts - np.outer(dot_slab, ni)

        # Update the data in the scatter plots
        # We'll clear old data by calling set_offsets or set_data_3d (for 3D).
        # For a 3D scatter, we can do something like:
        if slab_pts.shape[0] > 0:
            slab_scatter._offsets3d = (slab_pts[:, 0],
                                       slab_pts[:, 1],
                                       slab_pts[:, 2])
        else:
            # If no points, set them empty
            slab_scatter._offsets3d = ([], [], [])

        if x_proj.shape[0] > 0:
            proj_scatter._offsets3d = (x_proj[:, 0],
                                       x_proj[:, 1],
                                       x_proj[:, 2])
        else:
            proj_scatter._offsets3d = ([], [], [])

        # You can update the title or any text as well
        ax.set_title(f"Sliding Slab Animation - Step {frame_index+1}/{num_steps}")

        # Return updated artists so FuncAnimation knows what changed
        return slab_scatter, proj_scatter

    # -------------------------------
    # 4) Create the animation
    # -------------------------------
    anim = FuncAnimation(fig,
                         func=update,
                         frames=num_steps,
                         interval=500,   # delay in ms between frames
                         blit=False)     # blit=True can be faster, but trickier for 3D
    plt.show()


def main():
    """
    Main function that:
      1) Parses args
      2) Loads pcd, depth_map, and results dict
      3) Calls animate_slabs_over_edge to show an animation
    """
    parser = argparse.ArgumentParser(description="Animate a sliding slab over the point cloud.")
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="Index for the results/depthmap/pcd naming.")
    parser.add_argument("--thickness_factor", type=float, default=1.0,
                        help="Slab thickness factor relative to edge length/steps.")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps along the edge to animate.")
    args = parser.parse_args()

    # 1) Load data
    pcd_path = f'../captures/pc_{args.index}.pcd'
    depthmap_path = f'../captures/depthmap_{args.index}.npy'
    pickle_path = f'../results/results_{args.index}.pkl'

    print(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"Loading depth map: {depthmap_path}")
    depth_map = np.load(depthmap_path)

    print(f"Loading corner parameters: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        results_dict = pickle.load(f)

    # 2) Animate
    animate_slabs_over_edge(results_dict, depth_map, pcd,
                            thickness_factor=args.thickness_factor,
                            num_steps=args.num_steps)


if __name__ == "__main__":
    main()
