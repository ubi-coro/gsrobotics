""" 
analyze_depthmap.py


Description:
	This script loads a depth map (e.g., from a GelSight sensor) and a corresponding RGB image, detects a prominent edge in the RGB (using Canny plus Hough lines), and analyzes that edge within the depth map. It reports several metrics such as:

    1. The line's location in pixel coordinates.
    2. The angle of the edge in the XY plane (angle_xy_deg).
    3. The average depth along that edge (avg_depth).
    4. A rough angle between left/right planes near the edge (alpha_deg).
    5. The angle of a bisector relative to the GelSight normal (beta_deg).
    6. An approximate depth measured along that bisector (d_bisector).

Core Functionality:
    1. Depth Map Loading: Reads a .npy file containing depth values (float). Converts to 8-bit for edge detection if needed.
    2. Image Loading: Reads an RGB image (PNG/JPG) from disk.
    3. Edge Detection: Applies Canny edge detection, followed by HoughLinesP to find prominent lines.
    4. Edge Analysis: Finds the longest line, samples depths along it, computes average depth, then samples a perpendicular cross-section to estimate angles.
    Visualization (optional): Plots the depth profile, as well as 2D images showing detected edges and line segments.
    5. Finally, the script saves the computed metrics into a pickle file (results_{index}.pkl) for later use or inspection.

Usage: 

	python analyze_depthmap.py --plot -i 0

	(where -i selects an index for loading depthmap_{index}.npy and img_{index}.png, and --plot activates plotting for debugging/visualization).


Author/Contact:
    - Jannick Stranghöner (Universität Bielefeld, jannick.stranghoener@uni-bielefeld.de), 2024
    - (No warranties; adapted for demonstration of GelSight 3D capabilities)
 """

import sys
import argparse
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_edge_in_rgb(rgb,
                     canny_threshold1=50,
                     canny_threshold2=150,
                     hough_threshold=50,
                     min_line_length=50,
                     max_line_gap=10):
    """
    1) Convert the RGB image to grayscale.
    2) Apply Canny edge detection.
    3) Use HoughLinesP to find prominent line segments.

    Returns:
        edges (binary edge image),
        lines (list of lines from Hough transform),
        edge_img_color (for visualization).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, L2gradient=True)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=0.5,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Visualization: draw lines on top of the original RGB
    edge_img_color = rgb.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edge_img_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return edges, lines, edge_img_color, gray


def plot_two_lines_with_angles(m_left, b_left, m_right, b_right):
    """
    Demonstration of:
      1) Plotting two lines in an (x,y) plane.
      2) Drawing the angle alpha between them at their intersection.
      3) Drawing the bisector line of that angle.
    """

    # ---------------------------------------------------------------------
    # 1) Find intersection (x_i, y_i)
    #    Solve m_left*x + b_left = m_right*x + b_right
    # ---------------------------------------------------------------------
    if np.isclose(m_left, m_right):
        # Lines are parallel or nearly so; skip intersection
        x_i = 0.0
        y_i = b_left
        print("Lines are nearly parallel; skipping angle plot.")
    else:
        x_i = (b_right - b_left) / (m_left - m_right)
        y_i = m_left*x_i + b_left

    # ---------------------------------------------------------------------
    # 2) Create a range of x-values for plotting each line
    #    We'll pick some region around the intersection
    # ---------------------------------------------------------------------
    x_min = x_i - 10
    x_max = x_i + 10
    xs = np.linspace(x_min, x_max, 200)

    # Evaluate each line
    y_left = m_left*xs + b_left
    y_right = m_right*xs + b_right

    # ---------------------------------------------------------------------
    # 3) Compute angles + bisector
    # ---------------------------------------------------------------------
    # Each slope m corresponds to an angle theta = arctan2(dy, dx) = arctan(m).
    # We'll define:
    theta_left  = np.arctan(m_left)
    theta_right = np.arctan(m_right)

    # Angle between lines (alpha):
    alpha_rad = np.abs(theta_right - theta_left)
    alpha_deg = np.degrees(alpha_rad)

    # Bisector angle (beta):
    beta_rad = 0.5*(theta_left + theta_right)
    beta_deg = np.degrees(beta_rad)

    # ---------------------------------------------------------------------
    # 4) Plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(xs, y_left, 'g-', label=f'Left line (m={m_left:.2f}, b={b_left:.2f})')
    ax.plot(xs, y_right, 'r-', label=f'Right line (m={m_right:.2f}, b={b_right:.2f})')

    # Mark intersection
    ax.plot(x_i, y_i, 'ko', label='Intersection')

    # 4a) Draw the bisector line
    #     The bisector has slope = tan(beta_rad).
    m_bis = np.tan(beta_rad)
    # We'll plot it through the same intersection point
    x_bis_vals = np.linspace(x_i - 10, x_i + 10, 200)
    y_bis_vals = m_bis*(x_bis_vals - x_i) + y_i
    ax.plot(x_bis_vals, y_bis_vals, 'b--',
            label=f'Bisector (beta={beta_deg:.1f}°)')

    # 4b) Draw an arc representing alpha at the intersection
    #     We'll do a small "angle arc" in data coords
    #     For that, we need an Arc patch.
    arc_radius = 2.0  # Choose an arc radius in your data units
    # We'll define an "angle in degrees" for the arc from min(theta_left,theta_right)
    # to max(theta_left,theta_right). We must convert from slope-based angles
    # to degrees in the same coordinate reference.
    theta1_deg = np.degrees(min(theta_left, theta_right))
    theta2_deg = np.degrees(max(theta_left, theta_right))

    arc = patches.Arc(
        (x_i, y_i),          # center
        2*arc_radius, 2*arc_radius,
        angle=0,             # no extra rotation of the arc itself
        theta1=theta1_deg,
        theta2=theta2_deg,
        color='k'
    )
    ax.add_patch(arc)

    # Label alpha near the midpoint of the arc
    angle_mid = 0.5*(theta1_deg + theta2_deg)
    angle_mid_rad = np.radians(angle_mid)
    label_x = x_i + arc_radius*np.cos(angle_mid_rad)
    label_y = y_i + arc_radius*np.sin(angle_mid_rad)
    ax.text(label_x, label_y, r'$\alpha$', fontsize=14, ha='center', va='center')

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.legend()
    ax.set_title("Two Lines with Angle and Bisector")
    plt.axis('equal')


def compute_line_angle_and_height(depth_map, lines, plot_cross_section=True):
    """
    Selects the longest line from 'lines' as 'the edge'.
    Then:
      - Computes its angle in the image XY plane (angle_xy_deg).
      - Computes the average depth along that line (avg_depth).
      - Computes a rough "angle of incidence" by sampling depth
        in a perpendicular direction near the midpoint (angle_incidence_deg).

    Args:
        depth_map: 2D array of depth values (float)
        lines: list of lines from Hough transform (on the RGB image)

    Returns:
        angle_xy_deg, avg_depth, angle_incidence_deg
        or (None, None, None) if no valid line was found.
    """
    if lines is None or len(lines) == 0:
        print("No lines found by Hough transform.")
        return None, None, None

    # Pick the longest line
    longest_line = None
    longest_length = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > longest_length:
            longest_length = length
            longest_line = (x1, y1, x2, y2)

    if not longest_line:
        print("No valid line found.")
        return None, None, None

    x1, y1, x2, y2 = longest_line

    # 1) Angle in the image XY plane
    dx = x2 - x1
    dy = y2 - y1
    angle_xy = np.arctan2(dy, dx)  # radians
    angle_xy_deg = np.degrees(angle_xy)

    # 2) Average depth along the line
    num_samples = int(longest_length)
    xs = np.linspace(x1, x2, num_samples, dtype=int)
    ys = np.linspace(y1, y2, num_samples, dtype=int)

    depth_values = []
    for (xi, yi) in zip(xs, ys):
        # Check bounds in depth map
        if 0 <= yi < depth_map.shape[0] and 0 <= xi < depth_map.shape[1]:
            zval = depth_map[yi, xi]
            if not np.isnan(zval):
                depth_values.append(zval)

    if len(depth_values) == 0:
        avg_depth = float('nan')
    else:
        avg_depth = float(np.nanmean(depth_values))

    # 3) Angle of incidence: sample perpendicular to the line in the depth map
    # Perp direction to (dx, dy) is (-dy, dx).
    cross_len = 25
    num_sample_point_edge = 10
    num_sample_points_cross = 50
    perp_dx = -dy
    perp_dy = dx

    # scale perp vector
    cross_norm = np.sqrt(perp_dx**2 + perp_dy**2) / (cross_len / 2)
    perp_dx /= cross_norm
    perp_dy /= cross_norm

    # calculate edge mid-point
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2

    # sample along cross
    cross_depths_left = []
    cross_distances_left = []
    cross_depths_right = []
    cross_distances_right = []

    for mx, my in zip(
            np.linspace(x1, x2, num_sample_point_edge),
            np.linspace(y1, y2, num_sample_point_edge)
    ):
        pxs = np.linspace(mx - perp_dx, mx + perp_dx, num_sample_points_cross, dtype=int)
        pys = np.linspace(my - perp_dy, my + perp_dy, num_sample_points_cross, dtype=int)

        for i in range(num_sample_points_cross):
            cx, cy = pxs[i], pys[i]
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                zval = depth_map[cy, cx]
                if not np.isnan(zval):
                    dist = np.sqrt((cx - mx)**2 + (cy - my)**2)
                    if i <= num_sample_points_cross / 2:
                        cross_depths_left.append(zval)
                        # measure distance from left-most point
                        cross_distances_left.append(-dist)
                    else:
                        cross_depths_right.append(zval)
                        # measure distance from left-most point
                        cross_distances_right.append(dist)

        # add midpoint depth value to both lists
        zval = depth_map[int(my), int(mx)]
        cross_depths_left.append(zval)
        cross_depths_right.append(zval)
        cross_distances_right.append(0)
        cross_distances_left.append(0)

    if len(cross_depths_left) + len(cross_depths_right) < 2:
        alpha_deg = float('nan')
    else:
        # 1) Fit the left slope
        p_left = np.polyfit(cross_distances_left, cross_depths_left, 1)
        m_left = p_left[0]     # slope from the left fit
        b_left = p_left[1]     # intercept

        # 2) Fit the right slope
        p_right = np.polyfit(cross_distances_right, cross_depths_right, 1)
        m_right = p_right[0]
        b_right = p_right[1]

        # ---------------------------------------------------------
        # 3) Compute the angle alpha between the two slopes
        # ---------------------------------------------------------
        ## ADDED:
        alpha_rad = np.abs(
            np.arctan((m_right - m_left) / (1.0 + m_left*m_right))
        )
        alpha_deg = np.degrees(alpha_rad)

        # we choose the obtuse angle if the "inner" angle is larger than 90°
        vec_left = np.array([1, m_left])
        vec_right = np.array([1, m_right])
        if np.dot(vec_left, vec_right) > 0:
            alpha_deg = 180 - alpha_deg

        # 4) Compute the “middle” angle beta, i.e., the bisector angle
        ## ADDED:
        theta_left = np.arctan(m_left)   # angle of left line w.r.t. x-axis (radians)
        theta_right = np.pi + np.arctan(m_right)  # angle of right line
        beta_rad = 0.5 * (theta_left + theta_right)
        beta_deg = np.degrees(beta_rad)
        m_bisector = np.tan(beta_rad)

        # 5) Compute depth along bisector
        #d_0 = cross_depths_left[-1]
        d_0 = avg_depth
        d_bisector = d_0 / np.sin(beta_rad)

        # ---------------------------------------------------------
        # 6) Plot each fit as before
        #    (Your existing plotting code here)
        # ---------------------------------------------------------
        if plot_cross_section:
            x_left = np.linspace(-cross_len / 2, 0, 100)
            z_fit_left = np.polyval(p_left, x_left)

            x_right = np.linspace(0, cross_len / 2, 100)
            z_fit_right = np.polyval(p_right, x_right)

            # plot_two_lines_with_angles(m_left, b_left, m_right, b_right)

            plt.figure()

            # Left data
            plt.plot(cross_distances_left, cross_depths_left, 'o', label='Left Depth')
            plt.plot(x_left, z_fit_left, '-g',
                     label=(f'Left fit slope={m_left:.4f}, intercept={b_left:.4f}'))

            # Right data
            plt.plot(cross_distances_right, cross_depths_right, 'o', label='Right Depth')
            plt.plot(x_right, z_fit_right, '-r',
                     label=(f'Right fit slope={m_right:.4f}, intercept={b_right:.4f}'))

            # bisector
            x = np.linspace(-1, 1, 100)
            plt.plot(x, m_bisector * x + d_0, '-k', label=(f'Bisector slope={m_bisector:.4f}'))

            plt.xlabel("Cross Distance [pixels from midpoint]")
            plt.ylabel("Depth [mm / 10]")
            plt.title("Cross-Section Depth Profile")
            plt.legend()
            plt.axis('equal')
            plt.grid()

    return longest_line, angle_xy_deg, avg_depth, alpha_deg, beta_deg, d_bisector, m_left, m_right


def main():
    """
    Usage:
        python analyze_depthmap.py path/to/depthmap.npy path/to/image.png

    Steps:
      1) Load the depth map from a .npy file.
      2) Load an RGB image (PNG) to detect edges.
      3) Use the detected edge's line to compute average depth & angle from the depth map.
    """
    parser = argparse.ArgumentParser(description="Analyze depth map and store results.")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="If set, plot the cross-section for visualization.")
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="Index for naming the output pickle file. Default=0")
    args = parser.parse_args()

    depth_map_path = f'../captures/depthmap_{args.index}.npy'
    rgb_path = f'../captures/img_{args.index}.png'
    print(f"Loading depth map from: {depth_map_path}")
    print(f"Loading RGB image from: {rgb_path}")

    # 1) Load the depth map (.npy)
    depth_map = np.load(depth_map_path)

    # Convert depth map to 8-bit for edge detection.
    # We'll scale the full range of depth_map to 0..255.
    min_val = np.nanmin(depth_map)
    max_val = np.nanmax(depth_map)
    if np.isnan(min_val) or np.isnan(max_val):
        raise ValueError("Depth map contains only NaNs or invalid data.")
    scale_factor = 255.0 / (max_val - min_val + 1e-9)
    depth_8u = np.uint8(np.clip((depth_map - min_val) * scale_factor, 0, 255))

    # Optionally denoise
    depth_8u = cv2.GaussianBlur(depth_8u, (5, 5), 0)

    # 2) Load the RGB image
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"ERROR: Could not load image from {rgb_path}")
        sys.exit(1)

    # 3) Detect edges on the RGB image
    edges, lines, edge_img_color, gray = find_edge_in_rgb(rgb, hough_threshold=30, max_line_gap=25)

    line = lines[0, 0]
    N = int(1.6 * np.sqrt(line[0]*line[2] + line[1]*line[3]))
    depth_vals = [depth_map[int(round(y)), int(round(x))] for x, y in zip(np.linspace(line[0], line[2], N), np.linspace(line[1], line[3], N))]
    # cropped_depth_vals = depth_vals[int(0.3*N): int(1.3*N)]
    cropped_depth_vals = depth_vals

    plt.figure()
    plt.plot(np.linspace(0,1,len(cropped_depth_vals)), cropped_depth_vals)
    plt.xlabel("Edge Distance (scaled px)")
    plt.ylabel("Depth (mm)")
    plt.title("Depth Profile along edge")

    # 4) Compute line angle + average depth + angle-of-incidence using the depth map
    longest_line, angle_xy_deg, avg_depth, alpha_deg, beta_deg, d_bisector, m_left, m_right = \
        compute_line_angle_and_height(depth_map, lines, plot_cross_section=args.plot)

    print("========== RESULTS ==========")
    if angle_xy_deg is not None:
        print(f"Edge location:                               {longest_line} [px]")
        print(f"Angle in the XY plane:                       {angle_xy_deg:.2f} [deg]")
        print(f"Average depth along the edge:                {avg_depth:.3f} [mm]")
        print(f"Angle between left and right edge:           {alpha_deg:.3f} [deg]")
        print(f"Angle between bisector and GelSight surface: {beta_deg:.3f} [deg]")
        print(f"Depth along bisector:                        {d_bisector:.3f} [mm]")
        plt.show()
    else:
        print("Could not compute measurements (no valid edge found).")

    results = {
        'angle_xy_deg': angle_xy_deg,
        'avg_depth': avg_depth,
        'alpha_deg': alpha_deg,
        'beta_deg': beta_deg,
        'd_bisector': d_bisector,
        'edge_location': longest_line,
        'm_left': m_left,
        'm_right': m_right
    }

    # Save to a pickle file that depends on index
    pickle_filename = f'../results/results_{args.index}.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(results, f)

    # After closing the plt plot, display images
    if args.plot:
        cv2.imshow("Gray", cv2.resize(gray, (0, 0), fx=2.0, fy=2.0))
        cv2.imshow("Depth", cv2.resize(depth_8u, (0, 0), fx=2.0, fy=2.0))
        cv2.imshow("Edges", cv2.resize(edges, (0, 0), fx=2.0, fy=2.0))
        cv2.imshow("Detected Lines on Depth Map", cv2.resize(edge_img_color, (0, 0), fx=2.0, fy=2.0))

        while True:
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
