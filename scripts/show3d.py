"""
show3d.py

Description:
    This script connects to a GelSight Mini device in real time, retrieves the current
    camera image, and applies a neural network reconstruction to generate a 3D point
    cloud (via the GelSight 3D reconstruction library, gs3drecon). The script displays
    a live view (in both 2D and 3D) of the GelSight sensor data and optionally allows
    the user to save depth maps, RGB images, and point clouds by pressing the 's' key.

Core Functionality:
    1. GelSight Device Connection:
       - Connects to the GelSight Mini via gsdevice.Camera, using a known or specified
         device ID (e.g., "GelSight Mini").
    2. Neural Network Loading:
       - Loads a pre-trained nnmini.pt model (a neural network for depth estimation)
         from the local path. (Select GPU/CPU via the GPU flag).
    3. Depth Map Computation:
       - Continuously captures frames from the GelSight sensor, passes them into
         nn.get_depthmap(...) to generate a depth map (masking markers if needed).
    4. 3D Visualization:
       - Uses gs3drecon.Visualize3D(...) to render a live 3D point cloud of the captured
         depth map. The user sees a 3D window updating in real time.
    5. User Interaction:
       - Press 'q' in the 2D window to quit the script.
       - Press 's' (detected by pynput keyboard listener) to save:
            a) the current depth map as a .npy file,
            b) the current RGB image as a .png file,
            c) the current point cloud as a .pcd file.
       - File naming is based on an incremental index plus a user-defined offset.

Flags/Variables (internal):
    - SAVE_VIDEO_FLAG: If True, saves an .avi video of the live feed to disk (disabled by default).
    - FIND_ROI: If True, allows the user to select a region of interest (ROI) in the initial frame.
    - GPU: If True, loads the neural network on CUDA instead of CPU.
    - MASK_MARKERS_FLAG: If True, masks marker areas in the GelSight image before depth reconstruction.
    - FILENAME: Base name for saved files. (Currently 'edge')
    - INDEX_OFFSET: An integer offset added to the saved index when pressing 's'.

Usage:
    python show3d.py

    (No specific command-line arguments are implemented here; see the code for
     toggling flags like SAVE_VIDEO_FLAG, FIND_ROI, etc.)

Notes:
    - On Linux, you may need to run 'v4l2-ctl --list-devices' to confirm the device ID
      for the GelSight camera if the default string ("GelSight Mini") doesn't match.

Author/Contact:
    - Jannick Stranghöner (Universität Bielefeld, jannick.stranghoener@uni-bielefeld.de), 2024
    - (No warranties; adapted for demonstration of GelSight 3D capabilities)
"""

import sys
import numpy as np
import cv2
import os
import gsdevice
import gs3drecon
import open3d

from pynput import keyboard


def main(argv):
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False
    MASK_MARKERS_FLAG = False
    FILENAME = 'edge'
    INDEX_OFFSET = 20

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")
    net_file_path = 'nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving video to {file_path}')

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)

    # ----------------------------------------------------------------------------
    # ADDED FOR DEPTH SAVE:
    # We'll store the latest depth map and an incremental index for saving multiple maps.
    # ----------------------------------------------------------------------------
    global_latest_depth_map = [None]  # using a mutable list so we can modify inside on_press
    global_latest_img = [None]  # using a mutable list so we can modify inside on_press
    depth_save_count = [0]           # similarly for the save counter

    def on_press(key):
        try:
            if key.char == 's':
                # Save the current point cloud, depth map and rgb image
                dm_current = global_latest_depth_map[0]
                img_current = global_latest_img[0]
                if dm_current is not None:
                    idx = depth_save_count[0] + INDEX_OFFSET

                    open3d.io.write_point_cloud(f'../captures/pc_{idx}.pcd', vis3d.pcd)
                    np.save(f'../captures/depthmap_{idx}.npy', dm_current)
                    cv2.imwrite(f'../captures/img_{idx}.png', img_current)

                    depth_save_count[0] += 1

                    print(f'Saved RGB image, depth map and point cloud for index {idx}!')
                else:
                    print('No depth map to save yet.')
        except AttributeError:
            # This occurs if key.char is None (e.g. special keys). We can ignore.
            pass
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    try:
        while dev.while_condition:

            # get the roi image
            f1 = dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            # ----------------------------------------------------------------------------
            # ADDED FOR DEPTH SAVE:
            # Update our global_latest_depth_map reference on each loop iteration
            # so the on_press callback can save the latest one.
            # ----------------------------------------------------------------------------
            global_latest_depth_map[0] = dm
            global_latest_img[0] = f1

            ''' Display the results '''
            vis3d.update(dm)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
