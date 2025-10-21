import numpy as np
import open3d as o3d
import os
import subprocess


def render_result(mesh, facemarks):
    if os.environ["DISPLAY"] == ":99":
        print("Cannot render result with virtual display.\n")
        return

    facemarks_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(facemarks))
    facemarks_pcd.colors = o3d.utility.Vector3dVector([ [1,0,1] for _ in range(len(facemarks)) ])

    o3d.visualization.draw([mesh, facemarks_pcd])


def _ensure_display():
    if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        print(f"Using existing display {os.environ['DISPLAY']}")
        return

    print("No display detected, starting Xvfb...")

    try:
        display_num = 99
        xvfb_cmd = [
            "Xvfb", f":{display_num}", "-screen", "0", "1024x768x24", "-ac"
        ]
        proc = subprocess.Popen(xvfb_cmd)

        os.environ["DISPLAY"] = f":{display_num}"

        print(f"Xvfb started on display {os.environ['DISPLAY']} (pid={proc.pid})")

    except FileNotFoundError:
        raise RuntimeError(
            "Xvfb is not installed, and no display is available. "
            "Install Xvfb (e.g., `sudo apt install xvfb`) or run with a display."
        )
