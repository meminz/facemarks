import numpy as np
import open3d as o3d
import os


def render_result(mesh, facemarks):
    display = os.environ.get("DISPLAY", "")
    if not display or display == ":99":
        print("Cannot render result without a display.\n")
        return

    facemarks_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(facemarks))
    facemarks_pcd.colors = o3d.utility.Vector3dVector([ [1,0,1] for _ in range(len(facemarks)) ])

    o3d.visualization.draw([mesh, facemarks_pcd])
