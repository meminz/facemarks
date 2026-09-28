import facemarks
import sys
import numpy as np

def main():
    meshes = facemarks.import_mesh_and_setup(sys.argv[1])
    prediction = facemarks.predict(meshes, projections=100)

    detected = [(coords, vid) for coords, vid
                in zip(prediction["facemarks_3d"], prediction["closest_vertex_ids"])
                if coords is not None]

    facemarks.render_result(meshes["original"], [coords for coords, _ in detected])


    closest_vertices = np.asarray(meshes["original"].vertices)[[vid for _, vid in detected]]
    facemarks.render_result(meshes["original"], closest_vertices)






if __name__ == "__main__":
    main()
