import facemarks
import sys
import numpy as np

def main():
    print("Hello from facemarks!")

    meshes = facemarks.import_mesh_and_setup(sys.argv[1])
    prediction = facemarks.predict(meshes, projections=100)

    facemarks.render_result(meshes["original"], prediction["facemarks_3d"])

    closest_vertices = np.asarray(meshes["original"].vertices)[prediction["closest_vertex_ids"]]
    facemarks.render_result(meshes["original"], closest_vertices)




if __name__ == "__main__":
    main()
