import core
import sys

def main():
    print("Hello from facemarks!")

    meshes = core.import_mesh_and_setup(sys.argv[1])
    prediction = core.predict(meshes, projections=100)

    core.render_result(meshes["original"], prediction["facemarks_3d"])




if __name__ == "__main__":
    main()
