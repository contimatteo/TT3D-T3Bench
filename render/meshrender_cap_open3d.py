import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
import imageio
import open3d as o3d

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--name', type=str)
args = parser.parse_args()

obj_path = args.path
# print(obj_path, "???")

# load mesh
try:
    # mesh = trimesh.load(obj_path, force='mesh')
    o3d_mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(obj_path)
    assert o3d_mesh.has_vertex_colors()
    assert o3d_mesh.has_vertex_normals()
except Exception as e:    
    icosphere = trimesh.creation.icosphere(subdivisions=0)
    for idx, v in enumerate(icosphere.vertices):
        color = Image.fromarray(np.zeros((512, 512, 3)).astype(np.uint8))
        imageio.imwrite(f'{args.name}/{idx:03d}.png', color)
    exit(0)

vertices = np.asarray(o3d_mesh.vertices)
# # normalize mesh to unit cube [-1, 1]^3
# mesh.vertices -= mesh.vertices.mean(axis=0)
# mesh.vertices /= np.abs(mesh.vertices).max()
vertices -= vertices.mean(axis=0)
vertices /= np.abs(vertices).max()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)

# mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
mesh = pyrender.Mesh.from_points(
    points=o3d_mesh.vertices,
    colors=o3d_mesh.vertex_colors,
    normals=o3d_mesh.vertex_normals,
)
mesh.primitives[0].material.baseColorFactor = [1., 1., 1., 1.]

Radius = 2.2

# compose scene
scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# light = pyrender.Ambi

scene.add(mesh, pose=np.eye(4))
# scene.add(light, pose=  np.eye(4))

scores = {}

# use icosphere as poses
icosphere = trimesh.creation.icosphere(subdivisions=0, radius=Radius)
icosphere.vertices *= Radius
for idx, v in enumerate(icosphere.vertices):
    new_scene = deepcopy(scene)
    # location [v]; vector [-v]
    # v = np.array([2, -0.5, 2])
    f = v / np.linalg.norm(v)
    u = np.array([0, 0, 1]) if np.abs(np.dot(f, [0, 0, 1])) < 0.99 else np.array([0, 1, 0])
    r = np.cross(u, -v)
    r = r / np.linalg.norm(r)
    u_ = np.cross(-v, r)
    u_ = u_ / np.linalg.norm(u_)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([-r, u_, f], axis=1)
    pose[:3, 3] = v
    new_scene.add(camera, pose=pose)

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(new_scene, flags=pyrender.constants.RenderFlags.FLAT)

    # convert color to PIL image
    color = Image.fromarray(color.astype(np.uint8))
    imageio.imwrite(f'{args.name}/{idx:03d}.png', color)
