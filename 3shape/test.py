import numpy as np
import os
import trimesh
import glob
import random
import seaborn as sns
import json
from pathlib import Path
import io
from PIL import Image


def get_plane(fname):
    crimefile = open(r"C:\temp\planes\{}.txt".format(fname), 'r')
    line = crimefile.readlines()[0].replace('\n', '')
    data = np.array(line.split(',')).astype(float).reshape(2, 3)
    normals = data[0, :]
    origin = data[1, :]
    return origin, normals

def show_mesh_and_pale(fname):
    mesh_o = trimesh.load_mesh(r"C:\temp\Segmented\{}.ply".format(fname))
    origin, normals = get_plane(fname)
    scene = trimesh.Scene([mesh_o, trimesh.path.creation.grid(side=20,
                                                  plane_origin = origin,
                                                  plane_normal = normals)])
    return scene


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_fnames(pathList):
    res = []
    for fpath in pathList:
        fname = Path(fpath).stem
        res.append(fname)
    return res

files_to_process = glob.glob(r"C:\temp\render\*.jpg")
origins = []
normals = []

for cp in files_to_process:
    fname = os.path.splitext(os.path.basename(cp))[0]
    if not os.path.exists(r"C:\temp\planes\{}.txt".format(fname)):
        print(f"{fname} does not exists")
    else:
        origin, normal = get_plane(fname)
        origins.append(origin)
        normals.append(normal)
