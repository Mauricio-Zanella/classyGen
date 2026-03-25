import os
import pyvista as pv
import classy_blocks as cb
import numpy as np 
from common import Spline
from tqdm import tqdm


def fixPoints(points, radius, angleMax = 90.0, distanceMin = 1e-12):
    diffs = np.diff(points, axis=0)
    dist_sq = np.sum(diffs * diffs, axis=1) 
    keep_duplicate_mask = np.ones(len(points), dtype=bool)

    for i in range(len(points) - 1):
        if dist_sq[i] < distanceMin**2:
            keep_duplicate_mask[i + 1] = False

    points_clean = points[keep_duplicate_mask]
    radius_clean = radius[keep_duplicate_mask]
 


    vectors = np.diff(points_clean, axis=0)          
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms

    dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)

    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)

    keep_angle_mask = np.ones(len(points_clean), dtype=bool)
    for j in range(len(angles_deg)):
        if angles_deg[j] > angleMax:
            keep_angle_mask[j + 1] = False  

    points = points_clean[keep_angle_mask]
    radius = radius_clean[keep_angle_mask]

    return points, radius


# def fixPoints(points, radius):
#     vectors = np.diff(points, axis=0)
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     for i in range(len(norms)):
#         if 
#     norms = np.where(norms == 0, 1.0, norms)
#     unit_vectors = vectors / norms

#     # Compute dot products between consecutive unit vectors
#     dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)

#     # Clip dot products to [-1, 1] to avoid numerical issues
#     dot_products = np.clip(dot_products, -1.0, 1.0)

#     # Compute angles using arccos
#     angles_rad = np.arccos(dot_products)


#     return points, radius

def tangents(points):
    tang = []
    n = len(points)
    for i in range(n):
        if i == 0:
            t = points[i+1] - points[i]
        elif i == n-1:
            t = points[i] - points[i-1]
        else:
            t = points[i+1] - points[i-1]
        norm = np.linalg.norm(t)
        if norm > 0:
            tang.append(t / norm)
        else:
            tang.append(tang[-1] if i>0 else np.array([1,0,0]))
    return tang


def findRadius(v, r):
    if abs(v[0]) < abs(v[1]) and abs(v[0]) < abs(v[2]):
        p = np.cross(v, [0, 1, 0])
    else:
        p = np.cross(v, [1, 0, 0])
    
    p = r * p / np.linalg.norm(p)
    
    return p


csv = '/home/mauricio/Documents/Unesp/CFD/classyGen/curve.csv'


radius = np.genfromtxt(csv, delimiter=',', usecols=5)[1::]
points = np.genfromtxt(csv, delimiter=',', usecols=[6,7,8])[1::]
points, radius = fixPoints(points,radius)
tang = tangents(points)
# a = [points[0]]
# for i in range(len(points)-1):
#     b = np.linalg.norm((points[i]-points[1+1]))
#     if b<20:
#         a.append(points[i+1])
    
# print(a)
# print(max(a))
# print(min(a))

# a = Spline(np.array(points))
# # 
# plotter = pv.Plotter()
# plotter.add_mesh(a.GetSplinePolyData(), color='black', point_size=10, render_points_as_spheres=True, label='Original Points')
# plotter.add_legend()
# plotter.show()


# exit()
mesh = cb.Mesh()
base = '/home/mauricio/Documents/Unesp/CFD/classyGen'

frames = []   

t0 = tang[0]
up = np.array([0, 0, 1])
if abs(np.dot(t0, up)) > 0.99:
    up = np.array([0, 1, 0])
v1 = np.cross(t0, up)
v1 = v1 / np.linalg.norm(v1)
v2 = np.cross(t0, v1)      
v2 = v2 / np.linalg.norm(v2)   

frames.append((v1, v2))

for i in range(1, len(points)):
    t_curr = tang[i]
    v1_prev, v2_prev = frames[i-1]

    v1 = v1_prev - np.dot(v1_prev, t_curr) * t_curr
    norm = np.linalg.norm(v1)
    if norm < 1e-12:          
        v1 = v2_prev - np.dot(v2_prev, t_curr) * t_curr
        norm = np.linalg.norm(v1)
    v1 = v1 / norm
    v2 = np.cross(t_curr, v1)
    frames.append((v1, v2))

sketches = []
r = 0.4
for i in tqdm(range(len(points))):
    v1, v2 = frames[i]
    corner1 = points[i] + r * v1
    corner2 = points[i] + r * v2

    sketch = cb.SplineDisk(
        center_point=points[i],
        corner_1_point=corner1,
        corner_2_point=corner2,
        side_1=0, 
        side_2=0,
        n_outer_spline_points=20
    )
    sketches.append(sketch)

skip = int(np.floor(len(sketches)/100))
extrude = cb.LoftedShape(sketches[0], sketches[-1], [*sketches[1:-2:skip]])
# extrude = cb.LoftedShape(sketches[0], sketches[100], [*sketches[0:99]])

for axis in range(3):
    extrude.chop(axis, count=10)




mesh = cb.Mesh()

mesh.add(extrude)

base = '/home/mauricio/Documents/Unesp/CFD/classyGen'
mesh.write(os.path.join(base, "case", "system", "blockMeshDict"))