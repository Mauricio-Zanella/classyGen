import os
import pandas as pd
import pyvista as pv
import classy_blocks as cb
import numpy as np 
from tqdm import tqdm

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
df = pd.read_csv(csv)
# print(df.columns)

points = np.array(df[['Points_0', 'Points_1', 'Points_2']].values)
radius = np.array(df['MaximumInscribedSphereRadius'].values)

tang = tangents(points)
# plotter = pv.Plotter()
# plotter.add_mesh(pv.PolyData(points), color='black', point_size=10, render_points_as_spheres=True, label='Original Points')
# plotter.add_legend()
# plotter.show()


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

# skip = int(np.floor(len(sketches)/100))
# extrude = cb.LoftedShape(sketches[0], sketches[-1], [*sketches[1:-2:skip]])
extrude = cb.LoftedShape(sketches[0], sketches[100], [*sketches[0:99]])

for axis in range(3):
    extrude.chop(axis, count=10)




mesh = cb.Mesh()

mesh.add(extrude)

base = '/home/mauricio/Documents/Unesp/CFD/classyGen'
mesh.write(os.path.join(base, "case", "system", "blockMeshDict"))