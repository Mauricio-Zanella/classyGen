import os
import pyvista as pv
import classy_blocks as cb
import numpy as np 
from common import Spline
from tqdm import tqdm


import numpy as np

def fixPoints(points, radius, angleMax=90.0, distanceMin=1e-12):
    diffs = np.diff(points, axis=0)
    dist_sq = np.sum(diffs * diffs, axis=1) 
    mask = np.ones(len(points), dtype=bool)

    for i in range(len(points) - 1):
        if dist_sq[i] < distanceMin**2:
            mask[i + 1] = False

    fixedPoints = points[mask]
    fixedRadius = radius[mask]


    mask = np.ones(len(fixedPoints), dtype=bool)

    i = 1
    while i < len(fixedPoints) - 1:
        v1 = fixedPoints[i] - fixedPoints[i - 1]
        n1 = np.linalg.norm(v1)
        if n1 < distanceMin:
            mask[i] = False
            i += 1
            continue
        v1 /= n1

        v2 = fixedPoints[i + 1] - fixedPoints[i]
        n2 = np.linalg.norm(v2)
        if n2 < distanceMin:
            mask[i + 1] = False
            i += 1
            continue
        v2 /= n2

        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        if angle > angleMax:
            j = i + 1
            while j < len(fixedPoints):
                v2 = fixedPoints[j] - fixedPoints[i]
                n2 = np.linalg.norm(v2)

                if n2 < distanceMin:
                    mask[j] = False
                    j += 1
                    continue

                v2 /= n2

                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))

                if angle <= angleMax:
                    break

                mask[j] = False
                j += 1

            if j < len(fixedPoints):
                i = j
            else:
                break
        else:
            i += 1

    points = fixedPoints[mask]
    radius = fixedRadius[mask]

    return points, radius

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

points = Spline(np.array(points), n_spline_points = 300).GetPoints()
radius = np.interp(np.linspace(0, 1, len(points)), np.linspace(0, 1, len(radius)), radius)
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
# 
print(len(points))
print(len(radius))
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
print(len(sketches))

segCount = 30
segSize = len(sketches)/30
mesh = cb.Mesh()

for i in range(segCount-1):
    first = int(np.round(i*segSize))
    last = int(np.round((i+1)*segSize))
    extrude = cb.LoftedShape(sketches[first], sketches[last], [*sketches[first+1:last-1]])
    for axis in range(3):
        extrude.chop(axis, count=10)

    mesh.add(extrude)





# skip = int(np.floor(len(sketches)/100))
# extrude = cb.LoftedShape(sketches[0], sketches[-1], [*sketches[1:-2:skip]])
# extrude = cb.LoftedShape(sketches[0], sketches[141], [*sketches[0:141]])

# for axis in range(3):
#     extrude.chop(axis, count=10)




# mesh = cb.Mesh()

# mesh.add(extrude)

base = '/home/mauricio/Documents/Unesp/CFD/classyGen'
mesh.write(os.path.join(base, "case", "system", "blockMeshDict"))