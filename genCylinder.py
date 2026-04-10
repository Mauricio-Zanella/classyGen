# import pyvista as pv
import classy_blocks as cb
import numpy as np 
from common import Spline
from tqdm import tqdm
# import vmtk
# from vmtk import vmtkbranchextractor
from vmtk import pypes


# Tangent function, aproximates tangents with secant
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
        tang.append(t / norm)
    return tang


# Starting to use vmtk library
centerline = '/home/mauricio/Documents/Unesp/CFD/AneuriskDatabase/models/C0001/morphology/centerlines.vtp'
output = '/home/mauricio/Documents/Unesp/CFD/classyGen/branches.vtp'
args = f'vmtkbranchextractor -ifile {centerline} -ofile {output} -radiusarray MaximumInscribedSphereRadius'
myPype = pypes.PypeRun(args)


# Loads points and radius arrays
csv = '/home/mauricio/Documents/Unesp/CFD/classyGen/curve.csv'
radius = np.genfromtxt(csv, delimiter=',', usecols=5)[1::]
points = np.genfromtxt(csv, delimiter=',', usecols=[6,7,8])[1::]


# Modifies Points and Radius for Spline usage
points = Spline(np.array(points), n_spline_points = 300).GetPoints()
radius = np.interp(np.linspace(0, 1, len(points)), np.linspace(0, 1, len(radius)), radius)


tang = tangents(points)
base = '/home/mauricio/Documents/Unesp/CFD/classyGen'


# Defining first base circlepoints
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


# Defines next base circlepoints according to the last
for i in range(1, len(points)):
    t = tang[i]
    v1p, v2p = frames[i-1]
    v1 = v1p - np.dot(v1p, t) * t
    norm = np.linalg.norm(v1)
    if norm < 1e-12:          
        v1 = v2p - np.dot(v2p, t) * t
        norm = np.linalg.norm(v1)
    v1 = v1 / norm
    v2 = np.cross(t, v1)
    frames.append((v1, v2))


# Uses base circlepoints to craft the real circle
sketches = []
# r = 0.4
for i in tqdm(range(len(points))):
    v1, v2 = frames[i]
    corner1 = points[i] + radius[i] * v1
    corner2 = points[i] + radius[i] * v2

    sketch = cb.SplineDisk(
        center_point=points[i],
        corner_1_point=corner1,
        corner_2_point=corner2,
        side_1=0, 
        side_2=0,
        n_outer_spline_points=20
    )
    sketches.append(sketch)


# Method for creating finer meshes
segCount = 30
segSize = len(sketches)/30
mesh = cb.Mesh()
for i in range(segCount):
    first = int(np.round(i*segSize))
    last = int(np.round((i+1)*segSize))
    if i == segCount-1:
        last = len(sketches)-1
    extrude = cb.LoftedShape(sketches[first], 
                             sketches[last], 
                             [*sketches[first+1:last-1]])
    for axis in range(3):
        extrude.chop(axis, count=10)

    mesh.add(extrude)

# Writes BlockMesh
base = '/home/mauricio/Documents/Unesp/CFD/classyGen'
mesh.write(f'{base}/case/system/blockMeshDict')