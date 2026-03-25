# Copyright (C) 2022, Iago L. de Oliveira

# vmtk4aneurysms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of general tools."""

import sys
import vtk
import numpy as np
from scipy.interpolate import make_splprep

def FlattenDict(
        pyobj,
        keystring=''
    ):

    if type(pyobj) == dict:
        keystring = keystring + '_' if keystring else keystring

        for k in pyobj:
            yield from FlattenDict(pyobj[k], keystring + str(k))

    else:
        yield keystring, pyobj

def RemoveArrayConsecutiveDuplicates(
        points: np.ndarray
    ) -> np.ndarray:
    """
    Removes consecutive duplicate 3D points from a Numpy array.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) where N is the
            number of points.

    Returns:
        np.ndarray: A new NumPy array containing only the unique consecutive
            points from the input array.
    """

    if points.shape[0] < 2:
        return points

    # Use boolean indexing to keep only points that are not identical to the
    # previous one
    unique_mask = np.ones(
                      points.shape[0],
                      dtype=bool
                  )

    # Check if all coordinates are the same for each row
    unique_mask[1:] = ~np.all(
                          points[1:] == points[:-1],
                          axis=1
                      )

    return points[unique_mask]

def DistanceBetweenConsecutivePoints(
        points: np.ndarray
    ) -> np.ndarray:
    """
    Computes the Euclidean distance between consecutive 3D points in a NumPy
    array.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) where N is the
            number of points.

    Returns:
        A NumPy array of shape (N-1,) containing the distances between
        (point[0], point[1]), (point[1], point[2]), ..., (point[N-2],
        point[N-1]).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input 'points' must be an Nx3 NumPy array.")

    if points.shape[0] < 2:
        return np.array([]) # No distances if less than 2 points

    # Calculate vectors between consecutive points
    vectors = points[1:] - points[:-1]

    # Compute the Euclidean norm (magnitude) of each vector
    distances = np.linalg.norm(vectors, axis=1)

    return distances

class Spline():
    """Creates spline object based on points coordinates."""

    def __init__(
            self,
            points_coords: np.ndarray,
            n_spline_points: int=100
        ) -> tuple:
        """Build spline based on input points.

        Args:
            points_coords: A 2D NumPy array of shape (N, 3) where N is the
                number of points.

            n_spline_points: Number of points to generate along the spline.
        """

        # Remove consecuive duplicated points
        # (possible with centerlines)
        self._orig_points_coords_array = RemoveArrayConsecutiveDuplicates(
                                             points_coords
                                         )
        self._npoints = n_spline_points

        self._points = None
        self._tangents = None
        self._bspline_object = None
        self._parametric_space = None
        self._tangent_field_name = "Tangents"

        self._build_spline()

        # Evaluate with default number of points
        self.Evaluate(self._npoints)

    def __deepcopy__(self, memo):
        # Create a new instance of the class
        new_instance = type(self)(
            points_coords=self._orig_points_coords_array.copy(),
            n_spline_points=self._npoints
        )

        # Add the new object to the memo dictionary
        memo[id(self)] = new_instance

        # Copy over the evaluated points and tangents
        new_instance._points = self._points.copy() \
                                if self._points is not None \
                                else None

        new_instance._tangents = self._tangents.copy() \
                                    if self._tangents is not None \
                                    else None

        new_instance._bspline_object = self._bspline_object
        new_instance._parametric_space = self._parametric_space
        new_instance._tangent_field_name = self._tangent_field_name

        return new_instance

    def _build_spline(self):

        # Compute distance coordinates
        incrDistances = DistanceBetweenConsecutivePoints(
                            self._orig_points_coords_array
                        )

        distanceCoord = np.cumsum(incrDistances)
        # Add the zero elemtn at the begin of the array
        # (match sizes with npoints)
        distanceCoord = np.insert(distanceCoord, 0, 0.0)

        # get spline object
        self._bspline_object, self._parametric_space = make_splprep(
                                                           self._orig_points_coords_array.T,
                                                           u=distanceCoord,
                                                           k=3, # Ensure cubic interpolation
                                                           s=0.1 # A small smoothing factor
                                                       )

    def Evaluate(self, n_points):
        # Evaluate the BSpline object at new parameter values
        # This generates points along the spline curve.
        parametricValues = np.linspace(
                               self._parametric_space.min(),
                               self._parametric_space.max(),
                               n_points
                           )

        self._points   = self._bspline_object(parametricValues, nu=0).T
        tangents = self._bspline_object(parametricValues, nu=1).T

        # Normalize tangents directly
        norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        # Avoid division by zero if a norm is zero
        norms[norms == 0] = 1.0

        self._tangents = tangents/norms

    def EvaluateNonUniformSpacing(
            self,
            parametric_values: np.ndarray
        )   -> tuple:
        """
        Evaluates the stored B-spline object at a specific array of arc length
        (parametric) values.

        Args:
            parametric_values (np.ndarray): Array of arc length values to
                evaluate the spline at.
        """
        # Clamp parameters to the valid range
        # Check for potential bounds issue just in case, though clipping is
        # handled internally now.
        if np.any(parametric_values > self._parametric_space.max()):
            print(
                "Warning: parametric_values exceeds spline bounds. "
                "Clamping parameters."
            )

        parametricValues = np.clip(
                                parametric_values,
                                self._parametric_space.min(),
                                self._parametric_space.max()
                           )

        self._points   = self._bspline_object(parametricValues, nu=0).T
        tangents = self._bspline_object(parametricValues, nu=1).T

        # Normalize tangents directly
        norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        # Avoid division by zero if a norm is zero
        norms[norms == 0] = 1.0

        self._tangents = tangents/norms

    def GetPoints(self) -> np.ndarray:
        return self._points

    def GetTangents(self) -> np.ndarray:
        return self._tangents

    def GetSplinePolyData(self) -> vtk.vtkPolyData:
        """
        Returns the vtkPolyData object representing the spline points.

        The curve is formed by connecting consecutive points with line
        segments.

        Returns:
            A vtkPolyData object containing the points and the line cells that
            form the curve.
        """

        # Create a vtkPoints object to store the coordinates
        vtkPoints = vtk.vtkPoints()
        for point in self._points:
            vtkPoints.InsertNextPoint(
                point[0],
                point[1],
                point[2]
            )

        # Create a vtkCellArray to store the line segments
        vtkLines = vtk.vtkCellArray()

        # Create line segments by connecting consecutive points
        for i in range(len(self._points) - 1):
            line = vtk.vtkLine()

            # First point of the segment
            line.GetPointIds().SetId(0, i)
            # Second point of the segment
            line.GetPointIds().SetId(1, i + 1)
            vtkLines.InsertNextCell(line)

        # Create the vtkPolyData object
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(vtkPoints)
        polyData.SetLines(vtkLines)

        # Create Tangents field
        pointDataArray = vtk.vtkFloatArray()
        pointDataArray.SetNumberOfComponents(3)

        pointDataArray.SetName(self._tangent_field_name)
        for tgn in self._tangents:
            pointDataArray.InsertNextTuple(tgn)

        polyData.GetPointData().SetActiveVectors(self._tangent_field_name)
        polyData.GetPointData().SetVectors(pointDataArray)

        return polyData
