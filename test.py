from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
import matplotlib as mpl
mpl.use('Qt5Agg')

ctrlpts = [[5.0, 5.0, 0.0], [5.0, 10.0, 0.0], [10.0, 10.0, 5.0], [10.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 5.0, 10.0], [5.0, 5.0, 15.0], [5.0, 10.0, 15.0], [10.0, 10.0, 15.0], [10.0, 5.0, 20.0], [5.0, 5.0, 20.0]]

# Create a B-Spline curve instance
curve = BSpline.Curve()

# Set up curve
curve.degree = 3
curve.ctrlpts = ctrlpts

# Auto-generate knot vector
curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)

# Set evaluation delta
curve.delta = 0.01

# Plot the control point polygon and the evaluated curve
curve.vis = VisMPL.VisCurve3D()
curve.render()
curve._vis_component.render