import os
import json

from OCC.Extend.TopologyUtils import TopologyExplorer, discretize_edge, discretize_wire, get_type_as_string
from OCC.Extend.ShapeFactory import get_oriented_boundingbox, get_aligned_boundingbox, \
    measure_shape_mass_center_of_gravity
from OCC.Extend.DataExchange import read_step_file, export_shape_to_svg
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve, BRepAdaptor_Curve2d
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBoundsProperties
from OCC.Core.GeomAbs import *
from OCC.Core.TColgp import *
from OCC.Core.TColStd import *
from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import (brepgprop_LinearProperties,
                                brepgprop_SurfaceProperties,
                                brepgprop_VolumeProperties)
from abc import ABC, abstractmethod
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.BRepTools import breptools_Write
from OCC.Core.Convert import Convert_CircleToBSplineCurve,Convert_EllipseToBSplineCurve

class Curve:
    def __init__(self, face, surf):
        self.face = face
        self.surf = surf

    @abstractmethod
    def extract_curve_data(self,param1,param2):
        pass


class CurveFactory:

    def create_curve_object(self, curve_adapter, face, surf, c_id):

        curve_type = curve_adapter.GetType()

        if curve_type == GeomAbs_BSplineCurve:
            surf.trimmed = True
            bspline_curve = curve_adapter.BSpline()
            return BSplineCurve(bspline_curve, face, surf, c_id)


        elif curve_type == GeomAbs_Line:
            surf.trimmed = True
            line_curve = curve_adapter.Line()
            return Line(line_curve, face, surf, c_id)


        elif curve_type == GeomAbs_Circle:
            surf.trimmed = True
            circle_curve = curve_adapter.Circle()
            return Circle(circle_curve,face,surf,c_id)


        elif curve_type == GeomAbs_Ellipse:
            surf.trimmed = True
            ellipse_curve = curve_adapter.Ellipse()
            return Ellipse(ellipse_curve,face,surf,c_id)


class BSplineCurve(Curve):
    def __init__(self, bspline_curve, face, surf, c_id):
        super(BSplineCurve, self).__init__(face, surf)
        self.bspline_curve = bspline_curve
        self.curve_info = {}
        self.curve_type = "spline"
        self.face = face
        self.order = 0
        self.n_points = 0
        self.ctrl_points = []
        self.weights = []
        self.knotvector = []
        self.c_id = c_id


    def extract_curve_data(self, f_id,curve_adapter,trim_type):

        self.degree=self.bspline_curve.Degree()
        self.rational = self.bspline_curve.IsRational()
        self.continuity=self.bspline_curve.Continuity()
        self.order = self.bspline_curve.Degree() + 1
        self.n_points = self.bspline_curve.NbPoles()
        self.periodic = self.bspline_curve.IsPeriodic()

        if self.periodic:
            self.bspline_curve.SetNotPeriodic()
        # FIXME Not work properly for the trim curves from surfaces, Here I use 3d array points instead, just like the extract_data for faces
        if trim_type == '2d':
            p = TColgp_Array1OfPnt2d(1, self.bspline_curve.NbPoles())
        elif trim_type == '3d':
            p = TColgp_Array1OfPnt(1, self.bspline_curve.NbPoles())
        else:
            return None
        self.bspline_curve.Poles(p)

        for pi in range(p.Length()):
            self.ctrl_points.append(list(p.Value(pi + 1).Coord()))

        k = TColStd_Array1OfReal(1, self.bspline_curve.NbPoles() + self.bspline_curve.Degree() + 1)
        self.bspline_curve.KnotSequence(k)

        for ki in range(k.Length()):
            self.knotvector.append(k.Value(ki + 1))

        w = TColStd_Array1OfReal(1, self.bspline_curve.NbPoles())
        self.bspline_curve.Weights(w)
        for wi in range(w.Length()):
            self.weights.append(w.Value(wi + 1))

        self.curve_info['type'] = self.curve_type
        self.curve_info['curve id'] = self.c_id
        self.curve_info['rational']=self.rational
        self.curve_info['degree']=self.degree
        self.curve_info['knotvector']=self.knotvector
        self.curve_info['control_points']={}
        self.curve_info['control_points']['points']=self.ctrl_points
        self.curve_info['control_points']['weights']=self.weights
        self.curve_info['reversed']=0

        return self.curve_info


class Line(Curve):
    def __init__(self, line_curve, face, surf, c_id):
        super(Line, self).__init__(face, surf)
        self.line_curve = line_curve
        self.curve_info = {}
        self.c_id = c_id
        self.face = face
        self.surf = surf
        self.location=[]
        self.direction=[]

    def extract_curve_data(self, f_id,curve_adapter,trim_type):
        first_parameter = curve_adapter.FirstParameter()
        last_parameter = curve_adapter.LastParameter()
        start_pt = gp_Pnt2d()
        end_pt = gp_Pnt2d()
        curve_adapter.D0(first_parameter,start_pt)
        curve_adapter.D0(last_parameter,end_pt)




        self.location=list(self.line_curve.Location().Coord())
        self.direction = list(self.line_curve.Direction().Coord())

        self.curve_info['type'] = 'line'
        self.curve_info['curve id'] = self.c_id
        self.curve_info['data'] = {}
        self.curve_info['data']['location'] = self.location
        self.curve_info['data']['direction'] = self.direction
        self.curve_info['data']['start_point'] = start_pt.Coord()
        self.curve_info['data']['end_point'] = end_pt.Coord()
        self.curve_info['data']['first_parameter'] = first_parameter
        self.curve_info['data']['last_parameter'] = last_parameter


        return self.curve_info


class Circle(Curve):
    def __init__(self, circle_curve, face, surf, c_id):
        super(Circle, self).__init__(face, surf)
        self.circle_curve = circle_curve
        self.curve_info = {}
        self.c_id = c_id
        self.face = face
        self.surf = surf
        self.location = []
        self.radius=0
        self.x_axis=[]
        self.y_axis=[]

    def extract_curve_data(self, f_id,curve_adapter):

        first_parameter = curve_adapter.FirstParameter()
        last_parameter = curve_adapter.LastParameter()
        start_pt = gp_Pnt2d()
        end_pt = gp_Pnt2d()
        curve_adapter.D0(first_parameter, start_pt)
        curve_adapter.D0(last_parameter, end_pt)

        bspline=Convert_CircleToBSplineCurve(self.circle_curve)
        bspline_info = {}
        bspline_info['degree'] = bspline.Degree()
        bspline_info['number_of_poles'] = bspline.NbPoles()
        bspline_info['number_of_knots'] = bspline.NbKnots()
        bspline_info['poles'] = []
        bspline_info['weights'] = []
        bspline_info['knots'] = []
        bspline_info['multiplicity'] = []
        for index in range(bspline.NbPoles()):
            bspline_info['poles'].append(bspline.Pole(index))
            bspline_info['weights'].append(bspline.Weight(index))
        for index in range(bspline.NbKnots()):
            bspline_info['knots'].append(bspline.Knot(index))
            bspline_info['multiplicity'].append(bspline.Multiplicity(index))

        self.location = list(self.circle_curve.Location().Coord())
        self.radius=self.circle_curve.Radius()
        self.x_axis=list(self.circle_curve.XAxis().Direction().Coord())
        self.y_axis = list(self.circle_curve.YAxis().Direction.Coord())

        self.curve_info['type'] = 'circle'
        self.curve_info['curve id'] = self.c_id
        self.curve_info['data'] = {}
        self.curve_info['data']['location'] = self.location
        self.curve_info['data']['radius'] = self.radius
        self.curve_info['data']['axis']=self.circle_curve.Axis()
        self.curve_info['data']['x_axis']= self.x_axis
        self.curve_info['data']['y_axis']=self.y_axis
        self.curve_info['data']['start_point'] = start_pt.Coord()
        self.curve_info['data']['end_point'] = end_pt.Coord()
        self.curve_info['data']['first_parameter'] = first_parameter
        self.curve_info['data']['last_parameter'] = last_parameter
        self.curve_info['data']['bspline'] = bspline_info

        return self.curve_info


class Ellipse(Curve):
    def __init__(self, ellipse_curve, face, surf, c_id):
        super(Ellipse, self).__init__(face, surf)
        self.ellipse_curve = ellipse_curve
        self.curve_info = {}
        self.c_id = c_id
        self.face = face
        self.surf = surf
        self.focus1=[]
        self.focus2 =[]
        self.x_axis=[]
        self.y_axis=[]

    def extract_curve_data(self, f_id,curve_adapter):

        first_parameter = curve_adapter.FirstParameter()
        last_parameter = curve_adapter.LastParameter()
        start_pt = gp_Pnt2d()
        end_pt = gp_Pnt2d()
        curve_adapter.D0(first_parameter, start_pt)
        curve_adapter.D0(last_parameter, end_pt)

        bspline = Convert_EllipseToBSplineCurve(self.ellipse_curve)
        bspline_info = {}
        bspline_info['degree'] = bspline.Degree()
        bspline_info['number_of_poles'] = bspline.NbPoles()
        bspline_info['number_of_knots'] = bspline.NbKnots()
        bspline_info['poles'] = []
        bspline_info['weights'] = []
        bspline_info['knots'] = []
        bspline_info['multiplicity'] = []
        for index in range(bspline.NbPoles()):
            bspline_info['poles'].append(bspline.Pole(index))
            bspline_info['weights'].append(bspline.Weight(index))
        for index in range(bspline.NbKnots()):
            bspline_info['knots'].append(bspline.Knot(index))
            bspline_info['multiplicity'].append(bspline.Multiplicity(index))



        self.location = list(self.ellipse_curve.Location().Coord())
        self.focus1=list(self.ellipse_curve.Focus1().Coord())
        self.focus2=list(self.ellipse_curve.Focus2().Coord())
        self.axis = self.ellipse_curve.Axis()
        self.x_axis=list(self.ellipse_curve.XAxis().Direction().Coord())
        self.y_axis = list(self.ellipse_curve.YAxis().Direction.Coord())
        self.major_radius=self.ellipse_curve.MajorRadius()
        self.minor_radius=self.ellipse_curve.MinorRadius()

        self.curve_info['type'] = 'ellipse'
        self.curve_info['curve id'] = self.c_id
        self.curve_info['data'] = {}
        self.curve_info['data']['location'] = self.location
        self.curve_info['data']['focus1'] = self.focus1
        self.curve_info['data']['focus2'] = self.focus2
        self.curve_info['data']['axis']=self.axis
        self.curve_info['data']['x_axis']= self.x_axis
        self.curve_info['data']['y_axis']=self.y_axis
        self.curve_info['data']['major_radius'] = self.major_radius
        self.curve_info['data']['minor_radius'] = self.minor_radius
        self.curve_info['data']['start_point'] = start_pt.Coord()
        self.curve_info['data']['end_point'] = end_pt.Coord()
        self.curve_info['data']['first_parameter'] = first_parameter
        self.curve_info['data']['last_parameter'] = last_parameter
        self.curve_info['data']['bspline'] = bspline_info

        return self.curve_info





