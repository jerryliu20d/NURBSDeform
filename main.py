from abstract import *

def main():
    filename = 'pyramid'
    # filename = '3d Nurbs Area'
    shp = read_step_file('data/models/' + filename + '.step')
    t = TopologyExplorer(shp)
    for face_idx, subshape in enumerate(t.faces()):
        surface_factory = SurfaceFactory()
        surface = surface_factory.create_surface_object(subshape, face_idx)
        if surface is not None:
            surface.extract_data()
            # surface.dump_data('Models/parsed/'+filename, face_idx)


if __name__ == '__main__':
    main()


# %%
# surface.config['shape']['data']['control_points']['trims']['data'][0][0]
# surface.config['shape']['data']['control_points'].keys()
# %%

filename = 'pyramid'
filename = 'Cube'
# filename = '3d Nurbs Area'
shp = read_step_file('data/models/' + filename + '.step')
t = TopologyExplorer(shp)
print(t.number_of_faces_from_edge(next(t.edges())))
subshape = next(t.faces())
face_idx = 0
surface_factory = SurfaceFactory()
surface = surface_factory.create_surface_object(subshape, face_idx)
if surface is not None:
    surface.extract_data()


class Surface:
    def __init__(self, surface):
        self._init_ctp = surface.config['shape']['data']['control_points']['points'] # 3d list type, u*v*3
        self._ctp = surface.config['shape']['data']['control_points']['points'] # 3d list type u*v*3
        self.weight = surface.config['shape']['data']['control_points']['weights'] # 2d list type u*v
        self.degree_u = surface.config['shape']['data']['degree_u'] # int
        self.degree_v = surface.config['shape']['data']['degree_v'] # int
        self.knotvector_u = surface.config['shape']['data']['knotvector_u'] # list
        self.knotvector_v = surface.config['shape']['data']['knotvector_v'] # list
        self.size_u = surface.config['shape']['data']['size_u'] # int
        self.size_v = surface.config['shape']['data']['size_v'] # int
        self.bounds = surface.config['shape']['data']['bounds'] # list of int, [u_low,u_up,v_low,v_up], used for trim2d
        self.trim2d = self.extract_trim2d(surface) # 2d rep of trims, return list of geomdl.BSpline.Curve
        self.trim3d = self.extract_trim3d(surface) # list of Curve object.

    def extract_trim2d(self, surface):
        # todo how should I represent the 2d NURBS in property?
        # maybe use original style, convenient to convert it back?
        # or scale it to [0,1]?
        return surface.config['shape']['data']['control_points']['trims']['data'][0]

    def extract_trim3d(self, surface):
        return surface.config['shape']['data']['control_points']['trims']['data3D'][0]

    @property
    def ctp(self):
        return self._ctp

    @ctp.setter
    def ctp(self, new_ctp):
        # todo make sure the data type is still list of list
        self._ctp = new_ctp

    @ctp.deleter
    def ctp(self):
        self._ctp = self._init_ctp

class Curve:
    def __init__(self):
        pass
    # todo We first create edges and connect it to Surface.trim3d according to OCC topology: TopologyExplorer.faces_from_edge(...) and TopoDS_Shape.IsSame(...)
    # todo how should I merge the two duplicated/close enough edges? NURBS object will have separate edges, but solids have shared edges, check TopologyExplorer of Cube.step
    # what if the object is not NURBS?