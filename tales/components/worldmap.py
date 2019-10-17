from tales.components import Component
from tales.worldmap.mesh import MeshGenerator, MapParameters


class WorldMap(Component):
    def __init__(self):
        self.mesh_gen = MeshGenerator(MapParameters())
        self.mesh = self.mesh_gen.build_mesh()
