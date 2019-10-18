from tales.components import Component
from tales.worldmap.dataclasses import MapParameters
from tales.worldmap.mesh_generator import MeshGenerator


class WorldMap(Component):
    def __init__(self):
        self.mesh_gen = MeshGenerator(MapParameters())
        self.mesh = self.mesh_gen.build_mesh()
