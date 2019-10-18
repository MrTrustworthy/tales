from tales.components import Component
from tales.worldmap.dataclasses import MapParameters
from tales.worldmap.mesh_generator import MeshGenerator


class WorldMap(Component):
    def __init__(self):
        params = MapParameters()
        self.mesh_gen = MeshGenerator(params)

