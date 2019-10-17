from tales.components import Component
from tales.worldmap.mesh import MeshGenerator


class WorldMap(Component):
    def __init__(self, num_nodes: int = 1024, seed: int = 23):
        self.num_nodes = num_nodes
        self.mesh_gen = MeshGenerator(num_nodes, seed)
        self.mesh = self.mesh_gen.build_mesh()
