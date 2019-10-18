import numpy as np
from typing import Optional

from tales.worldmap.mesh import Mesh
from tales.worldmap.elevator import Elevator
from tales.worldmap.dataclasses import MapParameters


class MeshGenerator:
    def __init__(self, map_params: MapParameters):
        self.map_params = map_params

        self.mesh: Optional[Mesh] = None
        self.elevator: Optional[Elevator] = None
        self.build_mesh()

    def build_mesh(self) -> "Mesh":
        np.random.seed(self.map_params.seed)
        self.mesh = Mesh(self.map_params)
        self.elevator = Elevator(self.mesh, self.map_params)
        self.elevator.generate_heightmap()
        self.mesh.elevation = self.elevator.elevation

    def update_params(self, map_params: MapParameters):
        self.map_params = map_params
        self.build_mesh()
