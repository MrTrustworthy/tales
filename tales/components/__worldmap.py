from random import randint
from typing import List

from tales.components import Component
import numpy as np
from scipy import spatial


class WorldMap(Component):
    def __init__(self, mode="test", num_nodes=1024):
        self.num_nodes = num_nodes

        points = WorldMap.build_grid(self.num_nodes)
        self._vor = spatial.Voronoi(points)

        # self._regions = [self.vor.regions[i] for i in self.vor.point_region]
        # self._vor_vertices = self.vor.vertices
        # self._num_vor_vertices = self.vor_vertices.shape[0]

        # self.build_adjs()
        # self.improve_vxs()
        # self.calc_edges()
        # self.distort_vxs()
        # self.elevation = np.zeros(self.nvxs + 1)
        # self.erodability = np.ones(self.nvxs)

        self.tiles = Tile.from_voronoi(self._vor)
        self.center_points = np.array([t.center for t in self.tiles]).flatten()
        self.edge_points = np.concatenate([t.vertices.flatten() for t in self.tiles])

    @staticmethod
    def build_grid(num_nodes, iterations=2):
        points = np.random.random((num_nodes, 2))
        # smoothing of points
        for _ in range(iterations):
            vor = spatial.Voronoi(points)
            newpts = []
            for idx in range(len(vor.points)):
                pt = vor.points[idx, :]
                region = vor.regions[vor.point_region[idx]]
                if -1 in region:
                    newpts.append(pt)
                else:
                    vxs = np.asarray([vor.vertices[i, :] for i in region])
                    vxs[vxs < 0] = 0
                    vxs[vxs > 1] = 1
                    newpt = np.mean(vxs, 0)
                    newpts.append(newpt)
            points = np.asarray(newpts)
        return points


class Tile:

    @classmethod
    def from_voronoi(cls, voronoi) -> List["Tile"]:
        tiles = []
        for center, point_region_idx in zip(voronoi.points, voronoi.point_region):
            # take the center point and all the vertices that define that points' "region"
            region = voronoi.regions[point_region_idx]
            verts = [voronoi.vertices[region_vertex_idx] for region_vertex_idx in region if region_vertex_idx != -1]
            vertices = np.array(verts)
            tile = cls(center, vertices)
            tiles.append(tile)
        return tiles

    def __init__(self, center, vertices):
        self.center = center
        self.vertices = vertices
        self.drawable_polygon = np.concatenate([self.center, self.vertices.flatten(), self.vertices.flatten()[:2]])
        self.drawable_poly_length = len(self.drawable_polygon) // 2
        self.drawable_poly_color = (randint(0, 255), randint(0, 255), randint(0, 255)) * self.drawable_poly_length

    def __repr__(self):
        return f'C: {self.center} [{self.vertices}]'

    def __hash__(self):
        return hash(str(self.center))