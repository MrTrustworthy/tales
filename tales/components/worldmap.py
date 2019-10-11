from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import noise
import numpy as np
from scipy import spatial

from tales.components import Component
import pdb

from tales.utils.math import distance


@dataclass
class Adjacency:
    adjacent_points: Dict[Any, Any]
    adjacent_vertices: Dict[Any, Any]  # adj_vxs
    region_idx_to_point_idx: Dict[Any, Any]  # vx_regions
    adjacency_map: Dict[Any, Any]
    vertex_is_edge: Dict[int, bool]  # self.edge


class WorldMap(Component):
    def __init__(self, num_nodes: int = 1024, seed: int = 89):
        self.num_nodes = num_nodes
        self.mesh = Mesh(number_points=num_nodes, seed=seed)


class Mesh:
    def __init__(self, number_points: int, seed: int):
        self.number_points = number_points
        self.seed = seed

        self.points = self._improve_points(self._generate_points())

        # points
        #       Coordinates of input points.
        # vertices
        #       Coordinates of the Voronoi vertices.
        # ridge_points
        #       Indices of the points between which each Voronoi ridge lies.
        # ridge_vertices
        #       Indices of the Voronoi vertices forming each Voronoi ridge.
        # regions
        #       Indices of the Voronoi vertices forming each Voronoi region.
        #       -1 indicates vertex outside the Voronoi diagram.
        # point_region
        #       Index of the Voronoi region for each input point.
        #       If qhull option “Qc” was not specified, the list will contain -1 for points
        #       that are not associated with a Voronoi region.
        self.vor = spatial.Voronoi(self.points)

        # v_regions map the index of a point in self.points to a region
        self.v_regions = [self.vor.regions[idx] for idx in self.vor.point_region]
        self.v_vertices = self.vor.vertices  # vxs
        self.v_number_vertices = self.v_vertices.shape[0]

        # adjacencies give us maps that we can use to quickly look up nodes that belong together
        self.v_adjacencies = self._build_adjacencies()
        self.remove_outliers()
        self.v_adjacencies.vertex_is_edge = self.calculate_edges()
        self.v_distorted_vertices = self.distorted_vertices(self.v_vertices)
        self.elevation = np.zeros(self.v_number_vertices + 1)
        self.elevation, self.erodability = self.generate_heightmap()

        # pdb.set_trace()

    def _generate_points(self) -> np.ndarray:
        np.random.seed(self.seed)
        points = np.random.random((self.number_points, 2))
        return points

    def _improve_points(self, points: np.ndarray, iterations=2) -> np.ndarray:
        """Moves points a little further away from each other to make all 'tiles' more equal in size and spread"""
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

    def _build_adjacencies(self) -> Adjacency:

        adjacent_points = defaultdict(list)
        adjacent_vertices = defaultdict(list)
        region_idx_to_point_idx = defaultdict(list)
        adjacency_map = np.zeros((self.v_number_vertices, 3), np.int32) - 1

        # find all points that are neighbouring a different point
        for p1, p2 in self.vor.ridge_points:
            adjacent_points[p1].append(p2)
            adjacent_points[p2].append(p1)

        # find all ridge vertices that are neighbouring a different ridge vertice
        for v1, v2 in self.vor.ridge_vertices:
            adjacent_vertices[v1].append(v2)
            adjacent_vertices[v2].append(v1)

        for k, v in adjacent_vertices.items():
            if k != -1:
                adjacency_map[k, :] = v

        # build a region-point-index to point-index map
        for point_idx in range(self.number_points):
            region = self.v_regions[point_idx]
            for region_point_idx in region:
                if region_point_idx == -1:
                    continue
                region_idx_to_point_idx[region_point_idx].append(point_idx)

        return Adjacency(
            adjacent_points,
            adjacent_vertices,
            region_idx_to_point_idx,
            adjacency_map,
            None  # can we set a default in the dataclass? this is overwritten later
        )

    def remove_outliers(self):
        # The Voronoi algorithm will create points outside of [0, 1] at the very edges
        # we want to remove them or the map might extend far, far beyond its borders
        for vertex_idx in range(self.v_number_vertices):
            point = self.points[self.v_adjacencies.region_idx_to_point_idx[vertex_idx]]
            self.v_vertices[vertex_idx, :] = np.mean(point, 0)

    def calculate_edges(self):
        n = self.v_number_vertices
        edges = np.zeros(n, np.bool)
        for vertex_idx in range(n):
            adjs = self.v_adjacencies.adjacent_vertices[vertex_idx]
            if -1 in adjs:
                edges[vertex_idx] = True
        return edges

    def distorted_vertices(self, vertices: np.ndarray):
        distorted_vertices = vertices.copy()
        noises = self.perlin_on_vertices(vertices)
        distorted_vertices[:, 0] += noises
        distorted_vertices[:, 1] += noises
        return distorted_vertices

    def perlin_on_vertices(self, vertices: np.ndarray):
        assert vertices.shape[1] == 2
        base = np.random.randint(1000)
        return np.array([noise.pnoise2(x, y, lacunarity=1.7, octaves=3, base=base) for x, y in vertices])

    def generate_heightmap(self):
        elevation = np.zeros(self.v_number_vertices + 1)
        elevation[:-1] = 0.5 + ((self.v_distorted_vertices - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(self.v_vertices, 0.5)
        mountains = np.random.random((5, 2))

        # this doesn't seem to be doing much
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.v_vertices, m) ** 2 / 0.005) ** 2

        zero_mean_distortions = self.v_distorted_vertices - self.v_distorted_vertices.mean()
        random_1_2 = np.random.normal(0, 2, (1, 2))
        random_scalar = np.random.normal(0, 0.5)
        along = ((zero_mean_distortions * random_1_2).sum(1) + random_scalar) * 10
        erodability = np.exp(4 * np.arctan(along))

        for i in range(5):
            self.create_rift(elevation)
            self.relax(elevation)
        for i in range(5):
            self.relax(elevation)
        self.soften_elevation(elevation)

        return elevation, erodability

    def create_rift(self, elevation):
        side = 5 * (distance(self.v_distorted_vertices, 0.5) ** 2 - 1)
        value = np.random.normal(0, 0.3)
        elevation[:-1] += np.arctan(side) * value

    def relax(self, elevation: np.ndarray):
        """Modifies neighboring elevation vertices to be closer in height, smoothing heightmap"""
        newelev = np.zeros_like(elevation[:-1])
        for u in range(self.v_number_vertices):
            adjs = [v for v in self.v_adjacencies.adjacent_vertices[u] if v != -1]
            if len(adjs) < 2:
                continue
            newelev[u] = np.mean(elevation[adjs])
        elevation[:-1] = newelev

    def soften_elevation(self, elevation: np.ndarray):
        # noinspection PyArgumentList
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        assert 1 >= elevation.max() and elevation.min() >= 0
        elevation **= 0.5

    def shore_heightmap(self):

        n = self.nvxs
        self.elevation = np.zeros(n + 1)
        self.elevation[:-1] = 0.5 + ((self.dvxs - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        self.elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(self.vxs, 0.5)
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2

        along = (((self.dvxs - 0.5) * np.random.normal(0, 2, (1, 2))).sum(1) + np.random.normal(0, 0.5)) * 10

        for i in range(5):
            self.rift()
            self.relax()
        for i in range(5):
            self.relax()
        self.soften_elevation()

        sealevel = np.random.randint(20, 40)
        self.raise_sealevel(sealevel)
        self.do_erosion(100, 0.025)

        self.raise_sealevel(np.random.randint(sealevel, sealevel + 20))
        self.clean_coast()
