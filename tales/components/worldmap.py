from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Tuple

import noise
import numpy as np
from scipy import spatial, sparse
from scipy.sparse import linalg

from tales.components import Component

from tales.utils.math import distance

ndarr = np.ndarray


def copy_first_arg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self, first, *rest = args
        first = first.copy()
        return func(self, first, *rest, **kwargs)

    return wrapper


@dataclass
class Adjacency:
    adjacent_points: Dict[Any, Any]
    adjacent_vertices: Dict[Any, Any]  # adj_vxs
    region_idx_to_point_idx: Dict[Any, Any]  # vx_regions
    adjacency_map: ndarr  # adj_mat
    vertex_is_edge: np.array  # self.edge


class WorldMap(Component):
    def __init__(self, num_nodes: int = 1024, seed: int = 23):
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
        # only need the erodability to do the erosion again later
        self.elevation, self.erodability = self.generate_heightmap()

        # pdb.set_trace()

    def _generate_points(self) -> ndarr:
        np.random.seed(self.seed)
        points = np.random.random((self.number_points, 2))
        return points

    def _improve_points(self, points: ndarr, iterations=2) -> ndarr:
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

    def calculate_edges(self) -> ndarr:
        n = self.v_number_vertices
        edges = np.zeros(n, np.bool)
        for vertex_idx in range(n):
            adjs = self.v_adjacencies.adjacent_vertices[vertex_idx]
            if -1 in adjs:
                edges[vertex_idx] = True
        return edges

    def distorted_vertices(self, vertices: ndarr) -> ndarr:
        distorted_vertices = vertices.copy()
        noises = self.perlin_on_vertices(vertices)
        distorted_vertices[:, 0] += noises
        distorted_vertices[:, 1] += noises
        return distorted_vertices

    def perlin_on_vertices(self, vertices: ndarr) -> ndarr:
        assert vertices.shape[1] == 2
        base = np.random.randint(1000)
        return np.array([noise.pnoise2(x, y, lacunarity=1.7, octaves=3, base=base) for x, y in vertices])

    # HEIGHTMAP

    def generate_heightmap(self):
        elevation = np.zeros(self.v_number_vertices + 1)
        elevation[:-1] = 0.5 + ((self.v_distorted_vertices - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(self.v_vertices, 0.5)
        mountains = np.random.random((5, 2))

        # this doesn't seem to be doing much
        for m in mountains:
            elevation[:-1] += np.exp(-distance(self.v_vertices, m) ** 2 / 0.005) ** 2

        zero_mean_distortions = self.v_distorted_vertices - self.v_distorted_vertices.mean()
        random_1_2 = np.random.normal(0, 2, (1, 2))
        random_scalar = np.random.normal(0, 0.5)
        along = ((zero_mean_distortions * random_1_2).sum(1) + random_scalar) * 10
        erodability = np.exp(4 * np.arctan(along))

        for i in range(5):
            elevation = self.create_rift(elevation)
            elevation = self.relax(elevation)
        for i in range(5):
            elevation = self.relax(elevation)
        elevation = self.soften_elevation(elevation)

        raise_amount = np.random.randint(20, 40)
        elevation = self.raise_sealevel(elevation, raise_amount)
        elevation = self.erode(elevation, erodability, 1, 0.025)
        elevation = self.raise_sealevel(elevation, np.random.randint(raise_amount, raise_amount + 20))
        elevation = self.clean_coast(elevation)
        return elevation, erodability

    @copy_first_arg
    def create_rift(self, elevation: ndarr) -> ndarr:
        side = 5 * (distance(self.v_distorted_vertices, 0.5) ** 2 - 1)
        value = np.random.normal(0, 0.3)
        elevation[:-1] += np.arctan(side) * value
        return elevation

    @copy_first_arg
    def relax(self, elevation: ndarr) -> ndarr:
        """Modifies neighboring elevation vertices to be closer in height, smoothing heightmap"""
        newelev = np.zeros_like(elevation[:-1])  # can't i just make elevation have one vertice less???
        for u in range(self.v_number_vertices):
            adjs = [v for v in self.v_adjacencies.adjacent_vertices[u] if v != -1]
            if len(adjs) < 2:
                continue
            newelev[u] = np.mean(elevation[adjs])
        elevation[:-1] = newelev
        return elevation

    @copy_first_arg
    def soften_elevation(self, elevation: ndarr) -> ndarr:
        # noinspection PyArgumentList
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        assert 1 >= elevation.max() and elevation.min() >= 0
        elevation **= 0.5
        return elevation

    @copy_first_arg
    def raise_sealevel(self, elevation: ndarr, amount_percent: int) -> ndarr:
        assert 0 <= amount_percent <= 100
        maxheight = elevation.max()
        elevation -= np.percentile(elevation, amount_percent)
        elevation *= maxheight / elevation.max()
        elevation[-1] = 0
        return elevation

    @copy_first_arg
    def clean_coast(self, elevation: ndarr, n=3, outwards=True) -> ndarr:
        for _ in range(n):
            new_elev = elevation[:-1].copy()
            for u in range(self.v_number_vertices):
                if self.v_adjacencies.vertex_is_edge[u] or elevation[u] <= 0:
                    continue
                adjs = self.v_adjacencies.adjacent_vertices[u]
                adjelevs = elevation[adjs]
                if np.sum(adjelevs > 0) == 1:
                    new_elev[u] = np.mean(adjelevs[adjelevs <= 0])
            elevation[:-1] = new_elev
            if outwards:
                for u in range(self.v_number_vertices):
                    if self.v_adjacencies.vertex_is_edge[u] or elevation[u] > 0:
                        continue
                    adjs = self.v_adjacencies.adjacent_vertices[u]
                    adjelevs = elevation[adjs]
                    if np.sum(adjelevs <= 0) == 1:
                        new_elev[u] = np.mean(adjelevs[adjelevs > 0])
                elevation[:-1] = new_elev
        return elevation

    # EROSION
    def erode(self, elevation: ndarr, erodability: ndarr, iterations: int, rate: float = 0.01) -> ndarr:
        """ Removes landmass next to water

        :param elevation:
        :param erodability:
        :param iterations: how many iterations to run.
        :param rate: smaller rates lead to "thinner" erosion lines (river beds, ...)
        :return:
        """
        assert iterations > 0, "Need at least 1 iteration of erosion"
        elevation = elevation.copy()
        for _ in range(iterations):
            downhill = Mesh.erode_calc_downhill(elevation, self.v_adjacencies, self.v_number_vertices)
            flow = Mesh.erode_calc_flow(elevation, downhill, self.v_number_vertices)
            slopes = Mesh.erode_calc_slopes(elevation, downhill, self.v_vertices)
            elevation = Mesh.erode_step(elevation, erodability, flow, slopes, rate)
            elevation, downhill = self.erode_infill(elevation, downhill, self.v_adjacencies, self.v_number_vertices)
            elevation[-1] = 0
        return elevation

    @staticmethod
    def erode_calc_downhill(elevation: ndarr, adjacency: Adjacency, num_verts: int) -> ndarr:
        """Calculates a "downhill" array

        :param num_verts:
        :param adjacency:
        :param elevation:
        :return: array of length len(vertices)-1
        """
        dhidxs = np.argmin(elevation[adjacency.adjacency_map], 1)
        downhill = adjacency.adjacency_map[np.arange(num_verts), dhidxs]
        downhill[elevation[:-1] <= elevation[downhill]] = -1
        downhill[adjacency.vertex_is_edge] = -1
        return downhill

    @staticmethod
    def erode_calc_flow(elevation: ndarr, downhill: ndarr, num_verts: int) -> ndarr:
        rain = np.ones(num_verts) / num_verts
        i = downhill[downhill != -1]
        j = np.arange(num_verts)[downhill != -1]
        dmat = sparse.eye(num_verts) - sparse.coo_matrix((np.ones_like(i), (i, j)), (num_verts, num_verts)).tocsc()
        flow = linalg.spsolve(dmat, rain)
        flow[elevation[:-1] <= 0] = 0
        return flow

    @staticmethod
    def erode_calc_slopes(elevation: ndarr, downhill: ndarr, vertices: ndarr) -> ndarr:
        dist = distance(vertices, vertices[downhill, :])
        slope = (elevation[:-1] - elevation[downhill]) / (dist + 1e-9)
        slope[downhill == -1] = 0
        return slope

    @staticmethod
    def erode_step(elevation: ndarr, erodability: ndarr, flow: ndarr, slopes: ndarr, max_step: float) -> ndarr:
        elevation = elevation.copy()
        river_rate = -flow ** 0.5 * slopes  # river erosion
        slope_rate = -slopes ** 2 * erodability  # slope smoothing
        rate = 1000 * river_rate + slope_rate
        rate[elevation[:-1] <= 0] = 0
        elevation[:-1] += rate / np.abs(rate).max() * max_step
        return elevation

    @staticmethod
    def erode_infill(elevation: ndarr, downhill: ndarr, adj: Adjacency, num_verts: int) -> Tuple[ndarr, ndarr]:
        elevation = elevation.copy()
        while True:
            sinks = Mesh.get_sinks(elevation, downhill, adj)
            if np.all(sinks == -1):
                return elevation, downhill
            h, u, v = Mesh.find_lowest_sill(elevation, sinks, adj)
            sink = sinks[u]
            if downhill[v] != -1:
                elevation[v] = elevation[downhill[v]] + 1e-5
            sinkelev = elevation[:-1][sinks == sink]
            h = np.where(sinkelev < h, h + 1e-3 * (h - sinkelev), sinkelev) + 1e-5
            elevation[:-1][sinks == sink] = h
            downhill = Mesh.erode_calc_downhill(elevation, adj, num_verts)

    @staticmethod
    def get_sinks(elevation: ndarr, downhill: ndarr, adj: Adjacency) -> ndarr:
        sinks = downhill.copy()
        water = elevation[:-1] <= 0
        sinklist = np.where((sinks == -1) & ~water & ~adj.vertex_is_edge)[0]
        sinks[sinklist] = sinklist
        sinks[water] = -1
        while True:
            newsinks = sinks.copy()
            newsinks[~water] = sinks[sinks[~water]]
            newsinks[sinks == -1] = -1
            if np.all(sinks == newsinks):
                break
            sinks = newsinks
        return sinks

    @staticmethod
    def find_lowest_sill(elevation: ndarr, sinks: ndarr, adjacency: Adjacency) -> Tuple[float, int, int]:

        height = 10000
        maps = np.any((sinks[adjacency.adjacency_map] == -1) & adjacency.adjacency_map != -1, 1)
        edges = np.where((sinks != -1) & maps)[0]

        best_edge_corner = 0, 0
        for edge in edges:
            adjs = [v for v in adjacency.adjacent_vertices[edge] if v != -1]
            for adj in adjs:
                if sinks[adj] != -1:
                    continue
                newh = max(elevation[adj], elevation[edge])
                if newh >= height:
                    continue
                height = newh
                best_edge_corner = edge, adj
        assert height < 10000 and best_edge_corner != (0, 0)
        edge, adj = best_edge_corner

        return height, edge, adj
