from collections import defaultdict

import noise
import numpy as np
from dataclasses import dataclass
from scipy import spatial, sparse
from scipy.sparse import linalg
from typing import Dict, Tuple, Optional, List

from tales.utils.math import distance

ndarr = np.ndarray
IntListDict = Dict[int, List[int]]


@dataclass(frozen=True)
class Adjacency:
    adjacent_points: IntListDict
    adjacent_vertices: IntListDict  # adj_vxs
    region_idx_to_point_idx: IntListDict  # vx_regions
    adjacency_map: ndarr  # adj_mat
    vertex_is_edge: np.array  # self.edge


@dataclass(frozen=True)
class MapParameters:
    number_points: int = 1024
    seed: int = 23
    point_smoothing: int = 2  # moves center points away from each other. decreasing effectiveness above ~4
    number_mountains: int = 5  # 80  # higher numbers create a more "spotty" land area
    number_rifts: int = 10  # higher numbers create a smoother side-to-side elevation profile, and less middle spots


class MeshGenerator:

    def __init__(self, map_params: MapParameters):
        self.map_params = map_params

        self.mesh: Optional[Mesh] = None
        self.elevator: Optional[Elevator] = None

    def build_mesh(self) -> "Mesh":
        np.random.seed(self.map_params.seed)
        self.mesh = Mesh(self.map_params)
        self.elevator = Elevator(self.mesh, self.map_params)
        self.elevator.generate_heightmap()
        self.update_elevation()
        return self.mesh

    def update_elevation(self):
        self.mesh.elevation = self.elevator.elevation


class Mesh:
    def __init__(self, map_params: MapParameters):
        self.map_params = map_params
        self.number_points = self.map_params.number_points

        self.center_points = self._generate_points(self.map_params.point_smoothing)

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
        self.vor = spatial.Voronoi(self.center_points)

        # v_regions map the index of a point in self.center_points to a region
        self.v_regions: List[List[int]] = [self.vor.regions[idx] for idx in self.vor.point_region]

        self.v_number_vertices = self.vor.vertices.shape[0]

        # adjacencies give us maps that we can use to quickly look up nodes that belong together
        self.v_adjacencies = self._calculate_adjacencies()

        # all the vertices we need, aka the points that separate one region (based on center points) from others
        self.v_vertices = self._remove_outliers(self.vor.vertices)

        self.v_vertice_noise = self._vertice_noise(self.v_vertices)

        self.elevation: Optional[ndarr] = None

    def _generate_points(self, iterations: int) -> ndarr:
        points = np.random.random((self.number_points, 2))

        # Moves points a little further away from each other to make all 'tiles' more equal in size and spread
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

    def _calculate_adjacencies(self) -> Adjacency:

        adjacent_points: IntListDict = defaultdict(list)
        adjacent_vertices: IntListDict = defaultdict(list)
        region_idx_to_point_idx: IntListDict = defaultdict(list)
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
            self._calculate_edges(adjacent_vertices)
        )

    def _remove_outliers(self, vertices: ndarr) -> ndarr:
        # The Voronoi algorithm will create points outside of [0, 1] at the very edges
        # we want to remove them or the map might extend far, far beyond its borders
        vertices = vertices.copy()
        for vertex_idx in range(self.v_number_vertices):
            point = self.center_points[self.v_adjacencies.region_idx_to_point_idx[vertex_idx]]
            vertices[vertex_idx, :] = np.mean(point, 0)
        return vertices

    def _calculate_edges(self, adjacent_vertices: IntListDict) -> ndarr:
        n = self.v_number_vertices
        edges = np.zeros(n, np.bool)
        for vertex_idx in range(n):
            adjs = adjacent_vertices[vertex_idx]
            if -1 in adjs:
                edges[vertex_idx] = True
        return edges

    def _vertice_noise(self, vertices: ndarr) -> ndarr:
        assert vertices.shape[1] == 2
        vertice_noise = self.v_vertices.copy()
        # perlin noise on vertices
        base = np.random.randint(1000)
        noises = np.array([noise.pnoise2(x, y, lacunarity=1.7, octaves=3, base=base) for x, y in vertices])
        vertice_noise[:, 0] += noises
        vertice_noise[:, 1] += noises
        return vertice_noise


class Elevator:

    def __init__(self, mesh: Mesh, map_params: MapParameters):
        self.mesh = mesh
        self.map_params = map_params

        self.elevation: Optional[ndarr] = None
        self.erodability: Optional[ndarr] = None

    def generate_heightmap(self) -> ndarr:
        num_verts = self.mesh.v_number_vertices
        num_points = self.mesh.number_points
        adj = self.mesh.v_adjacencies
        vertice_noise = self.mesh.v_vertice_noise
        verts = self.mesh.v_vertices
        regions = self.mesh.v_regions
        params = self.map_params

        elevation = Elevator._create_baseline_elevation(vertice_noise, verts, num_verts)

        # this doesn't seem to be doing much
        elevation = Elevator._create_mountains(elevation, verts, params)

        elevation = Elevator._create_rifting(elevation, adj, vertice_noise, num_verts, params)

        raise_amount = np.random.randint(20, 40)
        elevation = Elevator._raise_sealevel(elevation, raise_amount)

        erodability = Elevator._create_erodability(vertice_noise)
        elevation = Elevator._erode(elevation, erodability, adj, verts, num_verts, 1, 0.025)

        raise_amount = np.random.randint(raise_amount, raise_amount + 20)
        elevation = Elevator._raise_sealevel(elevation, raise_amount)

        elevation = Elevator._clean_coast(elevation, adj, num_verts, 3, True)

        downhill = Elevator._calc_downhill(elevation, adj, num_verts)
        elevation, downhill = Elevator._infill(elevation, downhill, adj, num_verts)
        downhill = Elevator._calc_downhill(elevation, adj, num_verts)
        flow = Elevator._calc_flow(elevation, downhill, num_verts)
        slopes = Elevator._calc_slopes(elevation, downhill, verts)
        elevation_pts = Elevator._calc_elevation_pts(num_points, regions, elevation)

        self.elevation, self.erodability = elevation, erodability
        return elevation

    @staticmethod
    def _erode(
            elevation: ndarr, erodability: ndarr,
            adj: Adjacency, verts: ndarr, num_verts: int,
            iterations: int, rate: float) -> ndarr:
        """ Removes landmass next to water

        Smaller rates lead to "thinner" erosion lines (river beds, ...)
        """
        assert iterations > 0, "Need at least 1 iteration of erosion"
        elevation = elevation.copy()
        for _ in range(iterations):
            downhill = Elevator._calc_downhill(elevation, adj, num_verts)
            flow = Elevator._calc_flow(elevation, downhill, num_verts)
            slopes = Elevator._calc_slopes(elevation, downhill, verts)
            elevation = Elevator._erode_step(elevation, erodability, flow, slopes, rate)
            elevation, downhill = Elevator._infill(elevation, downhill, adj, num_verts)
            elevation[-1] = 0
        return elevation

    @staticmethod
    def _create_baseline_elevation(vertice_noise: ndarr, verts: ndarr, num_verts: int) -> ndarr:
        """Creates a baseline elevation based on the randomized vertice noise"""
        elevation = np.zeros(num_verts + 1)
        elevation[:-1] = 0.5 + ((vertice_noise - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(verts, 0.5)
        return elevation

    @staticmethod
    def _create_erodability(vertice_noise: ndarr) -> ndarr:
        zero_mean_vertice_noise = vertice_noise - vertice_noise.mean()
        random_1_2 = np.random.normal(0, 2, (1, 2))
        random_scalar = np.random.normal(0, 0.5)
        along = ((zero_mean_vertice_noise * random_1_2).sum(1) + random_scalar) * 10
        erodability = np.exp(4 * np.arctan(along))
        return erodability

    @staticmethod
    def _create_mountains(elevation: ndarr, verts: ndarr, params: MapParameters) -> ndarr:
        """Creates elevated "spots" of land that will be harder to erode"""
        elevation = elevation.copy()
        mountains = np.random.random((params.number_mountains, 2))
        for m in mountains:
            elevation[:-1] += np.exp(-distance(verts, m) ** 2 / 0.005) ** 2
        return elevation

    @staticmethod
    def _create_rifting(
            elevation: ndarr, adj: Adjacency, vertice_noise: ndarr, num_verts: int, params: MapParameters
    ) -> ndarr:
        for i in range(params.number_rifts):
            elevation = Elevator._create_rift(elevation, vertice_noise)
            elevation = Elevator._relax(elevation, adj, num_verts)
        elevation = Elevator._relax(elevation, adj, num_verts)
        elevation = Elevator._soften_elevation(elevation)
        return elevation

    @staticmethod
    def _create_rift(elevation: ndarr, vertice_noise: ndarr) -> ndarr:
        elevation = elevation.copy()
        side = 5 * (distance(vertice_noise, 0.5) ** 2 - 1)
        value = np.random.normal(0, 0.3)
        elevation[:-1] += np.arctan(side) * value
        return elevation

    @staticmethod
    def _relax(elevation: ndarr, adj: Adjacency, num_verts: int) -> ndarr:
        """Modifies neighboring elevation vertices to be closer in height, smoothing heightmap
        """
        elevation = elevation.copy()
        newelev = np.zeros_like(elevation[:-1])
        for u in range(num_verts):
            adjs = [v for v in adj.adjacent_vertices[u] if v != -1]
            if len(adjs) < 2:
                continue
            newelev[u] = np.mean(elevation[adjs])
        elevation[:-1] = newelev
        return elevation

    @staticmethod
    def _soften_elevation(elevation: ndarr) -> ndarr:
        """ Soft-normalizes elevation

        Elevation will be normalized to a range between 0 and 1,
        with values being scaled to be closer to 1 via sqrt(normalized_val)
        """
        # noinspection PyArgumentList
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        assert 1 >= elevation.max() and elevation.min() >= 0
        elevation **= 0.5
        return elevation

    @staticmethod
    def _raise_sealevel(elevation: ndarr, amount_percent: int) -> ndarr:
        assert 0 <= amount_percent <= 100
        elevation = elevation.copy()
        maxheight = elevation.max()
        elevation -= np.percentile(elevation, amount_percent)
        elevation *= maxheight / elevation.max()
        elevation[-1] = 0
        return elevation

    @staticmethod
    def _clean_coast(elevation: ndarr, adj: Adjacency, num_verts: int, iterations: int, outwards: bool) -> ndarr:
        elevation = elevation.copy()
        for _ in range(iterations):
            new_elev = elevation[:-1].copy()
            for u in range(num_verts):
                if adj.vertex_is_edge[u] or elevation[u] <= 0:
                    continue
                adjs = adj.adjacent_vertices[u]
                adjelevs = elevation[adjs]
                if np.sum(adjelevs > 0) == 1:
                    new_elev[u] = np.mean(adjelevs[adjelevs <= 0])
            elevation[:-1] = new_elev
            if outwards:
                for u in range(num_verts):
                    if adj.vertex_is_edge[u] or elevation[u] > 0:
                        continue
                    adjs = adj.adjacent_vertices[u]
                    adjelevs = elevation[adjs]
                    if np.sum(adjelevs <= 0) == 1:
                        new_elev[u] = np.mean(adjelevs[adjelevs > 0])
                elevation[:-1] = new_elev
        return elevation

    # EROSION
    @staticmethod
    def _calc_downhill(elevation: ndarr, adjacency: Adjacency, num_verts: int) -> ndarr:
        """Calculates a "downhill" array

        returns an array of length len(vertices)-1
        """
        dhidxs = np.argmin(elevation[adjacency.adjacency_map], 1)
        downhill = adjacency.adjacency_map[np.arange(num_verts), dhidxs]
        downhill[elevation[:-1] <= elevation[downhill]] = -1
        downhill[adjacency.vertex_is_edge] = -1
        return downhill

    @staticmethod
    def _calc_flow(elevation: ndarr, downhill: ndarr, num_verts: int) -> ndarr:
        rain = np.ones(num_verts) / num_verts
        i = downhill[downhill != -1]
        j = np.arange(num_verts)[downhill != -1]
        dmat = sparse.eye(num_verts) - sparse.coo_matrix((np.ones_like(i), (i, j)), (num_verts, num_verts)).tocsc()
        flow = linalg.spsolve(dmat, rain)
        flow[elevation[:-1] <= 0] = 0
        return flow

    @staticmethod
    def _calc_slopes(elevation: ndarr, downhill: ndarr, vertices: ndarr) -> ndarr:
        dist = distance(vertices, vertices[downhill, :])
        slope = (elevation[:-1] - elevation[downhill]) / (dist + 1e-9)
        slope[downhill == -1] = 0
        return slope

    @staticmethod
    def _erode_step(elevation: ndarr, erodability: ndarr, flow: ndarr, slopes: ndarr, max_step: float) -> ndarr:
        elevation = elevation.copy()
        river_rate = -flow ** 0.5 * slopes  # river erosion
        slope_rate = -slopes ** 2 * erodability  # slope smoothing
        rate = 1000 * river_rate + slope_rate
        rate[elevation[:-1] <= 0] = 0
        elevation[:-1] += rate / np.abs(rate).max() * max_step
        return elevation

    @staticmethod
    def _infill(elevation: ndarr, downhill: ndarr, adj: Adjacency, num_verts: int) -> Tuple[ndarr, ndarr]:
        elevation = elevation.copy()
        while True:
            sinks = Elevator._get_sinks(elevation, downhill, adj)
            if np.all(sinks == -1):
                return elevation, downhill
            h, u, v = Elevator._get_lowest_sill(elevation, sinks, adj)
            sink = sinks[u]
            if downhill[v] != -1:
                elevation[v] = elevation[downhill[v]] + 1e-5
            sinkelev = elevation[:-1][sinks == sink]
            h = np.where(sinkelev < h, h + 1e-3 * (h - sinkelev), sinkelev) + 1e-5
            elevation[:-1][sinks == sink] = h
            downhill = Elevator._calc_downhill(elevation, adj, num_verts)

    @staticmethod
    def _get_sinks(elevation: ndarr, downhill: ndarr, adj: Adjacency) -> ndarr:
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
    def _get_lowest_sill(elevation: ndarr, sinks: ndarr, adjacency: Adjacency) -> Tuple[float, int, int]:

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

    @staticmethod
    def _calc_elevation_pts(num_points: int, regions: List[List[int]], elevation: ndarr) -> ndarr:
        elevation_pts = np.zeros(num_points)
        for p in range(num_points):
            if regions[p]:
                elevation_pts[p] = np.mean(elevation[regions[p]])
        return elevation_pts
