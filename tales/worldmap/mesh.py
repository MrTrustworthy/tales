from collections import defaultdict

import noise
import numpy as np
from scipy import spatial
from typing import Optional, List

from tales.worldmap.dataclasses import Adjacency, MapParameters, ndarr, IntListDict


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
            self._calculate_edges(adjacent_vertices),
        )

    def _remove_outliers(self, vertices: ndarr) -> ndarr:
        # The Voronoi algorithm will create points outside of [0, 1] at the very edges
        # we want to remove them or the map might extend far, far beyond its borders
        vertices = vertices.copy()
        for vertex_idx in range(self.v_number_vertices):
            point = self.center_points[
                self.v_adjacencies.region_idx_to_point_idx[vertex_idx]
            ]
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
        noises = np.array(
            [
                noise.pnoise2(x, y, lacunarity=1.7, octaves=3, base=base)
                for x, y in vertices
            ]
        )
        vertice_noise[:, 0] += noises
        vertice_noise[:, 1] += noises
        return vertice_noise
