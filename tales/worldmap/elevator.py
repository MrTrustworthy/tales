import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from typing import Optional, Tuple, List

from tales.utils.math import distance
from tales.worldmap.dataclasses import MapParameters, Adjacency, ndarr
from tales.worldmap.mesh import Mesh


class Elevator:
    def __init__(self, mesh: Mesh, map_params: MapParameters):
        self.mesh = mesh
        self.map_params = map_params

        # elevation on index i is the elevation of v_vertices[i]
        self.elevation: Optional[ndarr] = None

        # TODO
        self.base_erodability: Optional[ndarr] = None

        # city locations given as indicies into v_vertices
        self.cities: Optional[ndarr] = None

        # elevation_pts on index i is the elevation of center_points[i]
        self.elevation_pts: Optional[ndarr] = None

        # flow on index i is the cumulative water flow of all flows leading downward to v_vertices[i]
        self.flow: Optional[ndarr] = None

        # downhill on index i is the index of the lowest neighbour of v_vertices[i]
        self.downhill: Optional[ndarr] = None

    def generate_heightmap(self):
        num_verts = self.mesh.v_number_vertices
        num_points = self.mesh.number_points
        adj = self.mesh.v_adjacencies
        vertice_noise = self.mesh.v_vertice_noise
        verts = self.mesh.v_vertices
        regions = self.mesh.v_regions
        params = self.map_params

        # Outlines
        elevation = Elevator._create_baseline_elevation(vertice_noise, verts, num_verts)
        elevation = Elevator._create_hills(elevation, verts, params)
        elevation = Elevator._create_rifting(elevation, adj, vertice_noise, num_verts, params)
        elevation = Elevator._soften_elevation(elevation, params)

        # clear sinks to make water flow possible
        downhill = Elevator._calc_downhill(elevation, adj, num_verts)
        elevation, downhill = Elevator._infill(elevation, downhill, adj, num_verts)

        # Erosion
        base_erodability = Elevator._create_base_erodability(vertice_noise)
        for _ in range(params.erosion_iterations):
            downhill = Elevator._calc_downhill(elevation, adj, num_verts)
            flow = Elevator._calc_flow(elevation, downhill, num_verts)
            slopes = Elevator._calc_slopes(elevation, downhill, verts)
            elevation = Elevator._erode_step(elevation, base_erodability, flow, slopes, params.erosion_rate)
            elevation, downhill = Elevator._infill(elevation, downhill, adj, num_verts)
            elevation[-1] = 0

        # clean up coastlines
        elevation = Elevator._raise_sealevel(elevation, params.percent_sea)
        elevation = Elevator._clean_coast(elevation, adj, num_verts, params)

        # one last step of sink filling
        elevation, downhill = Elevator._infill(elevation, downhill, adj, num_verts)

        # create the rivers
        rivers = Elevator._create_rivers(elevation, downhill, params.number_rivers)

        # let's see about that
        flow = Elevator._calc_flow(elevation, downhill, num_verts)
        elevation_pts = Elevator._calc_elevation_pts(num_points, regions, elevation)

        cities = Elevator._place_cities(params.number_cities, elevation, verts, flow, params)
        self.elevation, self.elevation_pts, self.base_erodability, self.cities, self.flow, self.downhill, self.rivers = \
            elevation, elevation_pts, base_erodability, cities, flow, downhill, rivers

    @staticmethod
    def _place_cities(
            num_cities: int, elevation: ndarr, verts: ndarr, flow: ndarr, params: MapParameters
    ) -> List[ndarr]:
        city_score = flow ** 0.5
        city_score[elevation[:-1] <= 0] = -9999999
        cities = []
        while len(cities) < num_cities:
            newcity = np.argmax(city_score)
            if np.random.random() < (len(cities) + 1) ** -0.2 and \
                    0.1 < verts[newcity, 0] < 0.9 and \
                    0.1 < verts[newcity, 1] < 0.9:
                cities.append(newcity)
            city_score -= params.city_spacing / (distance(verts, verts[newcity, :]) + 1e-9)
        return cities

    @staticmethod
    def _create_rivers(elevation: ndarr, downhill: ndarr, num_rivers: int) -> List[List[int]]:

        def get_downstream_neighbour(idx: int, exclude: List[int]) -> Optional[int]:
            lowest_neighbour = downhill[idx]
            return lowest_neighbour if lowest_neighbour != -1 and lowest_neighbour not in exclude else None

        get_river_starting_points = lambda: np.argsort(elevation)[::-1]
        is_already_river = lambda i: i in (r for ri in rivers for r in ri)
        is_below_water = lambda i: elevation[i] <= 0

        rivers = []

        # create all rivers
        for start_idx in get_river_starting_points():
            current = start_idx

            if is_already_river(start_idx) or elevation[start_idx] <= 0:
                continue

            # create a single river
            river = [current]
            while True:
                highest_neighbor = get_downstream_neighbour(current, river)
                # if there's no free neighbor, then stop there
                if highest_neighbor in (None, -1):
                    break
                river.append(highest_neighbor)
                # once river flows into the ocean, stop
                if is_below_water(highest_neighbor) or is_already_river(highest_neighbor):
                    break
                current = highest_neighbor

            rivers.append(river)
            if len(rivers) >= num_rivers:
                break

        return rivers

    @staticmethod
    def _create_baseline_elevation(vertice_noise: ndarr, verts: ndarr, num_verts: int) -> ndarr:
        """Creates a baseline elevation based on the randomized vertice noise"""
        elevation = np.zeros(num_verts + 1)
        elevation[:-1] = 0.5 + ((vertice_noise - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(verts, 0.5)
        return elevation

    @staticmethod
    def _create_base_erodability(vertice_noise: ndarr) -> ndarr:
        zero_mean_vertice_noise = vertice_noise - vertice_noise.mean()
        random_1_2 = np.random.normal(0, 2, (1, 2))
        random_scalar = np.random.normal(0, 0.5)
        along = ((zero_mean_vertice_noise * random_1_2).sum(1) + random_scalar) * 10
        base_erodability = np.exp(4 * np.arctan(along))
        return base_erodability

    @staticmethod
    def _create_hills(elevation: ndarr, verts: ndarr, params: MapParameters) -> ndarr:
        """Creates elevated "spots" of land that will be harder to erode and may form mountains/islands"""
        elevation = elevation.copy()
        hills = np.random.random((params.number_hills, 2))
        for hill_position in hills:
            elevation[:-1] += np.exp(-distance(verts, hill_position) ** 2 / 0.005) ** 2
        return elevation

    @staticmethod
    def _create_rifting(
            elevation: ndarr, adj: Adjacency, vertice_noise: ndarr, num_verts: int, params: MapParameters
    ) -> ndarr:
        for i in range(params.number_rifts):
            elevation = Elevator._create_rift(elevation, vertice_noise)
            elevation = Elevator._relax(elevation, adj, num_verts)
        elevation = Elevator._relax(elevation, adj, num_verts)
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
    def _soften_elevation(elevation: ndarr, params: MapParameters) -> ndarr:
        """ Soft-normalizes elevation

        Elevation will be normalized to a range between 0 and 1,
        with values being scaled to be closer to 1 via sqrt(normalized_val) at
        """
        # noinspection PyArgumentList
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        assert 1 >= elevation.max() and elevation.min() >= 0
        elevation **= params.elevation_softness
        return elevation

    @staticmethod
    def _raise_sealevel(elevation: ndarr, amount_percent: int) -> ndarr:
        """ Moves amount_percent of points under water, then scales everything up to the old max vals again
        """
        assert 0 <= amount_percent <= 100
        elevation = elevation.copy()
        maxheight = elevation.max()
        elevation -= np.percentile(elevation, amount_percent)
        elevation *= maxheight / elevation.max()
        elevation[-1] = 0
        return elevation

    @staticmethod
    def _clean_coast(elevation: ndarr, adj: Adjacency, num_verts: int, params: MapParameters) -> ndarr:
        """Removes sharp spikes at coasts that are removed by water crashing into it

        Basic principle: If all but one neighbors of a vertice are under water, lower it down to their mean
        This may also remove small/tiny islands

        This function quickly reaches a satisfying result at ~5 iterations,
        after which no more changes will be performed.
        """
        elevation = elevation.copy()
        for _ in range(params.coast_cleaning):

            new_elev = elevation[:-1].copy()

            # lower ground vertices at the coast
            for u in range(num_verts):
                # ignore edges and under-water vertices
                if adj.vertex_is_edge[u] or elevation[u] <= 0:
                    continue
                adjs = adj.adjacent_vertices[u]
                neighbour_elevations = elevation[adjs]
                if np.sum(neighbour_elevations > 0) == 1:
                    new_elev[u] = np.mean(neighbour_elevations[neighbour_elevations <= 0])
            elevation[:-1] = new_elev

            # raise water vertices that are surrounded by ground
            for u in range(num_verts):
                # ignore edges and above-water vertices
                if adj.vertex_is_edge[u] or elevation[u] > 0:
                    continue
                adjs = adj.adjacent_vertices[u]
                neighbour_elevations = elevation[adjs]
                if np.sum(neighbour_elevations <= 0) == 1:
                    new_elev[u] = np.mean(neighbour_elevations[neighbour_elevations > 0])
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
        is_downhill = downhill[downhill != -1]
        j = np.arange(num_verts)[downhill != -1]
        dense_matrix = (sparse.eye(num_verts) - sparse.coo_matrix((np.ones_like(is_downhill), (is_downhill, j)),
                                                                  (num_verts, num_verts)).tocsc())
        flow = linalg.spsolve(dense_matrix, rain)
        flow[elevation[:-1] <= 0] = 0
        return flow

    @staticmethod
    def _calc_slopes(elevation: ndarr, downhill: ndarr, vertices: ndarr) -> ndarr:
        dist = distance(vertices, vertices[downhill, :])
        slope = (elevation[:-1] - elevation[downhill]) / (dist + 1e-9)
        slope[downhill == -1] = 0
        return slope

    @staticmethod
    def _erode_step(elevation: ndarr, base_erodability: ndarr, flow: ndarr, slopes: ndarr,
                    erosion_rate: float) -> ndarr:
        elevation = elevation.copy()
        river_rate = -flow ** 0.5 * slopes  # river erosion
        slope_rate = -slopes ** 2 * base_erodability  # slope smoothing
        combined_rate = 1000 * river_rate + slope_rate
        combined_rate[elevation[:-1] <= 0] = 0

        erosion = combined_rate / np.abs(combined_rate).max()  # array of length num_vertices with values -1.0 to 0
        elevation[:-1] += erosion * erosion_rate
        return elevation

    @staticmethod
    def _infill(elevation: ndarr, downhill: ndarr, adjacency: Adjacency, num_verts: int) -> Tuple[ndarr, ndarr]:
        """This function fills in all the sinks

        At the end, from each point, there's a chain of downhill points leading to the edge of the map
        """
        elevation = elevation.copy()
        while True:
            sinks = Elevator._get_sinks(elevation, downhill, adjacency)
            if np.all(sinks == -1):
                return elevation, downhill
            height, edge, neighbour = Elevator._get_lowest_sill(elevation, sinks, adjacency)
            sink = sinks[edge]
            if downhill[neighbour] != -1:
                elevation[neighbour] = elevation[downhill[neighbour]] + 1e-5
            sinkelev = elevation[:-1][sinks == sink]
            height = np.where(sinkelev < height, height + 1e-3 * (height - sinkelev), sinkelev) + 1e-5
            elevation[:-1][sinks == sink] = height
            downhill = Elevator._calc_downhill(elevation, adjacency, num_verts)

    @staticmethod
    def _get_sinks(elevation: ndarr, downhill: ndarr, adj: Adjacency) -> ndarr:
        """Sinks are points not at the edges which are lower than all their neighbors"""
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

        height = 10000000
        maps = np.any((sinks[adjacency.adjacency_map] == -1) & adjacency.adjacency_map != -1, 1)
        edges = np.where((sinks != -1) & maps)[0]

        best_edge_corner = 0, 0
        for edge in edges:
            neighbours = [v for v in adjacency.adjacent_vertices[edge] if v != -1]
            for neighbour in neighbours:
                if sinks[neighbour] != -1:
                    continue
                newh = max(elevation[neighbour], elevation[edge])
                if newh >= height:
                    continue
                height = newh
                best_edge_corner = edge, neighbour
        assert height < 10000000 and best_edge_corner != (0, 0)
        edge, neighbour = best_edge_corner

        return height, edge, neighbour

    @staticmethod
    def _calc_elevation_pts(num_points: int, regions: List[List[int]], elevation: ndarr) -> ndarr:
        elevation_pts = np.zeros(num_points)
        for p in range(num_points):
            if regions[p]:
                elevation_pts[p] = np.mean(elevation[regions[p]])
        return elevation_pts
