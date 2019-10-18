from typing import Dict, List

import numpy as np
from dataclasses import dataclass

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
