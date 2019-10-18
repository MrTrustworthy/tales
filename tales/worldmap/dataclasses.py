import math

from typing import Dict, List

import numpy as np
from dataclasses import dataclass, field

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
    number_points: int = field(default=2048, metadata={
        "description": "Higher numbers create bigger maps",
        "ranges": {
            (256, 1024): "Small sized map",
            (1024, 4096): "Normal sized map",
            (4096, 8192): "Large sized map"
        }
    })
    seed: int = field(default=12, metadata={
        "description": "Random seed that makes map generation reproducible",
    })
    point_smoothing: int = field(default=2, metadata={
        "description": "Higher numbers move center points away from each other with decreasing effectiveness",
        "ranges": {
            (0, 1): "No smoothing, tiles may be very small or large",
            (1, 2): "Normal smoothing, tiles may vary but not by much",
            (2, 10): "Heavy smoothing, tiles are almost of the same size"
        }
    })
    number_mountains: int = field(default=50, metadata={
        "description": "Higher numbers create a more 'spotty' land area, at ~50 will often create inland seas",
        "ranges": {
            (0, 7): "Little inland seas, area mainly defined via rifts",
            (7, 30): "Smaller inland seas and smaller islands",
            (30, 70): "Multiple large inland seas and island chains"
        }
    })
    number_rifts: int = field(default=5, metadata={
        "description": "Higher numbers create a smoother side-to-side elevation profile, and less spots in the middle",
        "ranges": {
            (0, 5): "little end-to-end elevation shift, more inland seas and rugged coasts",
            (5, 50): "Flatter coastlines with clear end-to-end downhills, more bays but little inland seas",
            (50, 100): "Clear end-to-end downhill with very straight coast and mountain lines"
        }
    })
    elevation_softness: float = field(default=0.5, metadata={
        "description": "Higher numbers create a smoother, more balanced elevation profile. Will not change shore lines",
        "ranges": {
            (0.0, 0.25): "Little smoothing, large of high mountain ranges and deep seas",
            (0.25, 0.7): "Normal smoothing, some large mountains and smaller regions of deeper seas",
            (0.7, 2.5): "Heavy smoothing, elevation is very flat overall"
        }
    })
    coast_cleaning: int = field(default=3, metadata={
        "description": "Higher numbers create a smoother side-to-side elevation profile, and less spots in the middle",
        "ranges": {
            (0, 1): "Little cleaning, might have small spikes at the coast",
            (1, 4): "Normal cleaning, most spikes and tiny islands are removed",
            (4, 10): "High cleaning, should rarely result in any changes from normal cleaning"
        }
    })