from random import randint, seed

import numpy as np
import pyglet

from tales.components.worldmap import WorldMap
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType


def _col_from_number(number):
    number = int(number) * 255 * 3

    if number <= 255:
        return number, 0, 0
    number = number % 255
    if number <= 255:
        return 255, number, 0
    number = number % 255
    if number <= 255:
        return 255, 255, number
    raise ValueError(f"Number {number} too big")


def col_from_number(number):
    n = int(number * 255)
    return n, n, n


class MapDrawingSystem(System):
    COMPONENTS = [WorldMap]
    TYPE = SystemType.RENDERING

    def __init__(self, *args, draw_scale=800, centers=True, edges=True):
        super().__init__(*args)
        self.draw_scale = draw_scale
        self.centers = centers
        self.edges = edges

        self.cache = {}

    def update(self, entity: Entity, *args, **kwargs):
        map = entity.get_component_by_class(WorldMap)
        mesh = map.mesh

        for i, center in enumerate(mesh.points):
            # take the center point and all the vertices that define that points' "region"
            region = mesh.v_regions[i]
            vertices = map.mesh.v_vertices

            vertice_indicies = [region_vertex_idx for region_vertex_idx in region if region_vertex_idx != -1]
            verts = np.array([vertices[rvi] for rvi in vertice_indicies])

            drawable_poly = np.concatenate([center, verts.flatten(), verts.flatten()[:2]])
            amount = len(drawable_poly) // 2

            # assemble colors based on the elevation of the vertices we draw
            color_numbers = np.array([mesh.elevation[rvi] for rvi in vertice_indicies])
            mean = [color_numbers.mean()]  # use the mean as an approximation of the center
            color_numbers = np.concatenate([mean, color_numbers, [color_numbers[0]]])
            colors = np.array([col_from_number(cnn) for cnn in color_numbers]).flatten()


            seed(center[0] ** center[1])

            pyglet.graphics.vertex_list(
                amount,
                ('v2f/static', drawable_poly * self.draw_scale + 100),
                ('c3B/static', colors)
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)
