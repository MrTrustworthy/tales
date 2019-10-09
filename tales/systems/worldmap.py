from random import randint, seed

import numpy as np
import pyglet

from tales.components.worldmap import WorldMap
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType


def col_from_number(number):
    number = int(number)
    if number <= 255:
        return number, 0, 0
    number = number % 255
    if number <= 255:
        return 255, number, 0
    number = number % 255
    if number <= 255:
        return 255, 255, number
    raise ValueError(f"Number {number} too big")


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
            color_number = np.array([mesh.elevation[rvi] for rvi in vertice_indicies])
            mean = [color_number.mean()]  # use the mean as an approximation of the center
            color_number = np.concatenate([mean, color_number, [color_number[0]]])
            color_number_norm = (color_number - mesh.elevation.min()) / (mesh.elevation.max() - mesh.elevation.min())
            colors = np.array([col_from_number(cnn * 255 * 3) for cnn in color_number_norm]).flatten()

            seed(center[0] ** center[1])

            pyglet.graphics.vertex_list(
                amount,
                ('v2f/static', drawable_poly * self.draw_scale + 100),
                ('c3B', colors)
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)
