from random import randint, seed

import numpy as np
import pyglet

from tales.components.worldmap import WorldMap, Mesh
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


def col_from_number(number, base=100):
    assert 0 <= base <= 255
    n = int(min(abs(number) * base + base, 255))
    if number < 0:
        return (0, 0, n)
    return (n, n, 0)


class MapDrawingSystem(System):
    COMPONENTS = [WorldMap]
    TYPE = SystemType.RENDERING

    def __init__(self, *args, draw_scale=800, centers=True, edges=True):
        super().__init__(*args)
        self.draw_scale = draw_scale
        self.centers = centers
        self.edges = edges

        self.step = 0

    def update(self, entity: Entity, *args, **kwargs):
        self.step += 1
        map = entity.get_component_by_class(WorldMap)
        mesh = map.mesh
        #mesh.elevation = mesh.erode(mesh.elevation, mesh.erodability, 10, 0.01)

        self.draw_map(mesh)

    def draw_map(self, mesh: Mesh):

        for i, center in enumerate(mesh.points):
            # take the center point and all the vertices that define that points' "region"
            region = mesh.v_regions[i]
            vertices = mesh.v_vertices

            vertice_indicies = [region_vertex_idx for region_vertex_idx in region if region_vertex_idx != -1]
            verts = np.array([vertices[rvi] for rvi in vertice_indicies])

            drawable_poly = np.concatenate([center, verts.flatten(), verts.flatten()[:2]])
            amount = len(drawable_poly) // 2

            # assemble colors based on the elevation of the vertices we draw
            color_numbers = np.array([mesh.elevation[rvi] for rvi in vertice_indicies])
            mean = [np.median(color_numbers)]  # use the median as an approximation of the center

            color_numbers = np.concatenate([mean, color_numbers, [color_numbers[0]]])
            colors = np.array([col_from_number(cnn) for cnn in color_numbers]).flatten()

            pyglet.graphics.vertex_list(
                amount,
                ('v2f/static', drawable_poly * self.draw_scale + 100),
                ('c3B/static', colors)
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)

        if self.centers:
            draw_points = mesh.points.flatten() * self.draw_scale + 100
            point_amount = len(draw_points) // 2
            pyglet.graphics.draw(
                point_amount,
                pyglet.gl.GL_POINTS,
                ('v2f', draw_points),
                ('c3B', (255, 0, 0) * point_amount)
            )
