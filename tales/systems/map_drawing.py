import numpy as np
import pyglet
from pyglet.gl import *
from typing import Tuple, List, Dict, Optional

from tales.components.worldmap import WorldMap
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType
from tales.worldmap.dataclasses import MapParameters
from tales.worldmap.mesh import Mesh
from tales.worldmap.mesh_generator import MeshGenerator

CITY_IMAGE = pyglet.resource.image(f"resources/castle.png")
CITY_IMAGE.anchor_x = CITY_IMAGE.width // 2
CITY_IMAGE.anchor_y = CITY_IMAGE.height // 2


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


def col_from_number(number, base=100) -> Tuple[int, int, int]:
    assert 0 <= base <= 255
    n = int(min(abs(number) * base + base, 255))
    if number < 0:
        return 0, 0, n
    elif number > 0.7:
        return n, n, n
    else:
        return n, n, 0


class MapDrawingSystem(System):
    COMPONENTS = [WorldMap]
    TYPE = SystemType.RENDERING

    def __init__(self, *args, draw_scale=800, centers=False, cities=True):
        super().__init__(*args)
        self.draw_scale = draw_scale
        self.centers = centers
        self.cities = cities

        self.step = 0

    def update(self, entity: Entity, *args, **kwargs):
        self.step += 1
        map = entity.get_component_by_class(WorldMap)
        self.draw_map(map.mesh_gen)
        self.draw_centers(map.mesh_gen.mesh)
        # self.draw_cities(map.mesh_gen)
        self.draw_rivers(map.mesh_gen)

        # self._draw_attribute(map.mesh_gen, "flow")
        self._shift_parameter(map.mesh_gen, "number_rivers", 2)

    def draw_map(self, mesh_gen: MeshGenerator):
        mesh = mesh_gen.mesh

        for i, center in enumerate(mesh.center_points):
            # take the center point and all the vertices that define that points' "region"
            region = mesh.v_regions[i]
            vertices = mesh.v_vertices

            vertice_indicies = [vertex_idx for vertex_idx in region if vertex_idx != -1]
            verts = np.array([vertices[rvi] for rvi in vertice_indicies])

            drawable_poly = np.concatenate([center, verts.flatten(), verts.flatten()[:2]])
            amount = len(drawable_poly) // 2

            # assemble colors based on the elevation of the vertices we draw
            color_numbers = np.array([mesh.elevation[rvi] for rvi in vertice_indicies])
            center_color_num = [mesh_gen.elevator.elevation_pts[i]]

            color_numbers = np.concatenate([center_color_num, color_numbers, [color_numbers[0]]])
            colors = np.array([col_from_number(cnn) for cnn in color_numbers]).flatten()

            pyglet.graphics.vertex_list(
                amount,
                ("v2f/static", drawable_poly * self.draw_scale + 100),
                ("c3B/static", colors),
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)

    def draw_rivers(self, mesh_gen: MeshGenerator):
        rivers = mesh_gen.elevator.rivers

        for i, river in enumerate(rivers):
            river_verts = np.array([mesh_gen.mesh.v_vertices[r_idx] for r_idx in river]).flatten()
            amount = len(river_verts) // 2
            pyglet.gl.glLineWidth(3)
            pyglet.graphics.vertex_list(
                amount,
                ("v2f/static", river_verts * self.draw_scale + 100),
                ("c3B/static", (59, 179, 208) * amount),
            ).draw(pyglet.gl.GL_LINE_STRIP)

            pyglet.gl.glPointSize(3)
            pyglet.graphics.vertex_list(
                1,
                ("v2f", river_verts[:2] * self.draw_scale + 100),
                ("c3B", (255, 0, 0)),
            ).draw(pyglet.gl.GL_POINTS)

    def draw_cities(self, mesh_gen: MeshGenerator):
        if not self.cities:
            return
        draw_points = np.array([mesh_gen.mesh.v_vertices[i] for i in mesh_gen.elevator.cities])

        for x, y in draw_points * self.draw_scale + 100:
            CITY_IMAGE.blit(x, y)

    def draw_centers(self, mesh: Mesh):
        if not self.centers:
            return
        draw_points = mesh.center_points.flatten() * self.draw_scale + 100
        point_amount = len(draw_points) // 2
        pyglet.graphics.draw(
            point_amount,
            pyglet.gl.GL_POINTS,
            ("v2f", draw_points),
            ("c3B", (255, 0, 0) * point_amount),
        )

    def _draw_attribute(self, mesh_gen: MeshGenerator, attribute: str):
        """Draws a certain vertex attribute with redd-ish color"""
        attr = getattr(mesh_gen.elevator, attribute)
        values_norm = (attr - attr.min()) / (attr.max() - attr.min())
        for i, vert in enumerate(mesh_gen.mesh.v_vertices):
            pyglet.gl.glPointSize(3)
            pyglet.graphics.draw(
                1,
                pyglet.gl.GL_POINTS,
                ("v2f", vert * self.draw_scale + 100),
                ("c3B", (int(values_norm[i] * 205) + 49, 0, 0)),
            )

    def _shift_parameter(self, mesh_gen: MeshGenerator, property: str, factor: float = 1.0):
        mesh_gen.update_params(MapParameters(**{property: factor * self.step}))
        print(f"Shifted {property} to {factor * self.step}")
