from random import randint, seed

import numpy as np
import pyglet

from tales.components.worldmap import WorldMap
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType


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
        vor = map.mesh.vor

        for center, point_region_idx in zip(vor.points, vor.point_region):
            # take the center point and all the vertices that define that points' "region"
            region = vor.regions[point_region_idx]
            verts = np.array([vor.vertices[region_vertex_idx] for region_vertex_idx in region if region_vertex_idx != -1])
            drawable_poly = np.concatenate([center, verts.flatten(), verts.flatten()[:2]])
            amount = len(drawable_poly)//2

            seed(center[0] ** center[1])
            pyglet.graphics.vertex_list(
                amount,
                ('v2f/static', drawable_poly * self.draw_scale + 100),
                ('c3B', (randint(0, 255), randint(0, 255), randint(0, 255)) * amount)
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)

