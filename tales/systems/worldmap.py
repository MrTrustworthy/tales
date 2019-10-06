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

        for tile in map.tiles:
            pyglet.graphics.vertex_list(
                tile.drawable_poly_length,
                ('v2f/static', tile.drawable_polygon * self.draw_scale + 100),
                ('c3B', tile.drawable_poly_color)
            ).draw(pyglet.gl.GL_TRIANGLE_FAN)

        # batches should be faster, this doesn't work yet
        # batch = pyglet.graphics.Batch()
        # for tile in map.tiles:
        #     batch.add(
        #         tile.drawable_poly_length,
        #         pyglet.gl.GL_TRIANGLE_FAN,
        #         None,
        #         ('v2f/static', tile.drawable_polygon * self.draw_scale + 100),
        #         ('c3B', tile.drawable_poly_color)
        #     )
        # batch.draw()

        if self.centers:
            draw_points = map.center_points * self.draw_scale + 100
            point_amount = len(draw_points) // 2
            pyglet.graphics.draw(
                point_amount,
                pyglet.gl.GL_POINTS,
                ('v2f', draw_points),
                ('c3B', (255, 255, 255) * point_amount)
            )

        if self.edges:
            vor_edges = map.edge_points * self.draw_scale + 100
            edge_amount = len(vor_edges) // 2
            pyglet.graphics.draw(
                edge_amount,
                pyglet.gl.GL_POINTS,
                ('v2f', vor_edges),
                ('c3B', (0, 0, 255) * edge_amount)
            )
