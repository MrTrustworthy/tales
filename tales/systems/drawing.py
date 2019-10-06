import pyglet

from tales.components import Drawable, Position, Collider
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType
from tales.utils.graphics import get_circle_points


class UnitDrawingSystem(System):
    COMPONENTS = [Drawable, Position, Collider]
    TYPE = SystemType.RENDERING

    def __init__(self, *args, show_hitbox_circles=False):
        super().__init__(*args)
        self.show_hitbox_circles = show_hitbox_circles

    def update(self, entity: Entity, *args, **kwargs):
        draw = entity.get_component_by_class(Drawable)
        pos = entity.get_component_by_class(Position)
        collider = entity.get_component_by_class(Collider)

        if self.show_hitbox_circles:
            point_list = get_circle_points(pos.position.x, pos.position.y, collider.size // 2)
            pyglet.graphics.draw(len(point_list) // 2, pyglet.gl.GL_TRIANGLE_FAN, ("v2i", point_list))

        draw.image.blit(pos.position.x, pos.position.y)
