from tales.components.component import Component
from tales.utils.math import Vector2


class Position(Component):
    def __init__(self, x: int, y: int):
        # noinspection PyArgumentList
        self.position = Vector2(x, y)

    def set_to(self, new_pos: Vector2):
        self.position.update(new_pos)

    def reset(self, movement: "Movement"):
        if movement.last_step is None:
            raise ValueError("Can't reset position to the last one since there's no last position!")
        self.position.update(movement.last_step)


class Movement(Component):
    def __init__(self, x: int, y: int, speed: int):
        # noinspection PyArgumentList
        assert speed > 0, "Speed must be >0"
        # noinspection PyArgumentList
        self.target = Vector2(x, y)
        self.speed = speed
        self.last_step = None  # Needed to reset on collision


class Collider(Component):

    def __init__(self, x: int):
        self.size = x
