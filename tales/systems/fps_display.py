import time
import pyglet

from tales.entities.entity import Entity
from tales.systems import System
from tales.systems.system import SystemType


class FPSDisplaySystem(System):
    COMPONENTS = []
    TYPE = SystemType.RENDERING

    def __init__(self, *args):
        super().__init__(*args)
        self.text_size = 16
        self.window_size = 0.9
        self.last_time = time.time_ns()
        self.last_window = 0

    def update_all(self, *args, **kwargs):
        now = time.time_ns()
        elapsed = now - self.last_time
        self.last_time = now

        fps = 1000 / (elapsed / 1000 ** 2)
        window = ((self.last_window * self.window_size) + (fps * (1 - self.window_size)))
        self.last_window = window

        pyglet.text.Label(
            f"fps: {window:.0f} (now:{fps:.2f})",
            font_name="Times New Roman",
            font_size=self.text_size,
            x=0, y=self.text_size,
            anchor_x="left", anchor_y="center").draw()

    def update(self, entity: Entity, *args, **kwargs):
        """Not needed since system doesn't rely on entities"""
        pass
