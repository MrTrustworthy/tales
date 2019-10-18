from tales.components.component import Component
import pyglet


class Drawable(Component):
    def __init__(self, name: str):
        self.image = pyglet.resource.image(f"resources/{name}.png")
        self.image.anchor_x = self.image.width // 2
        self.image.anchor_y = self.image.height // 2
