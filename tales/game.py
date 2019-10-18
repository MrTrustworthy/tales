import pyglet

from tales.components import Drawable, Position, Collider, Movement
from tales.components.worldmap import WorldMap
from tales.entities.entity import Entity
from tales.systems import UnitDrawingSystem, MovementSystem, CollisionSystem
from tales.systems.fps_display import FPSDisplaySystem
from tales.systems.system import SystemType
from tales.systems.map_drawing import MapDrawingSystem
from tales.world import World


def make_players():
    return [Entity([
        Drawable("player"),
        Position(50, 50),
        Collider(32),
        Movement(950, 800, 100)
    ]), Entity([
        Drawable("player"),
        Position(950, 50),
        Collider(32),
        Movement(50, 800, 100)
    ]), Entity([
        WorldMap()
    ])]


def make_world():
    entities = make_players()

    return World(entities)


def make_systems(world):
    systems = [
        MapDrawingSystem(world),
        UnitDrawingSystem(world, show_hitbox_circles=True),
        MovementSystem(world),
        CollisionSystem(world),
        FPSDisplaySystem(world)
    ]
    return systems


class Game:
    def __init__(self):
        self.window_size = [1920, 1080]
        self.fps = 60  # setting it over 60 seems to have no effect - is refresh rate considered in clock.schedule?
        self.window = pyglet.window.Window(width=self.window_size[0], height=self.window_size[1], caption="Tales")

    def run(self):
        world = make_world()
        systems = make_systems(world)

        def run_step(dt):
            for system in systems:
                if system.system_type == SystemType.GAMEPLAY:
                    system.update_all(dt=dt)

        pyglet.clock.schedule_interval(run_step, 1 / self.fps)

        @self.window.event("on_draw")
        def draw():
            self.window.clear()
            for system in systems:
                if system.system_type == SystemType.RENDERING:
                    system.update_all()

        pyglet.app.run()
