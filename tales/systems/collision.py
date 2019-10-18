from tales.components import Position, Movement, Collider
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType


class CollisionSystem(System):
    COMPONENTS = [Collider, Movement]
    TYPE = SystemType.GAMEPLAY

    def update(self, entity: Entity, *args, dt=None, **kwargs):
        pos = entity.get_component_by_class(Position)
        collider = entity.get_component_by_class(Collider)
        movement = entity.get_component_by_class(Movement)

        others = [
            (e.get_component_by_class(Collider), e.get_component_by_class(Position))
            for e in self.world.get_entities_with_components([Collider, Position])
            if e is not entity
        ]

        for other_collider, other_pos in others:
            if not (pos.position.distance_to(other_pos.position) < collider.size + other_collider.size):
                continue

            pos.position.update(movement.last_step)
            entity.delete_component(movement)
            print(f"Collision of {entity} at {pos}, resetting position and clearing movement")
            return
