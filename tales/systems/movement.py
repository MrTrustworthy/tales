from tales.components import Position, Movement
from tales.entities.entity import Entity
from tales.systems.system import System, SystemType


class MovementSystem(System):
    COMPONENTS = [Position, Movement]
    TYPE = SystemType.GAMEPLAY

    def update(self, entity: Entity, *args, dt=None, **kwargs):
        pos_comp = entity.get_component_by_class(Position)
        target_comp = entity.get_component_by_class(Movement)

        p = pos_comp.position
        t = target_comp.target

        target_comp.last_step = p

        # if no more distance is needed, just set it down right now
        # removing the component is only done in the step _AFTER_ the destination is reached
        # this allows eg. for collision to reset the position or do something with the target_comp
        if p.distance_to(t) == 0:
            entity.delete_component(target_comp)
            return

        move_distance = target_comp.speed * dt

        # Don't overshoot on the last step, just set position to destination
        if p.distance_to(t) <= move_distance:
            pos_comp.set_to(t)
            return

        new_pos = p + ((t - p).normalize() * move_distance)
        pos_comp.set_to(new_pos)
