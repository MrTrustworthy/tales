from abc import abstractmethod
from enum import Enum
from typing import List, Type

from tales.components import Component
from tales.entities.entity import Entity
from tales.world import World


class SystemType(Enum):
    GAMEPLAY = "Gameplay"
    RENDERING = "Rendering"


class System:
    COMPONENTS: List[Type[Component]] = []
    TYPE: SystemType = SystemType.GAMEPLAY

    def __init__(self, world: World):
        self.world: World = world
        self.system_type = self.TYPE  # is set on the subclass
        self.components: List[Type[Component]] = self.COMPONENTS  # is set on the subclass

    def update_all(self, *args, **kwargs):
        for entity in self.world.get_entities_with_components(self.components):
            self.update(entity, *args, **kwargs)

    @abstractmethod
    def update(self, entity: Entity, *args, **kwargs):
        pass
