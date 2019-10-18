from typing import List, Type

from tales.components import Component
from tales.entities.entity import Entity


class World:
    def __init__(self, entities: List[Entity]):
        self.entities = entities

    def get_entities_with_components(self, components: List[Type[Component]]) -> List[Entity]:
        return [e for e in self.entities if e.has_components(components)]
