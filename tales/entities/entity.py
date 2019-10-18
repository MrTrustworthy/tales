import itertools
from typing import Type, TypeVar, Dict, List
from uuid import uuid4

from tales.components.component import (
    Component,
    NoSuchComponentException,
    ComponentAlreadyExistsException,
)

ComponentVar = TypeVar("ComponentVar")


class Entity:
    def __init__(self, components=None):
        self.id = uuid4()
        self._components: Dict[Type[ComponentVar], ComponentVar] = {}
        if isinstance(components, (list, tuple)):
            for component in components:
                self.add_component(component)
        elif components is not None:
            raise ValueError("Components must be an list or tuple")

    @property
    def components(self):
        return itertools.chain.from_iterable(self._components.values())

    def add_component(self, component: Component, replace_if_exists=True):
        cls = type(component)
        if not replace_if_exists and cls in self._components.keys():
            raise ComponentAlreadyExistsException(cls)
        self._components[cls] = component

    def has_components(self, components: List[Type[Component]]) -> bool:
        return set(self._components.keys()).issuperset(set(components))

    def delete_component(self, component: Component):
        del self._components[type(component)]

    def get_component_by_class(self, cls: Type[ComponentVar]) -> ComponentVar:
        try:
            return self._components[cls]
        except KeyError:
            raise NoSuchComponentException(cls)

    def __repr__(self):
        return f"Entity{{{self.id}}}[{[repr(c) for c in self._components]}]"
