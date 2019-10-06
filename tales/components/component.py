class Component:
    def __repr__(self):
        return f"Component{{{self.__class__}}}"


class NoSuchComponentException(Exception):
    pass


class ComponentAlreadyExistsException(Exception):
    pass
