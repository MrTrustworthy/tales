import math
from typing import Union

import numpy as np

Number = Union[int, float]


def distance(a, b):
    disp2 = (a - b) ** 2
    assert len(disp2.shape) == 2
    return np.sum(disp2, 1) ** 0.5



class Vector2:

    def __init__(self, x: Number, y: Number) -> None:
        self.x = x
        self.y = y

    def update(self, new: "Vector2") -> None:
        self.x = new.x
        self.y = new.y

    def distance_to(self, other: "Vector2") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> "Vector2":
        mag = self.magnitude
        self.x = self.x / mag
        self.y = self.y / mag
        return self

    def __sub__(self, other: "Vector2"):
        return Vector2(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Vector2"):
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, val: Number):
        assert isinstance(val, (int, float)), "Value must be a number!"
        return Vector2(self.x * val, self.y * val)
