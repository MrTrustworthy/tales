import math
from itertools import chain
from typing import Tuple

Color = Tuple[int, int, int]


def get_circle_points(center_x: float, center_y: float, radius: float,
                      num_segments: int = 128):
    """
    Draw a filled-in circle.

    :param float center_x: x position that is the center of the circle.
    :param float center_y: y position that is the center of the circle.
    :param float radius: width of the circle.
    :param int num_segments: float of triangle segments that make up this
         circle. Higher is better quality, but slower render time.
    """

    unrotated_point_list = []

    for segment in range(num_segments):
        theta = 2.0 * 3.1415926 * segment / num_segments

        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        unrotated_point_list.append((x, y))

    uncentered_point_list = []
    for point in unrotated_point_list:
        uncentered_point_list.append(rotate_point(point[0], point[1], 0, 0, 0))

    point_list = []
    for point in uncentered_point_list:
        point_list.append((point[0] + center_x, point[1] + center_y))

    vertex_indicies = list(int(x) for x in chain.from_iterable(point_list))

    return vertex_indicies


def rotate_point(x: float, y: float, cx: float, cy: float,
                 angle: float) -> (float, float):
    """
    Rotate a point around a center.

    :param x: x value of the point you want to rotate
    :param y: y value of the point you want to rotate
    :param cx: x value of the center point you want to rotate around
    :param cy: y value of the center point you want to rotate around
    :param angle: Angle, in degrees, to rotate
    :return: Return rotated (x, y) pair
    :rtype: (float, float)
    """
    temp_x = x - cx
    temp_y = y - cy

    # now apply rotation
    rotated_x = temp_x * math.cos(math.radians(angle)) - temp_y * math.sin(math.radians(angle))
    rotated_y = temp_x * math.sin(math.radians(angle)) + temp_y * math.cos(math.radians(angle))

    # translate back
    rounding_precision = 2
    x = round(rotated_x + cx, rounding_precision)
    y = round(rotated_y + cy, rounding_precision)

    return x, y
