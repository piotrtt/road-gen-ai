from src.road_components.definitions import StraightRoad
import scenariogeneration.xodr as xodr
from typing import Tuple


def generate_straight_road(component: StraightRoad, odr: xodr.OpenDrive, sequence_index: int) -> Tuple[xodr.OpenDrive, xodr.Road]:
    """
    Generates code for a straight road.
    """
    
    road1 = xodr.create_road(xodr.Line(component.length), id=1+sequence_index*10, left_lanes=component.left_lanes, right_lanes=component.right_lanes, lane_width=component.lane_width)
    odr.add_road(road1)
    continuation_road = road1
    return odr, continuation_road