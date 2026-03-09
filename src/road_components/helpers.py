import numpy as np
import scenariogeneration.xodr as xodr
import math


def curve_position_margin(angle: float) -> float:
    """
    Calculate the margin for the curve position based on the angle and lane width.
    
    Args:
        angle: The angle of the curve in radians"""
    if angle < math.pi:
        return -math.cos(angle)-1
    elif angle > math.pi:
        return math.cos(angle)+1
    else:
        return 0



def simple_connection(odr: xodr.OpenDrive, junction_id: int, road_from: xodr.Road, road_to: xodr.Road) -> xodr.OpenDrive:
    """
    Create a simple junction connecting two roads.
    
    Args:
        junction_id: Unique ID for the junction
        road_from: The road ending at this junction (successor connection)
        road_to: The road starting at this junction (predecessor connection)
    """
    junction_creator = xodr.CommonJunctionCreator(id=junction_id, name='simple_junction', startnum=junction_id)
    junction_creator.add_incoming_road_cartesian_geometry(road_from,
                x=0,
                y=0,
                heading=0,
                road_connection='successor')
    junction_creator.add_incoming_road_cartesian_geometry(road_to,
                x=0.01,
                y=0,
                heading=np.pi,
                road_connection='predecessor')
    junction_creator.add_connection(road_one_id=road_from.id,
                                    road_two_id=road_to.id)
    odr.add_junction_creator(junction_creator)
    return odr