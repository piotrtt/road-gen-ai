from src.road_components.definitions import Curve
import numpy as np
import scenariogeneration.xodr as xodr
from src.road_components.helpers import curve_position_margin
import math
from typing import Tuple

def generate_curve(component: Curve, odr: xodr.OpenDrive, sequence_index: int) -> Tuple[xodr.OpenDrive, xodr.Road]:
    """
    Generates code for a curve.
    """
    angle = component.angle
    radius = component.radius
    lane_width = component.lane_width
    right_lanes = component.right_lanes
    left_lanes = component.left_lanes
    # Initialize variables for both roads of the curve. Specify the length (in this case 100m) and number of lanes in each direction.
    road1 = xodr.create_road(xodr.Line(1), id=1+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    road2 = xodr.create_road(xodr.Line(1), id=2+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)

    # set zero elevation and slope
    xodr.ElevationCalculator(road1).set_zero_elevation()
    xodr.ElevationCalculator(road2).set_zero_elevation()

    # Initialize a variable for the angle between the roads in radians. Remember that the scenariogeneration library counts the angle from the previous road, so when the user gives an angle value of, say, 30 degrees to the right, you must subtract this value from 180 getting 150 degrees. When the user gives a value of, say, 45 degrees to the left, you must add 180 degrees to this value, getting 225 degrees, and so on.
    # In this example the user wanted a 135 degree turn to the right, which translates to 45 degree in the scenariogeneration coordinates, because 180-135=45
    angle_road2 = angle

    # initialize a variable for the CommonJunctionCreator. Remember a curve is a junction of two roads.
    junction_creator = xodr.CommonJunctionCreator(id = 100+sequence_index*100, name=f'curve{sequence_index}', startnum=100+sequence_index*100)

    # add roads to the junction
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x = 0,
                y = 0,
                heading=0,
                road_connection='successor')

    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x = 2*lane_width*radius*(abs(math.sin(angle_road2))), #the '2*3*2' part is a calculation of the spacing needed to make the turn passable. The first '2' is the number of lanes and the '3' is the lane width, these values should be adjusted accordingly. The second '2' is the factor by which we always increase the result so that the lanes do not overlap.
                y = 2*lane_width*radius*(curve_position_margin(angle_road2)), #the '2*3*2' part is a calculation of the spacing needed to make the turn passable. The first '2' is the number of lanes and the '3' is the lane width, these values should be adjusted accordingly. The second '2' is the factor by which we always increase the result so that the lanes do not overlap.
                heading=angle_road2,
                road_connection='predecessor')

    # add connection between the roads. If both roads have the same number of lanes, you don't need to specify the connections between particular lanes, as it is shown in this example.
    junction_creator.add_connection(road_one_id = 1+sequence_index*10,
                                    road_two_id = 2+sequence_index*10)

    # add all roads to the OpenDrive
    odr.add_road(road1)
    odr.add_road(road2)

    # add all junctions to the OpenDrive
    odr.add_junction_creator(junction_creator)

    # specify road for continuation
    continuation_road = road2
    
    return odr, continuation_road
