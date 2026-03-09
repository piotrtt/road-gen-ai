from src.road_components.definitions import Fork
import scenariogeneration.xodr as xodr
import math
from src.road_components.helpers import curve_position_margin
from typing import Tuple


def generate_fork(component: Fork, odr: xodr.OpenDrive, sequence_index: int) -> Tuple[xodr.OpenDrive, xodr.Road]:
    """
    Generates code for a road fork (one road splitting into two).
    """
    angle = component.angle
    lane_width = component.lane_width
    right_lanes = component.right_lanes
    left_lanes = component.left_lanes

    # Road 1 is the incoming road
    road1 = xodr.create_road(xodr.Line(20), id=1 + sequence_index * 10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    # Road 2 is the straight continuation
    road2 = xodr.create_road(xodr.Line(20), id=2 + sequence_index * 10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    # Road 3 branches off at the specified angle
    road3 = xodr.create_road(xodr.Line(20), id=3 + sequence_index * 10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)

    # set zero elevation and slope
    xodr.ElevationCalculator(road1).set_zero_elevation()
    xodr.ElevationCalculator(road2).set_zero_elevation()
    xodr.ElevationCalculator(road3).set_zero_elevation()

    # initialize a variable for the CommonJunctionCreator
    junction_creator = xodr.CommonJunctionCreator(id=100 + sequence_index * 100, name=f'fork{sequence_index}', startnum=100 + sequence_index * 100)

    # add the incoming road
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x=0,
                y=0,
                heading=0,
                road_connection='successor')

    # road2 continues straight ahead (heading = pi since it faces back toward the junction)
    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x=2 * lane_width * 2,
                y=0,
                heading=math.pi,
                road_connection='predecessor')

    # road3 branches off at the specified angle
    junction_creator.add_incoming_road_cartesian_geometry(road3,
                x=2 * lane_width * 2 * (abs(math.sin(angle))),
                y=2 * lane_width * 2 * (curve_position_margin(angle)),
                heading=angle,
                road_connection='predecessor')

    # connect incoming road to both outgoing roads
    junction_creator.add_connection(road_one_id=road1.id,
                                    road_two_id=road2.id)

    junction_creator.add_connection(road_one_id=road1.id,
                                    road_two_id=road3.id)

    # add all roads to the OpenDrive
    odr.add_road(road1)
    odr.add_road(road2)
    odr.add_road(road3)

    # add all junctions to the OpenDrive
    odr.add_junction_creator(junction_creator)

    # specify road for continuation (straight road continues)
    continuation_road = road2

    return odr, continuation_road
