from src.road_components.definitions import LaneSwitch
import scenariogeneration.xodr as xodr
import math
from typing import Tuple


def generate_lane_switch(component: LaneSwitch, odr: xodr.OpenDrive, sequence_index: int) -> Tuple[xodr.OpenDrive, xodr.Road]:
    """
    Generates code for a lane switch (road segment where the number of lanes changes).
    """
    lane_width = component.lane_width
    right_lanes_in = component.right_lanes
    left_lanes_in = component.left_lanes
    right_lanes_out = component.right_lanes_out
    left_lanes_out = component.left_lanes_out

    # Road 1 is the incoming road with the original lane configuration
    road1 = xodr.create_road(xodr.Line(20), id=1 + sequence_index * 10, left_lanes=left_lanes_in, right_lanes=right_lanes_in, lane_width=lane_width)
    # Road 2 is the outgoing road with the new lane configuration
    road2 = xodr.create_road(xodr.Line(20), id=2 + sequence_index * 10, left_lanes=left_lanes_out, right_lanes=right_lanes_out, lane_width=lane_width)

    # set zero elevation and slope
    xodr.ElevationCalculator(road1).set_zero_elevation()
    xodr.ElevationCalculator(road2).set_zero_elevation()

    # initialize a variable for the CommonJunctionCreator
    junction_creator = xodr.CommonJunctionCreator(id=100 + sequence_index * 100, name=f'lane_switch{sequence_index}', startnum=100 + sequence_index * 100)

    # add roads to the junction. Both roads are aligned straight ahead.
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x=0,
                y=0,
                heading=0,
                road_connection='successor')

    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x=2 * lane_width * 2,
                y=0,
                heading=math.pi,
                road_connection='predecessor')

    # When lane counts differ, specify individual lane connections.
    # Lanes are numbered from the center: right lanes are positive (1, 2, ...) and left lanes are negative (-1, -2, ...).
    # Connect the minimum overlapping lanes on each side.
    right_common = min(right_lanes_in, right_lanes_out)
    left_common = min(left_lanes_in, left_lanes_out)

    lanes_from = []
    lanes_to = []
    for i in range(1, right_common + 1):
        lanes_from.append(i)
        lanes_to.append(i)
    for i in range(1, left_common + 1):
        lanes_from.append(-i)
        lanes_to.append(-i)

    junction_creator.add_connection(road_one_id=road1.id,
                                    road_two_id=road2.id,
                                    lane_one_id=lanes_from,
                                    lane_two_id=lanes_to)

    # add all roads to the OpenDrive
    odr.add_road(road1)
    odr.add_road(road2)

    # add all junctions to the OpenDrive
    odr.add_junction_creator(junction_creator)

    # specify road for continuation
    continuation_road = road2

    return odr, continuation_road
