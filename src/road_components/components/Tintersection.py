from src.road_components.definitions import TIntersection
import numpy as np
import scenariogeneration.xodr as xodr
import math
from src.road_components.helpers import curve_position_margin
from typing import Tuple


def generate_t_intersection(odr: xodr.OpenDrive, component: TIntersection, sequence_index: int) -> Tuple[xodr.OpenDrive, xodr.Road]:
    """
    Generates code for a T-intersection.
    """
    angle = component.angle
    lane_width = component.lane_width
    right_lanes = component.right_lanes
    left_lanes = component.left_lanes
    road1 = xodr.create_road(xodr.Line(20), id=1+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    road2 = xodr.create_road(xodr.Line(20), id=2+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    road3 = xodr.create_road(xodr.Line(20), id=3+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)

    angle_road2 = angle
    angle_road3 = angle + math.pi

    # initialize a variable for the CommonJunctionCreator
    junction_creator = xodr.CommonJunctionCreator(id = 100, name='my_junction')

    # Add roads to the junction. Pay attention to the start point coordinates. Don't forget that all the predecessors' coordinates start at x=0, y=0 where the successor (i.e. the first road) ends.
    # The coordinates create a margin so the roads don't overlap. The exact value depends on the number of lanes. The default is 10m, as shown in this example.
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x = 0,
                y = 0,
                heading=0,
                road_connection='successor')

    # road2 represents the left turn so both coordinates are positive
    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x = 2*lane_width*2*(abs(math.sin(angle_road2))), 
                y = 2*lane_width*2*(curve_position_margin(angle_road2)),
                heading=angle,
                road_connection='predecessor')

    # road3 follows the direction of road1, so the y coordinate is 0 and the x coordinate doubled
    junction_creator.add_incoming_road_cartesian_geometry(road3,
                x = 2*lane_width*2*(abs(math.sin(angle_road2))),
                y = 2*lane_width*2*(curve_position_margin(angle_road3)),
                heading=angle_road3,
                road_connection='predecessor')



    # add connection between the roads. If both roads have the same number of lanes, you don't need to specify the connections between particular lanes
    junction_creator.add_connection(road_one_id = road1.id,
                                    road_two_id = road3.id)

    junction_creator.add_connection(road_one_id = road1.id,
                                    road_two_id = road2.id)

    junction_creator.add_connection(road_one_id = road2.id,
                                    road_two_id = road3.id)


    # add all roads to the OpenDrive
    odr.add_road(road1)
    odr.add_road(road2)
    odr.add_road(road3)

    # add all junctions to the OpenDrive
    odr.add_junction_creator(junction_creator)
    continuation_road = road3
    
    return odr, continuation_road
