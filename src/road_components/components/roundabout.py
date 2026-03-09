import numpy as np
import scenariogeneration.xodr as xodr
import os
import math
from typing import List, Tuple
from random import randint
from src.road_components.definitions import Roundabout
from src.road_components.helpers import curve_position_margin, simple_connection



def create_component(odr: xodr.OpenDrive, component: Roundabout, sequence_index, iteration_index=1, previous_road: xodr.Road | None = None) -> Tuple[xodr.Road, xodr.Road, xodr.Road]:
    radius = component.radius
    num_exits = component.num_exits
    lane_width = component.lane_width
    arm_length = component.arm_length
    right_lanes = component.right_lanes
    left_lanes = component.left_lanes
    print(f"Creating component {iteration_index} with previous road: {previous_road.id if previous_road else 'None'}")
    road1 = xodr.create_road(xodr.Line(1), id=1+sequence_index*100+iteration_index*10, left_lanes=0, right_lanes=right_lanes, lane_width=lane_width)
    road2 = xodr.create_road(xodr.Line(arm_length), id=2+sequence_index*100+iteration_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    road3 = xodr.create_road(xodr.Arc(1/(radius+num_exits*(2*2*lane_width+1)/2/math.pi), angle=2*np.pi/num_exits), id=3+sequence_index*100+iteration_index*10, left_lanes=0, right_lanes=right_lanes, lane_width=lane_width)

    angle_road2 = math.pi/2
    angle_road3 = math.pi

    # initialize a variable for the CommonJunctionCreator
    junction_creator = xodr.CommonJunctionCreator(id = 1000+iteration_index*100, name='my_junction', startnum=1000+iteration_index*100)

    # Add roads to the junction. Pay attention to the start point coordinates. Don't forget that all the predecessors' coordinates start at x=0, y=0 where the successor (i.e. the first road) ends.
    # The coordinates create a margin so the roads don't overlap. The exact value depends on the number of lanes. The default is 10m, as shown in this example.
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x = 0,
                y = 0,
                heading=0,
                road_connection='successor')

    # road2 represents the left turn so both coordinates are positive
    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x = 2*lane_width*(abs(math.sin(angle_road2))), 
                y = 2*lane_width*(curve_position_margin(angle_road2)),
                heading=angle_road2,
                road_connection='predecessor')

    # road3 follows the direction of road1, so the y coordinate is 0 and the x coordinate doubled
    junction_creator.add_incoming_road_cartesian_geometry(road3,
                x = 2*2*lane_width*(abs(math.sin(angle_road2))),
                y = 2*lane_width*(curve_position_margin(angle_road3)),
                heading=angle_road3,
                road_connection='predecessor')



    # add connection between the roads. If both roads have the same number of lanes, you don't need to specify the connections between particular lanes
    junction_creator.add_connection(road_one_id = road1.id,
                                    road_two_id = road2.id,
                                    lane_one_id=-1,
                                    lane_two_id=-1)

    junction_creator.add_connection(road_one_id = road2.id,
                                    road_two_id = road3.id,
                                    lane_one_id=1,
                                    lane_two_id=-1)

    junction_creator.add_connection(road_one_id = road1.id,
                                    road_two_id = road3.id,)


    # add all roads to the OpenDrive
    odr.add_road(road1)
    odr.add_road(road2)
    odr.add_road(road3)

    # add junction to the OpenDrive
    odr.add_junction_creator(junction_creator)
    
    if previous_road:
        simple_connection(odr, 1000+iteration_index*100+50, previous_road, road1)
    return road1, road2, road3

def generate_roundabout(component: Roundabout, odr: xodr.OpenDrive, sequence_index: int) -> xodr.OpenDrive:
    num_exits = component.num_exits
    previous_road = None
    first_road = None

    for i in range(1, num_exits+ 1):
        road1, _, previous_road = create_component(odr, component, sequence_index=sequence_index, iteration_index=i, previous_road=previous_road)
        if i == 1:
            first_road = road1  # Store the first road to close the loop later

    # Close the roundabout by connecting the last road to the first road
    if previous_road and first_road:
        simple_connection(odr, 1000+(num_exits+1)*100+50, previous_road, first_road)
    else:
        print("Error: Could not close the roundabout loop due to missing roads.")
    
    continuation_road = odr.roads[str(2+sequence_index*100+randint(1, num_exits+ 1)*10)]
    return odr, continuation_road

if __name__ == "__main__":
    odr = xodr.OpenDrive("roundabout.xodr")
    component = Roundabout(id="roundabout_1", sequence_index=0, num_exits=4, radius=10, lane_width=3, arm_length=20, right_lanes=1, left_lanes=1)
    a, continuation = generate_roundabout(component, odr, sequence_index=0)
    print(continuation)