from src.road_components.definitions import UShapedRoad


def generate_ushaped_road(component: UShapedRoad) -> str:
    """
    Generates code for a U-shaped road.
    """
    ushaped_road = """
    road1 = xodr.create_road(xodr.Line(length), id=1+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    road2 = xodr.create_road(xodr.Line(length), id=2+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)

    # set zero elevation and slope
    xodr.ElevationCalculator(road1).set_zero_elevation()
    xodr.ElevationCalculator(road2).set_zero_elevation()

    # Initialize a variable for the angle between the roads in radians. Remember that the scenariogeneration library counts the angle from the previous road, so when the user gives an angle value of, say, 30 degrees to the right, you must subtract this value from 180 getting 150 degrees. When the user gives a value of, say, 45 degrees to the left, you must add 180 degrees to this value, getting 225 degrees, and so on.
    angle_road2 = 0

    # calculate the distance needed to make the turn passable
    distance = distance+left_lanes*lane_width

    if direction == 'left':
        dir = 1
    else:
        dir = -1

    # initialize a variable for the CommonJunctionCreator. Remember a curve is a junction of two roads.
    junction_creator = xodr.CommonJunctionCreator(id = 100+sequence_index*100, name='curve{sequence_index}', startnum=100+sequence_index*100)

    # add roads to the junction
    junction_creator.add_incoming_road_cartesian_geometry(road1,
                x = 0,
                y = 0,
                heading=0,
                road_connection='successor')

    junction_creator.add_incoming_road_cartesian_geometry(road2,
                x = 2*lane_width*2*(abs(math.sin(angle_road2))), #the '2*3*2' part is a calculation of the spacing needed to make the turn passable. The first '2' is the number of lanes and the '3' is the lane width, these values should be adjusted accordingly. The second '2' is the factor by which we always increase the result so that the lanes do not overlap.
                y = (distance+2*lane_width*2)*dir, #the '2*3*2' part is a calculation of the spacing needed to make the turn passable. The first '2' is the number of lanes and the '3' is the lane width, these values should be adjusted accordingly. The second '2' is the factor by which we always increase the result so that the lanes do not overlap.
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
    """
    return ushaped_road