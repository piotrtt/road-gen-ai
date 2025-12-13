from src.road_components.definitions import StraightRoad

def generate_straight_road(component: StraightRoad) -> str:
    """
    Generates code for a straight road.
    """
    straight_road = """
    road1 = xodr.create_road(xodr.Line(length), id=1+sequence_index*10, left_lanes=left_lanes, right_lanes=right_lanes, lane_width=lane_width)
    odr.add_road(road1)
    continuation_road = road1
    """
    return straight_road
