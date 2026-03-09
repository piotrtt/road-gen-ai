from src.road_components.definitions import StraightRoad
import scenariogeneration.xodr as xodr
from typing import Tuple
from src.road_components.components.straight import generate_straight_road
from src.road_components.helpers import simple_connection
from src.road_components.definitions import StraightRoad
from random import randint

# odr = xodr.OpenDrive("test.xodr")
# comp = StraightRoad(id="test_road", length=100, left_lanes=2, right_lanes=2, lane_width=3.5, sequence_index=0)
# odr, road1 = generate_straight_road(comp, odr, sequence_index=0)
# odr, road2 = generate_straight_road(comp, odr, sequence_index=1)
# simple_connection(odr, junction_id=200, road_from=road1, road_to=road2)
# print(odr.roads)
a={'1': 'a', '2': 'b', '3': 'c', '4': 'd', '5': 'e'}
print(a[str(randint(1,5))])