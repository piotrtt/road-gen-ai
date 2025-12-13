from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field
import math

class RoadType(BaseModel):
    """Base class for all road definitions."""
    id: str = Field(..., description="Unique identifier for the road segment")
    sequence_index: int = Field(0, ge=0, description="The index of the component in the road network sequence")
    type: str

class StraightRoad(RoadType):
    """A straight road segment."""
    type: Literal["straight"] = "straight"
    length: float = Field(20.0, gt=0, description="Length of the road in meters")

class Curve(RoadType):
    """A curved road segment."""
    type: Literal["curve"] = "curve"
    radius: float = Field(20.0, gt=0, description="Radius of the curve in meters")
    angle: float = Field(math.pi*3/4, description="Angle of the curve in radians")

class LaneSwitch(RoadType):
    """A segment allowing lane switching."""
    type: Literal["lane_switch"] = "lane_switch"
    left_lanes_out: int = Field(2, ge=2, description="Number of lanes out")
    right_lanes_out: int = Field(2, ge=2, description="Number of lanes out")
    # TODO: Add specific fields for lane switching properties

class Fork(RoadType):
    """A road fork."""
    type: Literal["fork"] = "fork"
    angle: float = Field(math.pi*3/4, ge=math.pi/10, le=math.pi*9/10, description="Angle of the fork in radians")

class TIntersection(RoadType):
    """A T-intersection."""
    type: Literal["t_intersection"] = "t_intersection"
    angle: float = Field(math.pi*3/4, ge=math.pi/10, le=math.pi*9/10, description="Angle at which the roads meet in radians")

class Intersection(RoadType):
    """A general intersection (4-way, etc.)."""
    type: Literal["intersection"] = "intersection"
    spacing: float = Field(10.0, gt=0, description="Spacing between the roads in meters")

class Roundabout(RoadType):
    """A roundabout."""
    type: Literal["roundabout"] = "roundabout"
    radius: float = Field(20.0, gt=0, description="Radius of the roundabout")
    num_entries: int = Field(4, ge=3, le=6, description="Number of entries/exits")

class UShapedRoad(RoadType):
    """A U-shaped turn."""
    type: Literal["u_shape"] = "u_shape"
    length: float = Field(10.0, gt=0, description="Length of the road in meters")
    distance: float = Field(10.0, ge=0, description="Distance between the roads in meters")
    direction: Literal["left", "right"] = Field("right", description="Direction of the U-shaped turn")

RoadComponent = Union[
    StraightRoad, 
    Curve, 
    LaneSwitch, 
    Fork, 
    TIntersection, 
    Intersection, 
    Roundabout, 
    UShapedRoad
] # Discriminated Union based on 'type' field
