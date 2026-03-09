from src.road_components.definitions import RoadComponent, StraightRoad, Curve, Intersection
from src.road_components.code_blocks.straight import generate_straight_road
from src.road_components.code_blocks.curve import generate_curve
from src.road_components.code_blocks.intersection import generate_intersection
from pydantic import BaseModel
from typing import Callable

class Tool(BaseModel):
    name: str
    definition: dict
    fn: Callable

class RoadGeneratorAgent:
    """
    Aggregates specific component generators.
    Acts as a router (RAG-style) to fetch the correct code generation logic.
    """
    def __init__(self, tools: list[Tool]):
        # this is a mapping to easily access the definitions and callables
        self.tools = {tool.name: tool for tool in tools}
    
    def generate_code(self, component: RoadComponent) -> str:
        """
        Routes the component to the appropriate generator function.
        """
        tool = self.tools.get(component.type)
        if tool:
            return tool.fn(component)
        else:
            return f"# [WARNING] No generator found for component type: {type(component).__name__}"
