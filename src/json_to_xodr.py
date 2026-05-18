#!/usr/bin/env python3
"""
Converts a JSON road-network schema (as produced by the road network generators)
into an OpenDRIVE (.xodr) file using the scenariogeneration library.

Usage:
    python src/json_to_xodr.py outputs/graphs/random_network_001.json
    python src/json_to_xodr.py outputs/graphs/random_network_001.json --output outputs/xodr/my_road.xodr
"""

import argparse
import json
from pathlib import Path

import scenariogeneration.xodr as xodr

from src.road_components.definitions import (
    Curve,
    Fork,
    Intersection,
    LaneSwitch,
    Roundabout,
    StraightRoad,
    TIntersection,
    UShapedRoad,
)
from src.road_components.components.curve import generate_curve
from src.road_components.components.fork import generate_fork
from src.road_components.components.intersection import generate_intersection
from src.road_components.components.lane_switch import generate_lane_switch
from src.road_components.components.roundabout import generate_roundabout
from src.road_components.components.straight import generate_straight_road
from src.road_components.components.Tintersection import generate_t_intersection
from src.road_components.components.Ulane import generate_ushaped_road
from src.road_components.helpers import simple_connection


# Maps the "type" string in JSON to the Pydantic model class.
_TYPE_TO_MODEL = {
    "straight": StraightRoad,
    "curve": Curve,
    "lane_switch": LaneSwitch,
    "fork": Fork,
    "t_intersection": TIntersection,
    "intersection": Intersection,
    "roundabout": Roundabout,
    "u_shape": UShapedRoad,
}


def _parse_component(comp_dict: dict):
    """Deserialise one component dict into its typed Pydantic model."""
    comp_type = comp_dict.get("type")
    model_cls = _TYPE_TO_MODEL.get(comp_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown component type {comp_type!r}. "
            f"Supported: {list(_TYPE_TO_MODEL)}"
        )
    return model_cls(**comp_dict)


def _dispatch(component, odr: xodr.OpenDrive, sequence_index: int):
    """Route a typed component to its generator and return (odr, continuation_road)."""
    if isinstance(component, StraightRoad):
        return generate_straight_road(component, odr, sequence_index)
    if isinstance(component, Curve):
        return generate_curve(component, odr, sequence_index)
    if isinstance(component, Intersection):
        return generate_intersection(component, odr, sequence_index)
    if isinstance(component, Fork):
        return generate_fork(component, odr, sequence_index)
    if isinstance(component, LaneSwitch):
        return generate_lane_switch(component, odr, sequence_index)
    if isinstance(component, TIntersection):
        # Note: generate_t_intersection has odr as first argument.
        return generate_t_intersection(odr, component, sequence_index)
    if isinstance(component, UShapedRoad):
        # Note: generate_ushaped_road has odr as first argument.
        return generate_ushaped_road(odr, component, sequence_index)
    if isinstance(component, Roundabout):
        return generate_roundabout(component, odr, sequence_index)
    raise ValueError(f"No generator registered for {type(component).__name__}")


# Each component's sequence_index is multiplied by this factor before being passed to
# the generators so that road/junction IDs across components never collide.
# Roundabout internally uses  id = 1 + seq*100 + iter*10  (iter goes up to num_exits).
# Regular components use       id = k + seq*10             (k up to ~4).
# A factor of 100 ensures non-roundabout IDs live at seq*1000 and roundabout IDs at
# seq*10000 — both safely separated from each other and from their neighbours.
_SEQ_SCALE = 100


def _first_road_id(component, idx: int) -> str:
    """
    Return the string key in odr.roads for the first (entry) road of a component.

    Each generator receives sequence_index = idx * _SEQ_SCALE, so:
      - Regular components: id = 1 + (idx * _SEQ_SCALE) * 10
      - Roundabout:         id = 1 + (idx * _SEQ_SCALE) * 100 + 1 * 10
    """
    scaled = idx * _SEQ_SCALE
    if isinstance(component, Roundabout):
        return str(1 + scaled * 100 + 10)
    return str(1 + scaled * 10)


def convert(json_path: Path, output_path: Path) -> Path:
    """
    Convert a road-network JSON file to an OpenDRIVE .xodr file.

    Args:
        json_path:   Path to the JSON file containing a ``road_network`` list.
        output_path: Destination .xodr file path.

    Returns:
        The resolved output path after writing.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    raw_components = data.get("road_network", [])
    if not raw_components:
        raise ValueError(f"No components found under 'road_network' key in {json_path}")

    components = [_parse_component(c) for c in raw_components]

    odr = xodr.OpenDrive(output_path.stem)

    prev_continuation: xodr.Road | None = None

    for idx, component in enumerate(components):
        scaled_seq = idx * _SEQ_SCALE
        odr, continuation = _dispatch(component, odr, sequence_index=scaled_seq)

        if prev_continuation is not None:
            first_id = _first_road_id(component, idx)
            if first_id in odr.roads:
                # Use IDs well above any internal component junction:
                # regular components use 100 + scaled_seq*100 (max ~1M for 100 comps)
                # roundabouts use 1000 + iter*100 (~1750 max).
                # 10_000_000 + idx*100 is safely outside both ranges.
                inter_junction_id = 10_000_000 + idx * 100
                try:
                    simple_connection(
                        odr,
                        inter_junction_id,
                        prev_continuation,
                        odr.roads[first_id],
                    )
                except ValueError as exc:
                    # Some components (e.g. roundabouts whose closing-loop junction
                    # already claims the entry road's predecessor slot) cannot accept
                    # an additional inter-component junction.  Log and skip.
                    print(
                        f"[WARNING] Could not link component {idx-1} → {idx} "
                        f"({component.type}): {exc}"
                    )
            else:
                print(
                    f"[WARNING] Could not find entry road (id={first_id}) for "
                    f"component {idx} ({component.type}); skipping connection."
                )

        prev_continuation = continuation

    output_path.parent.mkdir(parents=True, exist_ok=True)
    odr.write_xml(str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSON road-network schema to OpenDRIVE (.xodr)"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to the input JSON road-network file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output .xodr file path "
            "(default: outputs/xodr/<input_stem>.xodr)"
        ),
    )
    args = parser.parse_args()

    json_path = args.json_file.resolve()
    if not json_path.exists():
        parser.error(f"File not found: {json_path}")

    if args.output:
        output_path = args.output.resolve()
    else:
        default_xodr_dir = json_path.parent.parent / "xodr"
        output_path = default_xodr_dir / json_path.with_suffix(".xodr").name

    print(f"Input:  {json_path}")
    print(f"Output: {output_path}")

    result = convert(json_path, output_path)
    print(f"Written: {result}")


if __name__ == "__main__":
    main()
