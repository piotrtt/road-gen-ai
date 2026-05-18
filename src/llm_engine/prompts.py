"""
Prompt loader and formatter for LLM-based road network generation.

Loads explicit prompt templates from files as recommended by Professor Bade.
"""

import json
from pathlib import Path
from string import Template
from typing import List, Dict, Optional


# Prompts directory
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


MODEL_FAMILY_MAP = {
    "gpt5": "gpt5",
    "gemini": "gemini",
    "claude": "claude",
}


def _detect_model_family(model_name: Optional[str]) -> Optional[str]:
    """
    Detect the model family from a model name string.

    Maps model identifiers to one of: 'gpt5', 'gemini', 'claude'.
    Returns None if the model doesn't match any known family.
    """
    if not model_name:
        return None
    name = model_name.lower()
    if "gpt" in name:
        return "gpt5"
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "claude"
    return None


def load_system_prompt(model_name: Optional[str] = None) -> str:
    """
    Load the system prompt template, selecting a model-specific variant if available.

    Checks for prompts/system_prompt_{family}.txt based on the model name.
    Falls back to prompts/system_prompt.txt if no model-specific prompt exists.

    Args:
        model_name: Optional model identifier (e.g., 'gpt-5.1', 'gemini-2.5-pro', 'claude-opus-4.6')

    Returns:
        System prompt string
    """
    family = _detect_model_family(model_name)
    if family:
        family_path = PROMPTS_DIR / f"system_prompt_{family}.txt"
        if family_path.exists():
            return family_path.read_text()

    prompt_path = PROMPTS_DIR / "system_prompt.txt"
    return prompt_path.read_text()


def load_user_prompt(
    num_components: int,
    existing_networks: Optional[List[Dict]] = None,
    max_examples: int = 5
) -> str:
    """
    Load and format the user prompt with dynamic content.

    Args:
        num_components: Number of components to generate
        existing_networks: Optional list of existing networks for diversity
        max_examples: Maximum number of existing networks to include

    Returns:
        Formatted user prompt string
    """
    # Load templates
    user_template = Template((PROMPTS_DIR / "user_prompt_template.txt").read_text())
    diversity_template = Template((PROMPTS_DIR / "diversity_section.txt").read_text())

    # Build diversity section if we have existing networks
    if existing_networks and len(existing_networks) > 0:
        # Limit to most recent examples
        recent = existing_networks[-max_examples:]

        # Format networks for display (simplified view)
        formatted_networks = []
        for i, network in enumerate(recent):
            components = network.get("road_network", [])
            types = [c["type"] for c in components]
            formatted_networks.append(f"  Network {i+1}: {types}")

        networks_json = "\n".join(formatted_networks)
        existing_section = diversity_template.substitute(
            existing_networks_json=networks_json
        )
    else:
        existing_section = "This is the first network being generated. Be creative!"

    # Format user prompt
    return user_template.substitute(
        num_components=num_components,
        existing_networks_section=existing_section
    )


def format_example_output(num_components: int = 3) -> str:
    """
    Generate an example output format for few-shot prompting.

    Args:
        num_components: Number of example components

    Returns:
        Example JSON string
    """
    example = [
        {
            "id": "straight_1",
            "sequence_index": 0,
            "type": "straight",
            "lane_width": 3.5,
            "right_lanes": 2,
            "left_lanes": 2,
            "length": 100.0
        },
        {
            "id": "curve_2",
            "sequence_index": 1,
            "type": "curve",
            "lane_width": 3.5,
            "right_lanes": 2,
            "left_lanes": 2,
            "radius": 20.0,
            "angle": 1.57
        },
        {
            "id": "intersection_3",
            "sequence_index": 2,
            "type": "intersection",
            "lane_width": 3.0,
            "right_lanes": 2,
            "left_lanes": 2,
            "spacing": 10.0
        }
    ]

    return json.dumps(example[:num_components], indent=2)


# JSON-Schema fragment for a single road component. Reused by:
#   - get_function_schema()         (single network: array<component>)
#   - get_propose_candidates_schema() (multi-network: array<array<component>>)
COMPONENT_ITEM_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique identifier"},
        "sequence_index": {"type": "integer", "description": "Position in sequence"},
        "type": {
            "type": "string",
            "enum": ["straight", "curve", "lane_switch", "fork",
                    "t_intersection", "intersection", "roundabout", "u_shape"],
        },
        "lane_width": {"type": "number", "minimum": 2.0, "maximum": 4.0},
        "right_lanes": {"type": "integer", "minimum": 1, "maximum": 3},
        "left_lanes": {"type": "integer", "minimum": 1, "maximum": 3},
        # Type-specific parameters
        "length": {"type": "number", "description": "For straight, u_shape"},
        "radius": {"type": "number", "description": "For curve, roundabout"},
        "angle": {"type": "number", "description": "For curve, fork, t_intersection"},
        "left_lanes_out": {"type": "integer", "description": "For lane_switch"},
        "right_lanes_out": {"type": "integer", "description": "For lane_switch"},
        "spacing": {"type": "number", "description": "For intersection"},
        "num_exits": {"type": "integer", "description": "For roundabout"},
        "arm_length": {"type": "number", "description": "For roundabout"},
        "distance": {"type": "number", "description": "For u_shape"},
        "direction": {"type": "string", "enum": ["left", "right"], "description": "For u_shape"},
    },
    "required": ["id", "sequence_index", "type", "lane_width", "right_lanes", "left_lanes"],
}


def get_function_schema() -> Dict:
    """
    JSON schema for the single-network ``generate_road_network`` tool used by
    ``LLMGenerator``.
    """
    return {
        "name": "generate_road_network",
        "description": "Generate a road network with specified components",
        "parameters": {
            "type": "object",
            "properties": {
                "road_network": {
                    "type": "array",
                    "description": "Array of road components in sequence",
                    "items": COMPONENT_ITEM_SCHEMA,
                },
            },
            "required": ["road_network"],
        },
    }


def get_propose_candidates_schema(num_candidates: int = 3) -> Dict:
    """
    JSON schema for the ``propose_candidates`` tool used by ``HybridGenerator``.

    The model must emit ``num_candidates`` distinct road networks in a single
    tool call. Each network is itself an array of components matching
    ``COMPONENT_ITEM_SCHEMA``.
    """
    return {
        "name": "propose_candidates",
        "description": (
            f"Propose exactly {num_candidates} DISTINCT candidate road networks "
            "for diversity scoring. Return all candidates in a single call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": num_candidates,
                    "maxItems": num_candidates,
                    "description": (
                        f"Exactly {num_candidates} candidate road networks. "
                        "Each candidate is an array of road components."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "road_network": {
                                "type": "array",
                                "description": "Array of road components in sequence",
                                "items": COMPONENT_ITEM_SCHEMA,
                            },
                        },
                        "required": ["road_network"],
                    },
                },
            },
            "required": ["candidates"],
        },
    }


def get_select_best_schema(num_candidates: int = 3) -> Dict:
    """
    JSON schema for the ``select_best`` tool used by ``HybridGenerator``.

    The model picks the index of the candidate it wants to commit. Diversity
    scores are computed deterministically by the server and shown to the model
    before this tool is invoked.
    """
    return {
        "name": "select_best",
        "description": (
            "Commit the chosen candidate by index. Pick the candidate with the "
            "best (lowest) diversity similarity score, or — if scores are tied "
            "— the one you think best advances diversity from prior networks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": num_candidates - 1,
                    "description": "Index of the chosen candidate (0-based).",
                },
                "justification": {
                    "type": "string",
                    "description": "One short sentence explaining the choice.",
                },
            },
            "required": ["index"],
        },
    }


if __name__ == "__main__":
    # Test prompt loading
    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(load_system_prompt()[:500] + "...")

    print("\n" + "=" * 60)
    print("USER PROMPT (no existing networks)")
    print("=" * 60)
    print(load_user_prompt(num_components=7))

    print("\n" + "=" * 60)
    print("USER PROMPT (with existing networks)")
    print("=" * 60)
    existing = [
        {"road_network": [{"type": "straight"}, {"type": "curve"}, {"type": "fork"}]},
        {"road_network": [{"type": "roundabout"}, {"type": "straight"}, {"type": "intersection"}]},
    ]
    print(load_user_prompt(num_components=7, existing_networks=existing))

    print("\n" + "=" * 60)
    print("FUNCTION SCHEMA")
    print("=" * 60)
    print(json.dumps(get_function_schema(), indent=2)[:500] + "...")
