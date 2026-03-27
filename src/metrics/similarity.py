"""
Custom similarity metrics for road network comparison.

Implements Professor Bade's recommended approach:
1. Topological similarity using Edit Distance on component type sequences
2. Geometric similarity comparing parameter distributions
3. Combined metric weighting both aspects

These metrics are used for:
- Selecting diverse road networks during generation (hybrid approach)
- Evaluating diversity of generated networks
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict
import math


# =============================================================================
# TOPOLOGICAL SIMILARITY (Edit Distance)
# =============================================================================

def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate Levenshtein (edit) distance between two sequences.

    The edit distance is the minimum number of operations (insertions,
    deletions, substitutions) needed to transform seq1 into seq2.

    Args:
        seq1: First sequence of component types
        seq2: Second sequence of component types

    Returns:
        Integer edit distance

    Example:
        >>> levenshtein_distance(['straight', 'curve'], ['straight', 'fork'])
        1  # One substitution: curve -> fork
    """
    m, n = len(seq1), len(seq2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )

    return dp[m][n]


def extract_type_sequence(road_network: Dict) -> List[str]:
    """
    Extract the sequence of component types from a road network.

    Args:
        road_network: Dictionary with 'road_network' key containing component list

    Returns:
        List of component type strings in sequence order
    """
    components = road_network.get('road_network', [])
    # Sort by sequence_index to ensure correct order
    sorted_components = sorted(components, key=lambda c: c.get('sequence_index', 0))
    return [c['type'] for c in sorted_components]


def topological_similarity(road_network_1: Dict, road_network_2: Dict) -> float:
    """
    Calculate topological similarity between two road networks using Edit Distance.

    Based on Professor Bade's recommendation: treat component types like letters
    in a string and use Edit Distance to measure how different the sequences are.

    Args:
        road_network_1: First road network dictionary
        road_network_2: Second road network dictionary

    Returns:
        Similarity score in [0, 1] where 1 = identical, 0 = maximally different

    Example:
        Network 1: [straight, curve, straight]
        Network 2: [straight, straight, curve]
        Edit distance = 2 (substitute positions 1 and 2)
        Max length = 3
        Normalized distance = 2/3 = 0.67
        Similarity = 1 - 0.67 = 0.33
    """
    seq1 = extract_type_sequence(road_network_1)
    seq2 = extract_type_sequence(road_network_2)

    if not seq1 and not seq2:
        return 1.0  # Both empty = identical

    edit_dist = levenshtein_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))

    # Normalize to [0, 1] range
    normalized_distance = edit_dist / max_len if max_len > 0 else 0

    # Convert distance to similarity (1 = identical, 0 = different)
    return 1.0 - normalized_distance


# =============================================================================
# GEOMETRIC SIMILARITY (Parameter Comparison)
# =============================================================================

# Parameter ranges for normalization (based on component_library.py)
PARAMETER_RANGES = {
    # Common parameters
    'lane_width': (2.0, 4.0),
    'right_lanes': (1, 3),
    'left_lanes': (1, 3),
    # Straight
    'length': (10, 200),
    # Curve
    'radius': (8, 40),
    'angle': (0, math.pi),  # 0 to 180 degrees in radians
    # Lane switch
    'left_lanes_out': (1, 3),
    'right_lanes_out': (1, 3),
    # Intersection
    'spacing': (5, 20),
    # Roundabout
    'num_exits': (3, 6),
    'arm_length': (5, 20),
    # U-shape
    'distance': (10, 25),
    # Direction is categorical, handled separately
}


def normalize_parameter(param_name: str, value: Any) -> float:
    """
    Normalize a parameter value to [0, 1] range.

    Args:
        param_name: Name of the parameter
        value: Raw parameter value

    Returns:
        Normalized value in [0, 1]
    """
    # Handle categorical parameters
    if param_name == 'direction':
        return 0.0 if value == 'left' else 1.0

    # Handle numeric parameters
    if param_name not in PARAMETER_RANGES:
        return 0.5  # Unknown parameter, use middle value

    min_val, max_val = PARAMETER_RANGES[param_name]
    range_size = max_val - min_val

    if range_size == 0:
        return 0.5

    normalized = (value - min_val) / range_size
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def get_type_specific_params(component_type: str) -> List[str]:
    """Get list of type-specific parameters for a component type."""
    type_params = {
        'straight': ['length'],
        'curve': ['radius', 'angle'],
        'lane_switch': ['left_lanes_out', 'right_lanes_out'],
        'fork': ['angle'],
        't_intersection': ['angle'],
        'intersection': ['spacing'],
        'roundabout': ['radius', 'num_exits', 'arm_length'],
        'u_shape': ['length', 'distance', 'direction'],
    }
    return type_params.get(component_type, [])


def extract_parameters_by_type(road_network: Dict) -> Dict[str, List[Dict]]:
    """
    Group components by type and extract their parameters.

    Args:
        road_network: Road network dictionary

    Returns:
        Dictionary mapping component type to list of parameter dicts
    """
    components = road_network.get('road_network', [])
    by_type = defaultdict(list)

    for comp in components:
        comp_type = comp['type']
        params = {
            'lane_width': comp.get('lane_width', 3.5),
            'right_lanes': comp.get('right_lanes', 2),
            'left_lanes': comp.get('left_lanes', 2),
        }
        # Add type-specific parameters
        for param in get_type_specific_params(comp_type):
            if param in comp:
                params[param] = comp[param]

        by_type[comp_type].append(params)

    return dict(by_type)


def calculate_type_parameter_distance(
    params_1: List[Dict],
    params_2: List[Dict],
    comp_type: str
) -> float:
    """
    Calculate parameter distance for a specific component type.

    Compares mean normalized parameter values between two sets of components.

    Args:
        params_1: List of parameter dicts from network 1
        params_2: List of parameter dicts from network 2
        comp_type: Component type name

    Returns:
        Distance in [0, 1] where 0 = identical parameters, 1 = maximally different
    """
    # Get all relevant parameters for this type
    all_params = ['lane_width', 'right_lanes', 'left_lanes'] + get_type_specific_params(comp_type)

    distances = []

    for param in all_params:
        # Calculate mean normalized value for each network
        values_1 = [p.get(param) for p in params_1 if param in p]
        values_2 = [p.get(param) for p in params_2 if param in p]

        if not values_1 or not values_2:
            continue

        mean_1 = sum(normalize_parameter(param, v) for v in values_1) / len(values_1)
        mean_2 = sum(normalize_parameter(param, v) for v in values_2) / len(values_2)

        # Distance between means
        distances.append(abs(mean_1 - mean_2))

    if not distances:
        return 0.5  # No comparable parameters

    # Return average distance across all parameters
    return sum(distances) / len(distances)


def geometric_similarity(road_network_1: Dict, road_network_2: Dict) -> float:
    """
    Calculate geometric similarity based on parameter distributions.

    For each component type present in both networks, compare mean parameter
    values. Types present in only one network contribute maximum distance.

    Args:
        road_network_1: First road network dictionary
        road_network_2: Second road network dictionary

    Returns:
        Similarity score in [0, 1] where 1 = identical parameters, 0 = maximally different
    """
    params_1 = extract_parameters_by_type(road_network_1)
    params_2 = extract_parameters_by_type(road_network_2)

    all_types = set(params_1.keys()) | set(params_2.keys())

    if not all_types:
        return 1.0  # Both empty

    type_distances = []

    for comp_type in all_types:
        in_1 = comp_type in params_1
        in_2 = comp_type in params_2

        if in_1 and in_2:
            # Type present in both - compare parameters
            dist = calculate_type_parameter_distance(
                params_1[comp_type],
                params_2[comp_type],
                comp_type
            )
            type_distances.append(dist)
        else:
            # Type in only one network - maximum distance
            type_distances.append(1.0)

    # Average distance across all types
    avg_distance = sum(type_distances) / len(type_distances)

    # Convert to similarity
    return 1.0 - avg_distance


# =============================================================================
# COMBINED SIMILARITY METRIC
# =============================================================================

def combined_similarity(
    road_network_1: Dict,
    road_network_2: Dict,
    topo_weight: float = 0.5,
    geom_weight: float = 0.5
) -> float:
    """
    Calculate combined similarity using both topological and geometric metrics.

    This is the main similarity function to use for:
    - Selecting diverse candidates during generation
    - Evaluating diversity of generated networks

    Args:
        road_network_1: First road network dictionary
        road_network_2: Second road network dictionary
        topo_weight: Weight for topological similarity (default 0.5)
        geom_weight: Weight for geometric similarity (default 0.5)

    Returns:
        Combined similarity score in [0, 1] where 1 = identical, 0 = maximally different
    """
    # Normalize weights
    total_weight = topo_weight + geom_weight
    topo_weight = topo_weight / total_weight
    geom_weight = geom_weight / total_weight

    topo_sim = topological_similarity(road_network_1, road_network_2)
    geom_sim = geometric_similarity(road_network_1, road_network_2)

    return topo_weight * topo_sim + geom_weight * geom_sim


def combined_distance(
    road_network_1: Dict,
    road_network_2: Dict,
    topo_weight: float = 0.5,
    geom_weight: float = 0.5
) -> float:
    """
    Calculate combined distance (inverse of similarity).

    Convenience function for algorithms that work with distances.

    Returns:
        Distance score in [0, 1] where 0 = identical, 1 = maximally different
    """
    return 1.0 - combined_similarity(road_network_1, road_network_2, topo_weight, geom_weight)


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def calculate_pairwise_similarities(
    road_networks: List[Dict],
    topo_weight: float = 0.5,
    geom_weight: float = 0.5
) -> Tuple[List[float], Dict]:
    """
    Calculate all pairwise similarities for a list of road networks.

    Args:
        road_networks: List of road network dictionaries
        topo_weight: Weight for topological similarity
        geom_weight: Weight for geometric similarity

    Returns:
        Tuple of (list of similarity values, statistics dict)
    """
    n = len(road_networks)
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = combined_similarity(
                road_networks[i],
                road_networks[j],
                topo_weight,
                geom_weight
            )
            similarities.append(sim)

    if similarities:
        stats = {
            'count': len(similarities),
            'mean': sum(similarities) / len(similarities),
            'min': min(similarities),
            'max': max(similarities),
            'std': (sum((s - sum(similarities)/len(similarities))**2 for s in similarities) / len(similarities)) ** 0.5
        }
    else:
        stats = {'count': 0, 'mean': 0, 'min': 0, 'max': 0, 'std': 0}

    return similarities, stats


if __name__ == "__main__":
    # Test the similarity functions
    print("Testing similarity metrics...\n")

    # Create test networks
    network1 = {
        "road_network": [
            {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 100},
            {"type": "curve", "sequence_index": 1, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "radius": 20, "angle": 1.57},
            {"type": "straight", "sequence_index": 2, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 50},
        ]
    }

    network2 = {
        "road_network": [
            {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 100},
            {"type": "straight", "sequence_index": 1, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 100},
            {"type": "curve", "sequence_index": 2, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "radius": 20, "angle": 1.57},
        ]
    }

    network3 = {
        "road_network": [
            {"type": "roundabout", "sequence_index": 0, "lane_width": 4.0, "right_lanes": 3, "left_lanes": 1, "radius": 25, "num_exits": 4, "arm_length": 15},
            {"type": "fork", "sequence_index": 1, "lane_width": 2.5, "right_lanes": 1, "left_lanes": 1, "angle": 0.78},
            {"type": "t_intersection", "sequence_index": 2, "lane_width": 3.0, "right_lanes": 2, "left_lanes": 2, "angle": 1.57},
        ]
    }

    print("Network 1 types:", extract_type_sequence(network1))
    print("Network 2 types:", extract_type_sequence(network2))
    print("Network 3 types:", extract_type_sequence(network3))
    print()

    print("Topological Similarities:")
    print(f"  N1 vs N2: {topological_similarity(network1, network2):.3f}")
    print(f"  N1 vs N3: {topological_similarity(network1, network3):.3f}")
    print(f"  N2 vs N3: {topological_similarity(network2, network3):.3f}")
    print()

    print("Geometric Similarities:")
    print(f"  N1 vs N2: {geometric_similarity(network1, network2):.3f}")
    print(f"  N1 vs N3: {geometric_similarity(network1, network3):.3f}")
    print(f"  N2 vs N3: {geometric_similarity(network2, network3):.3f}")
    print()

    print("Combined Similarities (50/50 weight):")
    print(f"  N1 vs N2: {combined_similarity(network1, network2):.3f}")
    print(f"  N1 vs N3: {combined_similarity(network1, network3):.3f}")
    print(f"  N2 vs N3: {combined_similarity(network2, network3):.3f}")
