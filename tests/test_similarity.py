"""
Unit tests for similarity metrics.
"""

import pytest
import math
from src.metrics.similarity import (
    levenshtein_distance,
    extract_type_sequence,
    topological_similarity,
    geometric_similarity,
    combined_similarity,
    normalize_parameter,
    calculate_pairwise_similarities,
)


class TestLevenshteinDistance:
    """Tests for edit distance calculation."""

    def test_identical_sequences(self):
        """Identical sequences have distance 0."""
        seq = ['straight', 'curve', 'straight']
        assert levenshtein_distance(seq, seq) == 0

    def test_empty_sequences(self):
        """Empty sequences have distance 0."""
        assert levenshtein_distance([], []) == 0

    def test_one_empty(self):
        """Distance to empty is length of other."""
        seq = ['a', 'b', 'c']
        assert levenshtein_distance(seq, []) == 3
        assert levenshtein_distance([], seq) == 3

    def test_single_substitution(self):
        """Single substitution has distance 1."""
        seq1 = ['straight', 'curve']
        seq2 = ['straight', 'fork']
        assert levenshtein_distance(seq1, seq2) == 1

    def test_single_insertion(self):
        """Single insertion has distance 1."""
        seq1 = ['straight', 'curve']
        seq2 = ['straight', 'fork', 'curve']
        assert levenshtein_distance(seq1, seq2) == 1

    def test_single_deletion(self):
        """Single deletion has distance 1."""
        seq1 = ['straight', 'fork', 'curve']
        seq2 = ['straight', 'curve']
        assert levenshtein_distance(seq1, seq2) == 1

    def test_complete_difference(self):
        """Completely different sequences."""
        seq1 = ['a', 'b', 'c']
        seq2 = ['x', 'y', 'z']
        assert levenshtein_distance(seq1, seq2) == 3

    def test_professor_example(self):
        """Example from Professor Bade's meeting (Katze -> Kater)."""
        # Adapted to our domain
        seq1 = ['straight', 'curve', 'straight']
        seq2 = ['straight', 'straight', 'curve']
        # Need 2 operations: swap positions of curve and second straight
        assert levenshtein_distance(seq1, seq2) == 2


class TestExtractTypeSequence:
    """Tests for type sequence extraction."""

    def test_basic_extraction(self):
        """Extract types in sequence order."""
        network = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "curve", "sequence_index": 1},
                {"type": "fork", "sequence_index": 2},
            ]
        }
        assert extract_type_sequence(network) == ['straight', 'curve', 'fork']

    def test_out_of_order(self):
        """Components out of order should be sorted."""
        network = {
            "road_network": [
                {"type": "curve", "sequence_index": 1},
                {"type": "straight", "sequence_index": 0},
                {"type": "fork", "sequence_index": 2},
            ]
        }
        assert extract_type_sequence(network) == ['straight', 'curve', 'fork']

    def test_empty_network(self):
        """Empty network returns empty list."""
        assert extract_type_sequence({"road_network": []}) == []
        assert extract_type_sequence({}) == []


class TestTopologicalSimilarity:
    """Tests for topological similarity."""

    def test_identical_networks(self):
        """Identical networks have similarity 1.0."""
        network = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "curve", "sequence_index": 1},
            ]
        }
        assert topological_similarity(network, network) == 1.0

    def test_completely_different(self):
        """Completely different types have low similarity."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "curve", "sequence_index": 1},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "fork", "sequence_index": 0},
                {"type": "roundabout", "sequence_index": 1},
            ]
        }
        # Edit distance = 2, max_len = 2, normalized = 1.0
        # Similarity = 1 - 1.0 = 0.0
        assert topological_similarity(network1, network2) == 0.0

    def test_partial_similarity(self):
        """Partial overlap gives intermediate similarity."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "curve", "sequence_index": 1},
                {"type": "straight", "sequence_index": 2},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "straight", "sequence_index": 1},
                {"type": "curve", "sequence_index": 2},
            ]
        }
        # Edit distance = 2, max_len = 3
        # Similarity = 1 - 2/3 = 0.333...
        sim = topological_similarity(network1, network2)
        assert 0.3 < sim < 0.4

    def test_different_lengths(self):
        """Networks of different lengths."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0},
                {"type": "curve", "sequence_index": 1},
                {"type": "fork", "sequence_index": 2},
            ]
        }
        # Edit distance = 2 (add curve, fork), max_len = 3
        # Similarity = 1 - 2/3 = 0.333...
        sim = topological_similarity(network1, network2)
        assert 0.3 < sim < 0.4


class TestNormalizeParameter:
    """Tests for parameter normalization."""

    def test_lane_width(self):
        """Lane width normalization."""
        assert normalize_parameter('lane_width', 2.0) == 0.0
        assert normalize_parameter('lane_width', 4.0) == 1.0
        assert normalize_parameter('lane_width', 3.0) == 0.5

    def test_direction_categorical(self):
        """Direction is categorical."""
        assert normalize_parameter('direction', 'left') == 0.0
        assert normalize_parameter('direction', 'right') == 1.0

    def test_unknown_parameter(self):
        """Unknown parameters return 0.5."""
        assert normalize_parameter('unknown_param', 42) == 0.5


class TestGeometricSimilarity:
    """Tests for geometric similarity."""

    def test_identical_parameters(self):
        """Identical parameters have similarity close to 1.0."""
        network = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 100},
            ]
        }
        sim = geometric_similarity(network, network)
        assert sim > 0.99  # Allow for floating point

    def test_different_types_no_overlap(self):
        """No type overlap gives low similarity."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "length": 100},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "roundabout", "sequence_index": 0, "lane_width": 3.5, "radius": 20, "num_exits": 4, "arm_length": 10},
            ]
        }
        # No common types, so max distance for each type
        sim = geometric_similarity(network1, network2)
        assert sim == 0.0

    def test_same_types_different_params(self):
        """Same types with different parameters."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 2.5, "right_lanes": 1, "left_lanes": 1, "length": 20},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 4.0, "right_lanes": 3, "left_lanes": 3, "length": 200},
            ]
        }
        sim = geometric_similarity(network1, network2)
        # Parameters are at opposite ends, should have low similarity
        assert sim < 0.3


class TestCombinedSimilarity:
    """Tests for combined similarity."""

    def test_identical_networks(self):
        """Identical networks have similarity 1.0."""
        network = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "right_lanes": 2, "left_lanes": 2, "length": 100},
            ]
        }
        assert combined_similarity(network, network) == 1.0

    def test_weight_influence(self):
        """Weights affect the result."""
        network1 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 3.5, "length": 100},
            ]
        }
        network2 = {
            "road_network": [
                {"type": "straight", "sequence_index": 0, "lane_width": 2.5, "length": 20},
            ]
        }
        # Same topology, different geometry
        topo_heavy = combined_similarity(network1, network2, topo_weight=0.9, geom_weight=0.1)
        geom_heavy = combined_similarity(network1, network2, topo_weight=0.1, geom_weight=0.9)

        # Topological similarity = 1.0 (identical types)
        # Geometric similarity < 1.0 (different params)
        # So topo_heavy should be higher
        assert topo_heavy > geom_heavy


class TestCalculatePairwiseSimilarities:
    """Tests for batch similarity calculation."""

    def test_three_networks(self):
        """Calculate all pairs for 3 networks."""
        networks = [
            {"road_network": [{"type": "straight", "sequence_index": 0}]},
            {"road_network": [{"type": "curve", "sequence_index": 0}]},
            {"road_network": [{"type": "fork", "sequence_index": 0}]},
        ]
        similarities, stats = calculate_pairwise_similarities(networks)

        # 3 networks -> 3 pairs
        assert stats['count'] == 3
        assert len(similarities) == 3

    def test_empty_list(self):
        """Empty list returns empty stats."""
        similarities, stats = calculate_pairwise_similarities([])
        assert stats['count'] == 0

    def test_single_network(self):
        """Single network has no pairs."""
        networks = [{"road_network": [{"type": "straight", "sequence_index": 0}]}]
        similarities, stats = calculate_pairwise_similarities(networks)
        assert stats['count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
