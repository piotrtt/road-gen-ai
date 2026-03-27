"""
Comprehensive evaluation metrics for road network generation.

Implements Professor Bade's recommended metrics:
1. Time to Quantity - Time to generate N valid maps
2. Reject Rate - Why maps were rejected
3. Diversity - Topological and geometric diversity analysis
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from src.metrics.similarity import (
    combined_similarity,
    topological_similarity,
    geometric_similarity,
    calculate_pairwise_similarities,
    extract_type_sequence,
)


@dataclass
class GenerationAttempt:
    """Record of a single generation attempt."""
    timestamp: float
    success: bool
    reason: Optional[str] = None
    component_types: Optional[List[str]] = None
    generation_time_ms: float = 0.0


@dataclass
class EvaluationMetrics:
    """
    Track and compute evaluation metrics for a generation run.

    Metrics tracked:
    - Time to Quantity: Total time to generate target number of valid maps
    - Reject Rate: Percentage and reasons for rejected generations
    - Diversity: Pairwise similarity statistics
    """

    approach: str
    target_quantity: int
    num_components: int

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Counters
    total_attempts: int = 0
    successful: int = 0

    # Rejection tracking
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    # Attempts log
    attempts: List[GenerationAttempt] = field(default_factory=list)

    # Generated networks for diversity analysis
    generated_networks: List[Dict] = field(default_factory=list)

    # Computed metrics (filled by finalize())
    time_to_quantity: Optional[float] = None
    reject_rate: Optional[float] = None
    diversity_stats: Optional[Dict] = None

    def start(self):
        """Start the evaluation timer."""
        self.start_time = time.time()

    def record_attempt(
        self,
        success: bool,
        network: Optional[Dict] = None,
        reason: Optional[str] = None,
        generation_time_ms: float = 0.0
    ):
        """
        Record a generation attempt.

        Args:
            success: Whether the attempt produced a valid network
            network: The generated network (if successful)
            reason: Reason for failure (if unsuccessful)
            generation_time_ms: Time taken for this attempt
        """
        self.total_attempts += 1

        component_types = None
        if success and network:
            self.successful += 1
            self.generated_networks.append(network)
            components = network.get("road_network", [])
            component_types = [c["type"] for c in components]
        elif reason:
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

        attempt = GenerationAttempt(
            timestamp=time.time(),
            success=success,
            reason=reason,
            component_types=component_types,
            generation_time_ms=generation_time_ms
        )
        self.attempts.append(attempt)

    def finalize(self):
        """Compute final metrics after generation is complete."""
        self.end_time = time.time()

        # Time to Quantity
        if self.start_time and self.end_time:
            self.time_to_quantity = self.end_time - self.start_time

        # Reject Rate
        if self.total_attempts > 0:
            self.reject_rate = 1.0 - (self.successful / self.total_attempts)
        else:
            self.reject_rate = 0.0

        # Diversity Statistics
        self._compute_diversity_stats()

    def _compute_diversity_stats(self):
        """Compute diversity statistics from generated networks."""
        if len(self.generated_networks) < 2:
            self.diversity_stats = {
                "count": len(self.generated_networks),
                "combined": {"mean": None, "std": None, "min": None, "max": None},
                "topological": {"mean": None, "std": None, "min": None, "max": None},
                "geometric": {"mean": None, "std": None, "min": None, "max": None},
            }
            return

        # Combined similarity
        combined_sims, combined_stats = calculate_pairwise_similarities(
            self.generated_networks
        )

        # Separate topological and geometric
        topo_sims = []
        geom_sims = []
        n = len(self.generated_networks)
        for i in range(n):
            for j in range(i + 1, n):
                topo_sims.append(topological_similarity(
                    self.generated_networks[i],
                    self.generated_networks[j]
                ))
                geom_sims.append(geometric_similarity(
                    self.generated_networks[i],
                    self.generated_networks[j]
                ))

        def calc_stats(sims):
            if not sims:
                return {"mean": None, "std": None, "min": None, "max": None}
            mean = sum(sims) / len(sims)
            std = (sum((s - mean) ** 2 for s in sims) / len(sims)) ** 0.5
            return {"mean": mean, "std": std, "min": min(sims), "max": max(sims)}

        self.diversity_stats = {
            "count": len(self.generated_networks),
            "num_pairs": len(combined_sims),
            "combined": calc_stats(combined_sims),
            "topological": calc_stats(topo_sims),
            "geometric": calc_stats(geom_sims),
        }

    def get_component_distribution(self) -> Dict[str, int]:
        """Get distribution of component types across all generated networks."""
        distribution = {}
        for network in self.generated_networks:
            for component in network.get("road_network", []):
                comp_type = component["type"]
                distribution[comp_type] = distribution.get(comp_type, 0) + 1
        return distribution

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if self.time_to_quantity is None:
            self.finalize()

        report = {
            "approach": self.approach,
            "target_quantity": self.target_quantity,
            "num_components": self.num_components,
            "summary": {
                "total_attempts": self.total_attempts,
                "successful": self.successful,
                "time_to_quantity_seconds": round(self.time_to_quantity, 3) if self.time_to_quantity else None,
                "reject_rate": round(self.reject_rate, 4) if self.reject_rate is not None else None,
            },
            "rejection_reasons": self.rejection_reasons,
            "diversity": self.diversity_stats,
            "component_distribution": self.get_component_distribution(),
            "timing": {
                "start": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            }
        }

        return report

    def print_report(self):
        """Print a human-readable report."""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print(f"EVALUATION REPORT: {self.approach.upper()}")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Target quantity: {self.target_quantity}")
        print(f"  Components per network: {self.num_components}")

        print(f"\nGeneration Summary:")
        print(f"  Total attempts: {report['summary']['total_attempts']}")
        print(f"  Successful: {report['summary']['successful']}")
        print(f"  Time to quantity: {report['summary']['time_to_quantity_seconds']:.2f}s")
        print(f"  Reject rate: {report['summary']['reject_rate']*100:.1f}%")

        if report['rejection_reasons']:
            print(f"\nRejection Reasons:")
            for reason, count in sorted(report['rejection_reasons'].items(),
                                        key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")

        if report['diversity'] and report['diversity']['combined']['mean'] is not None:
            print(f"\nDiversity Metrics (lower similarity = more diverse):")
            div = report['diversity']
            print(f"  Combined similarity:    mean={div['combined']['mean']:.3f}, "
                  f"std={div['combined']['std']:.3f}")
            print(f"  Topological similarity: mean={div['topological']['mean']:.3f}, "
                  f"std={div['topological']['std']:.3f}")
            print(f"  Geometric similarity:   mean={div['geometric']['mean']:.3f}, "
                  f"std={div['geometric']['std']:.3f}")

        print(f"\nComponent Distribution:")
        for comp_type, count in sorted(report['component_distribution'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {comp_type:20s}: {count:3d}")

    def save_report(self, output_path: Path):
        """Save report to JSON file."""
        report = self.generate_report()

        # Convert non-serializable items
        report_json = json.loads(json.dumps(report, default=str))

        with open(output_path, 'w') as f:
            json.dump(report_json, f, indent=2)

        return output_path


class EvaluationRunner:
    """
    Run comprehensive evaluation across multiple approaches.

    Generates networks, tracks metrics, and produces comparison reports.
    """

    def __init__(
        self,
        output_dir: Path = None,
        target_quantity: int = 50,
        num_components: int = 7
    ):
        """
        Initialize evaluation runner.

        Args:
            output_dir: Directory for evaluation outputs
            target_quantity: Number of networks to generate per approach
            num_components: Components per network
        """
        default_dir = Path(__file__).parent.parent.parent / "outputs" / "evaluation"
        self.output_dir = output_dir or default_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_quantity = target_quantity
        self.num_components = num_components
        self.results: Dict[str, EvaluationMetrics] = {}

    def evaluate_generator(self, generator, approach_name: str) -> EvaluationMetrics:
        """
        Evaluate a single generator.

        Args:
            generator: Generator instance with generate() method
            approach_name: Name of the approach

        Returns:
            EvaluationMetrics with results
        """
        metrics = EvaluationMetrics(
            approach=approach_name,
            target_quantity=self.target_quantity,
            num_components=self.num_components
        )

        print(f"\n📊 Evaluating {approach_name}...")
        print(f"   Target: {self.target_quantity} networks, {self.num_components} components each")

        metrics.start()

        while metrics.successful < self.target_quantity:
            attempt_start = time.time()

            try:
                network = generator.generate(self.num_components)
                attempt_time = (time.time() - attempt_start) * 1000

                # Validate network
                if self._validate_network(network):
                    metrics.record_attempt(
                        success=True,
                        network=network,
                        generation_time_ms=attempt_time
                    )
                    if metrics.successful % 10 == 0:
                        print(f"   Generated {metrics.successful}/{self.target_quantity}...")
                else:
                    metrics.record_attempt(
                        success=False,
                        reason="invalid_structure",
                        generation_time_ms=attempt_time
                    )

            except Exception as e:
                attempt_time = (time.time() - attempt_start) * 1000
                reason = str(type(e).__name__)
                metrics.record_attempt(
                    success=False,
                    reason=reason,
                    generation_time_ms=attempt_time
                )

            # Safety limit
            if metrics.total_attempts > self.target_quantity * 10:
                print(f"   ⚠️ Too many failures, stopping at {metrics.successful} networks")
                break

        metrics.finalize()
        self.results[approach_name] = metrics

        print(f"   ✅ Completed in {metrics.time_to_quantity:.2f}s")
        print(f"   Reject rate: {metrics.reject_rate*100:.1f}%")

        return metrics

    def _validate_network(self, network: Dict) -> bool:
        """Validate network structure."""
        if not isinstance(network, dict):
            return False

        road_network = network.get("road_network", [])
        if not isinstance(road_network, list):
            return False

        if len(road_network) == 0:
            return False

        return True

    def compare_results(self) -> Dict:
        """Generate comparison across all evaluated approaches."""
        if not self.results:
            return {}

        comparison = {
            "configuration": {
                "target_quantity": self.target_quantity,
                "num_components": self.num_components,
            },
            "approaches": {}
        }

        for name, metrics in self.results.items():
            report = metrics.generate_report()
            comparison["approaches"][name] = {
                "time_to_quantity": report["summary"]["time_to_quantity_seconds"],
                "reject_rate": report["summary"]["reject_rate"],
                "diversity_combined_mean": report["diversity"]["combined"]["mean"]
                    if report["diversity"]["combined"]["mean"] else None,
                "diversity_topological_mean": report["diversity"]["topological"]["mean"]
                    if report["diversity"]["topological"]["mean"] else None,
            }

        # Rankings
        approaches = list(comparison["approaches"].keys())

        # Fastest (lowest time)
        by_time = sorted(approaches,
                        key=lambda x: comparison["approaches"][x]["time_to_quantity"] or float('inf'))
        comparison["rankings"] = {
            "by_speed": by_time,
            "by_diversity": sorted(approaches,
                key=lambda x: comparison["approaches"][x]["diversity_combined_mean"] or float('inf')),
            "by_reject_rate": sorted(approaches,
                key=lambda x: comparison["approaches"][x]["reject_rate"] or float('inf')),
        }

        return comparison

    def print_comparison(self):
        """Print comparison table."""
        comparison = self.compare_results()

        if not comparison:
            print("No results to compare")
            return

        print("\n" + "=" * 80)
        print("EVALUATION COMPARISON")
        print("=" * 80)

        print(f"\nConfiguration: {self.target_quantity} networks, "
              f"{self.num_components} components each\n")

        # Table header
        print(f"{'Approach':<20} | {'Time (s)':<10} | {'Reject %':<10} | {'Diversity':<10}")
        print("-" * 60)

        for name, data in comparison["approaches"].items():
            time_str = f"{data['time_to_quantity']:.2f}" if data['time_to_quantity'] else "N/A"
            reject_str = f"{data['reject_rate']*100:.1f}%" if data['reject_rate'] is not None else "N/A"
            div_str = f"{data['diversity_combined_mean']:.3f}" if data['diversity_combined_mean'] else "N/A"
            print(f"{name:<20} | {time_str:<10} | {reject_str:<10} | {div_str:<10}")

        print("\nRankings:")
        print(f"  Fastest: {comparison['rankings']['by_speed']}")
        print(f"  Most diverse: {comparison['rankings']['by_diversity']}")
        print(f"  Lowest reject rate: {comparison['rankings']['by_reject_rate']}")

    def save_all_reports(self):
        """Save all evaluation reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Individual reports
        for name, metrics in self.results.items():
            filepath = self.output_dir / f"{name}_evaluation_{timestamp}.json"
            metrics.save_report(filepath)
            print(f"Saved: {filepath.name}")

        # Comparison report
        comparison = self.compare_results()
        comparison_path = self.output_dir / f"comparison_{timestamp}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved: {comparison_path.name}")


if __name__ == "__main__":
    # Quick test
    print("Testing EvaluationMetrics...")

    metrics = EvaluationMetrics(
        approach="test",
        target_quantity=5,
        num_components=7
    )

    metrics.start()

    # Simulate some generations
    test_network = {
        "road_network": [
            {"type": "straight", "sequence_index": 0},
            {"type": "curve", "sequence_index": 1},
        ]
    }

    for i in range(5):
        metrics.record_attempt(success=True, network=test_network, generation_time_ms=10.0)

    metrics.record_attempt(success=False, reason="test_failure")

    metrics.finalize()
    metrics.print_report()
