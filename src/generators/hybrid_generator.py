"""
HybridGenerator — LLM + deterministic-diversity feedback loop.

Per network:
  1. LLM proposes 3 candidates in a single tool call (``propose_candidates``).
  2. Server scores each candidate's mean ``combined_similarity`` against the
     networks already committed in this batch.
  3. LLM picks the best candidate via a second tool call (``select_best``),
     given the deterministic scores.
  4. The second-best candidate's score becomes the threshold for the next
     round. Up to ``max_rounds`` rounds run until the chosen score fails to
     beat the threshold; at that point the running best is committed.

The threshold resets between networks. ``_recent_networks`` is an in-memory
log of networks committed by this instance, used as the diversity reference.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from src.generators.base_generator import BaseGenerator
from src.generators.network_storage import NetworkStorage
from src.llm_engine.client import LLMClient
from src.llm_engine.prompts import (
    load_system_prompt,
    load_user_prompt,
    get_propose_candidates_schema,
    get_select_best_schema,
)
from src.metrics.similarity import combined_similarity


VALID_TYPES = {
    "straight", "curve", "lane_switch", "fork",
    "t_intersection", "intersection", "roundabout", "u_shape",
}
REQUIRED_FIELDS = {
    "id", "sequence_index", "type", "lane_width", "right_lanes", "left_lanes",
}


class HybridGenerator(BaseGenerator):
    """
    Generate networks via a feedback loop between the LLM and the project's
    deterministic diversity metric.
    """

    NUM_CANDIDATES = 3

    def __init__(
        self,
        output_dir=None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_rounds: int = 3,
        max_examples: int = 5,
        include_existing: bool = True,
    ):
        """
        Args:
            output_dir: Storage directory for networks.
            model_name: Model identifier (e.g. ``gpt-5.1``). Defaults to env LLM_MODEL.
            temperature: Sampling temperature; ``None`` uses the provider default.
            max_rounds: Maximum rounds in the per-network feedback loop. Round 1
                always commits; subsequent rounds must beat the running threshold
                to replace the current best.
            max_examples: Number of prior networks shown in the user prompt.
            include_existing: If False, the LLM is not shown any prior networks.
        """
        super().__init__(output_dir)
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.max_examples = max_examples
        self.include_existing = include_existing

        self.client = LLMClient(model_name=self.model_name, temperature=temperature)
        self.storage = NetworkStorage(storage_dir=output_dir)
        self.system_prompt = load_system_prompt(model_name=self.model_name)
        self._recent_networks: List[Dict] = []

    def get_name(self) -> str:
        return "hybrid"

    # ------------------------------------------------------------------ helpers
    def _fix_sequence_indices(self, road_network: List[Dict]) -> List[Dict]:
        for i, comp in enumerate(road_network):
            comp["sequence_index"] = i
            comp["id"] = f"{comp.get('type', 'comp')}_{i + 1}"
        return road_network

    def _validate_candidate(self, candidate: Dict, num_components: int) -> bool:
        if not isinstance(candidate, dict):
            return False
        rn = candidate.get("road_network")
        if not isinstance(rn, list) or len(rn) == 0:
            return False
        if abs(len(rn) - num_components) > 2:
            return False
        for comp in rn:
            if not isinstance(comp, dict):
                return False
            if REQUIRED_FIELDS - set(comp.keys()):
                return False
            if comp.get("type") not in VALID_TYPES:
                return False
        return True

    def _score_candidate(self, candidate: Dict, existing: List[Dict]) -> float:
        """Mean pairwise combined similarity vs all existing networks; 0.0 if none."""
        if not existing:
            return 0.0
        sims = [combined_similarity(candidate, e) for e in existing]
        return sum(sims) / len(sims)

    # =============================================================== LLM round 1
    def _propose_candidates(
        self, num_components: int, existing: List[Dict]
    ) -> Tuple[List[Dict], List[Dict[str, str]], Optional[Dict]]:
        """
        Force the model to call ``propose_candidates`` with NUM_CANDIDATES networks.

        Returns ``(valid_candidates, messages_for_round, assistant_tool_message)``:
            * valid_candidates -- list of validated, index-fixed candidate dicts
            * messages_for_round -- the full message list including the assistant
              tool_calls message, ready to be extended with a tool result and the
              follow-up user turn for the select step.
            * assistant_tool_message -- the assistant message (with tool_calls)
              that the follow-up step needs to reference by id.
        """
        user_prompt = load_user_prompt(
            num_components=num_components,
            existing_networks=existing if self.include_existing else None,
            max_examples=self.max_examples,
        )
        user_prompt += (
            f"\n\nIMPORTANT: Propose exactly {self.NUM_CANDIDATES} DISTINCT candidate "
            "networks in a single call to the `propose_candidates` tool. Each "
            "candidate should be a fresh take on the design (different type "
            "compositions and/or parameter zones) — don't return three near-copies. "
            "We will deterministically score each candidate's similarity to prior "
            "networks and ask you to pick the most diverse one."
        )

        tool = {"type": "function", "function": get_propose_candidates_schema(self.NUM_CANDIDATES)}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self.client.tool_call(messages, [tool], "propose_candidates")
        if not result:
            return [], messages, None

        args = result["arguments"]
        raw_candidates = args.get("candidates") or []
        valid: List[Dict] = []
        for cand in raw_candidates:
            if not self._validate_candidate(cand, num_components):
                continue
            cand["road_network"] = self._fix_sequence_indices(cand["road_network"])
            valid.append(cand)

        messages.append(result["raw_message"])
        return valid, messages, result

    # =============================================================== LLM round 2
    def _select_via_llm(
        self,
        candidates: List[Dict],
        scores: List[float],
        threshold: Optional[float],
        messages_so_far: List[Dict],
        propose_call: Dict,
    ) -> int:
        """Force ``select_best``; return the picked index. Falls back to argmin on error."""
        # Tool-result message answering the propose_candidates call.
        score_lines = [f"  candidate {i}: similarity = {s:.4f}" for i, s in enumerate(scores)]
        tool_payload = {
            "candidate_scores": [{"index": i, "similarity": round(s, 4)} for i, s in enumerate(scores)],
            "threshold_to_beat": round(threshold, 4) if threshold is not None else None,
        }
        msgs = list(messages_so_far) + [
            {
                "role": "tool",
                "tool_call_id": propose_call["tool_call_id"],
                "content": json.dumps(tool_payload),
            },
            {
                "role": "user",
                "content": (
                    "Diversity scores for the candidates you just proposed "
                    "(lower = MORE diverse from prior networks):\n"
                    + "\n".join(score_lines)
                    + (
                        f"\n\nThreshold to beat (must be <= this to improve on the "
                        f"running best): {threshold:.4f}.\n"
                        if threshold is not None
                        else "\n\nThis is the first round — any candidate will be accepted.\n"
                    )
                    + "\nCall the `select_best` tool with the index of the candidate "
                    "you want to commit. Prefer the lowest similarity; break ties by "
                    "type-composition novelty."
                ),
            },
        ]
        tool = {"type": "function", "function": get_select_best_schema(len(candidates))}
        result = self.client.tool_call(msgs, [tool], "select_best")
        if not result:
            return min(range(len(scores)), key=lambda i: scores[i])
        try:
            idx = int(result["arguments"].get("index", -1))
        except (TypeError, ValueError):
            idx = -1
        if not (0 <= idx < len(candidates)):
            print(f"[hybrid] select_best returned out-of-range index={idx}; "
                  f"falling back to argmin.")
            return min(range(len(scores)), key=lambda i: scores[i])
        return idx

    # ====================================================================== loop
    def _run_round(
        self,
        num_components: int,
        existing: List[Dict],
        threshold: Optional[float],
    ) -> Optional[Tuple[Dict, float, float]]:
        """
        One round of the feedback loop.

        Returns ``(chosen_candidate, chosen_score, second_best_score)`` or
        ``None`` if propose_candidates yielded too few valid candidates.
        """
        for _attempt in range(3):  # inner retry budget on propose
            candidates, messages, propose_call = self._propose_candidates(
                num_components, existing
            )
            if len(candidates) >= 2:
                break
            print(f"[hybrid] propose_candidates returned {len(candidates)} valid "
                  "candidates; retrying.")
        else:
            return None

        scores = [self._score_candidate(c, existing) for c in candidates]
        chosen_idx = self._select_via_llm(candidates, scores, threshold, messages, propose_call)
        chosen = candidates[chosen_idx]
        chosen_score = scores[chosen_idx]
        sorted_scores = sorted(scores)
        # Second-best (second-lowest) similarity; if only one candidate survived
        # validation past the early return, treat the running best as the
        # threshold proxy.
        second_best = sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]
        return chosen, chosen_score, second_best

    # ===================================================================== generate
    def generate(self, num_components: int) -> Dict:
        existing: List[Dict] = []
        if self.include_existing:
            if self._recent_networks:
                existing = list(self._recent_networks)
            else:
                existing = self.storage.load_all(approach="hybrid")

        # Globally-best candidate seen across all rounds (lowest similarity).
        best_so_far: Optional[Dict] = None
        best_score = float("inf")
        threshold: Optional[float] = None

        for round_idx in range(1, self.max_rounds + 1):
            outcome = self._run_round(num_components, existing, threshold)
            if outcome is None:
                if best_so_far is not None:
                    break
                raise RuntimeError(
                    f"HybridGenerator: round {round_idx} failed to produce ≥2 valid candidates."
                )
            chosen, chosen_score, second_best = outcome

            # Always update the running global best whenever this round's chosen
            # candidate is more diverse than any seen so far.
            if chosen_score < best_score:
                best_so_far, best_score = chosen, chosen_score

            # Threshold check: round 1 always passes; later rounds need to meet
            # the prior round's second-best (the spec's "minimum bar to clear").
            passes_threshold = (
                round_idx == 1
                or threshold is None
                or chosen_score <= threshold
            )

            print(
                f"[hybrid] round={round_idx} chosen_score={chosen_score:.4f} "
                f"running_best={best_score:.4f} "
                f"threshold={'-' if threshold is None else f'{threshold:.4f}'} "
                f"second_best={second_best:.4f} "
                f"-> {'pass' if passes_threshold else 'stop'}"
            )

            if passes_threshold:
                threshold = second_best  # raise the bar for the next round
            else:
                break

        assert best_so_far is not None
        # Persist into the in-memory log for the next network in this batch.
        self._recent_networks.append({"road_network": best_so_far["road_network"]})
        # Attach a small selection record for debugging / reporting.
        best_so_far.setdefault("_selection_metadata", {})
        best_so_far["_selection_metadata"].update(
            {"final_score": best_score, "max_rounds": self.max_rounds}
        )
        return best_so_far

    # ======================================================================= batch
    def generate_multiple(self, num_networks: int, num_components: int) -> list:
        saved_files = []
        for i in range(num_networks):
            print(f"Generating network {i + 1}/{num_networks}...")
            try:
                net = self.generate(num_components)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
            selection_info = net.pop("_selection_metadata", {})
            filepath = self.storage.save(
                net,
                approach="hybrid",
                metadata={
                    "model": self.model_name,
                    "num_components": len(net.get("road_network", [])),
                    "final_score": selection_info.get("final_score"),
                },
            )
            saved_files.append(filepath)
            types = [c["type"] for c in net["road_network"]]
            print(f"  Types: {types}")
            print(f"  Saved to {filepath.name}")
        return saved_files
