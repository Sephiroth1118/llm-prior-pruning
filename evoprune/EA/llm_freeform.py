"""MolLEO-style LLM operator that proposes a complete pruning policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .budget import BudgetRepairResult, CostFunction, repair_policy_budget
from .experience import EvaluatedPolicy, ExperienceArchive
from .llm_crossover import JSONSender
from .operators import OperatorError, balanced_mutation, uniform_crossover
from .policy import Policy, PolicyBounds


JSONResponse = Union[str, Mapping[str, Any]]


@dataclass(frozen=True)
class LLMFreeformProposal:
    reasoning: str
    policy: Policy
    hypothesis: str = ""
    evidence_used: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMFreeformOutcome:
    child: Policy
    raw_child: Optional[Policy]
    budget_repair: Optional[BudgetRepairResult]
    proposal: Optional[LLMFreeformProposal]
    prompt: Dict[str, Any]
    raw_response: Optional[str]
    fallback_used: bool
    fallback_reason: Optional[str]


def _parent_to_dict(parent: EvaluatedPolicy) -> Dict[str, Any]:
    return {
        "source_id": parent.source_id,
        "policy": parent.policy.to_dict(),
        "fitness": parent.fitness,
        "kl": parent.kl,
        "ppl": parent.ppl,
        "flops_reduction": parent.flops_reduction,
    }


def build_freeform_prompt(
    bounds: PolicyBounds,
    parent_a: EvaluatedPolicy,
    parent_b: EvaluatedPolicy,
    *,
    objective: Mapping[str, Any],
    experience_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Use the same information as masked crossover but request a full child."""

    prompt: Dict[str, Any] = {
        "task": (
            "Act as a MolLEO-style evolutionary operator. Propose one complete child "
            "policy that is expected to outperform both evaluated parents. You may "
            "recombine, mutate, or depart from parent gene values, but every output "
            "gene must respect the supplied bounds."
        ),
        "semantics": {
            "higher_level": "stronger compression",
            "fitness": "higher is better",
            "kl": "lower is better",
            "ppl": "lower is better",
        },
        "objective": dict(objective),
        "bounds": bounds.to_compact_dict(),
        "parents": {
            "A": _parent_to_dict(parent_a),
            "B": _parent_to_dict(parent_b),
        },
        "rules": [
            f"attn_levels and mlp_levels must each have exactly {bounds.num_layers} entries",
            "every level must be an integer within its supplied per-layer bounds",
            "the proposed policy must differ from both parents when a novel child is feasible",
            "account for the target compression budget when allocating levels",
            "prefer a testable structural hypothesis over arbitrary values",
            "return one JSON object and no markdown",
        ],
        "response_schema": {
            "reasoning": "short explanation",
            "hypothesis": "testable structural hypothesis",
            "attn_levels": ["one integer per layer"],
            "mlp_levels": ["one integer per layer"],
            "evidence_used": ["parent IDs or concise evidence labels"],
        },
    }
    if experience_context is not None:
        prompt["retrieved_experience"] = dict(experience_context)
    return prompt


def _parse_level_array(value: Any, expected: int, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} must be a list with exactly {expected} entries")
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name}[{index}] must be an integer")
        parsed.append(int(item))
    return tuple(parsed)


def parse_freeform_response(
    response: JSONResponse,
    *,
    bounds: PolicyBounds,
) -> Tuple[LLMFreeformProposal, Optional[str]]:
    if isinstance(response, str):
        raw_response = response
        value = json.loads(response)
    elif isinstance(response, Mapping):
        raw_response = json.dumps(response, ensure_ascii=False)
        value = dict(response)
    else:
        raise ValueError("LLM response must be a JSON string or mapping")
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")

    reasoning = str(value.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("LLM response requires non-empty reasoning")
    policy = Policy(
        attn_levels=_parse_level_array(
            value.get("attn_levels"), bounds.num_layers, "attn_levels"
        ),
        mlp_levels=_parse_level_array(
            value.get("mlp_levels"), bounds.num_layers, "mlp_levels"
        ),
    )
    bounds.validate(policy)
    evidence_raw = value.get("evidence_used", [])
    if not isinstance(evidence_raw, list):
        raise ValueError("evidence_used must be a list")
    return (
        LLMFreeformProposal(
            reasoning=reasoning,
            hypothesis=str(value.get("hypothesis", "")).strip(),
            policy=policy,
            evidence_used=tuple(str(item) for item in evidence_raw),
        ),
        raw_response,
    )


class LLMFreeformOperator:
    """Generate and validate a complete LLM-proposed policy."""

    def __init__(
        self,
        bounds: PolicyBounds,
        sender: JSONSender,
        *,
        rng: Optional[random.Random] = None,
        fallback_balanced_moves: int = 1,
    ) -> None:
        self.bounds = bounds
        self.sender = sender
        self.rng = rng or random.Random()
        self.fallback_balanced_moves = fallback_balanced_moves

    def generate(
        self,
        parent_a: EvaluatedPolicy,
        parent_b: EvaluatedPolicy,
        *,
        objective: Mapping[str, Any],
        archive: Optional[ExperienceArchive] = None,
        use_experience: bool = False,
        archive_fingerprints: Sequence[Tuple[int, ...]] = (),
        cost_fn: Optional[CostFunction] = None,
        target_budget: Optional[float] = None,
        budget_tolerance: float = 0.01,
    ) -> LLMFreeformOutcome:
        if use_experience and archive is None:
            raise ValueError("use_experience=True requires an experience archive")
        experience_context = None
        if use_experience and archive is not None:
            target = objective.get("target_flops_reduction")
            tolerance = objective.get("flops_tolerance")
            experience_context = archive.build_prompt_context(
                parent_a.policy,
                parent_b.policy,
                target_flops_reduction=(None if target is None else float(target)),
                flops_tolerance=(None if tolerance is None else float(tolerance)),
            )
        prompt = build_freeform_prompt(
            self.bounds,
            parent_a,
            parent_b,
            objective=objective,
            experience_context=experience_context,
        )
        forbidden = {
            parent_a.policy.fingerprint,
            parent_b.policy.fingerprint,
            *archive_fingerprints,
        }
        if archive is not None:
            forbidden.update(entry.policy.fingerprint for entry in archive.entries)
        raw_response: Optional[str] = None
        try:
            response = self.sender(json.dumps(prompt, ensure_ascii=False))
            raw_response = (
                response
                if isinstance(response, str)
                else json.dumps(response, ensure_ascii=False)
            )
            proposal, _ = parse_freeform_response(response, bounds=self.bounds)
            child, repair = self._repair_if_requested(
                proposal.policy, cost_fn, target_budget, budget_tolerance
            )
            if repair is not None and not repair.satisfied:
                raise OperatorError(f"LLM child budget repair failed: {repair.reason}")
            if child.fingerprint in forbidden:
                raise OperatorError(
                    "LLM child duplicates a parent or previously evaluated policy"
                )
            return LLMFreeformOutcome(
                child=child,
                raw_child=proposal.policy,
                budget_repair=repair,
                proposal=proposal,
                prompt=prompt,
                raw_response=raw_response,
                fallback_used=False,
                fallback_reason=None,
            )
        except Exception as exc:
            return self._fallback(
                parent_a,
                parent_b,
                prompt,
                raw_response,
                str(exc),
                cost_fn,
                target_budget,
                budget_tolerance,
                forbidden,
            )

    def _repair_if_requested(
        self,
        child: Policy,
        cost_fn: Optional[CostFunction],
        target_budget: Optional[float],
        budget_tolerance: float,
    ) -> Tuple[Policy, Optional[BudgetRepairResult]]:
        if (cost_fn is None) != (target_budget is None):
            raise ValueError("cost_fn and target_budget must be supplied together")
        if cost_fn is None or target_budget is None:
            return child, None
        repair = repair_policy_budget(
            child,
            self.bounds,
            cost_fn,
            target=target_budget,
            tolerance=budget_tolerance,
        )
        return repair.policy, repair

    def _fallback(
        self,
        parent_a: EvaluatedPolicy,
        parent_b: EvaluatedPolicy,
        prompt: Dict[str, Any],
        raw_response: Optional[str],
        reason: str,
        cost_fn: Optional[CostFunction],
        target_budget: Optional[float],
        budget_tolerance: float,
        forbidden: Sequence[Tuple[int, ...]],
    ) -> LLMFreeformOutcome:
        forbidden_set = set(forbidden)
        child = parent_a.policy
        repair = None
        for _ in range(32):
            crossover = uniform_crossover(
                parent_a.policy, parent_b.policy, self.bounds, self.rng
            )
            raw_child = crossover.child
            if self.fallback_balanced_moves > 0:
                try:
                    raw_child = balanced_mutation(
                        raw_child,
                        self.bounds,
                        self.rng,
                        num_moves=self.fallback_balanced_moves,
                    ).child
                except OperatorError:
                    pass
            child, repair = self._repair_if_requested(
                raw_child, cost_fn, target_budget, budget_tolerance
            )
            if (repair is None or repair.satisfied) and child.fingerprint not in forbidden_set:
                break
        if child.fingerprint in forbidden_set:
            reason += "; fallback could not find a novel child in 32 attempts"
        return LLMFreeformOutcome(
            child=child,
            raw_child=None,
            budget_repair=repair,
            proposal=None,
            prompt=prompt,
            raw_response=raw_response,
            fallback_used=True,
            fallback_reason=reason,
        )
