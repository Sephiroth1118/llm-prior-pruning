"""LLM-guided crossover constrained by deterministic evolutionary operators."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from .budget import BudgetRepairResult, CostFunction, repair_policy_budget
from .experience import EvaluatedPolicy, ExperienceArchive
from .operators import (
    CrossoverResult,
    MutationEdit,
    OperatorError,
    apply_mutations,
    balanced_mutation,
    compile_inheritance_crossover,
    uniform_crossover,
)
from .policy import GeneRef, Policy, PolicyBounds


JSONResponse = Union[str, Mapping[str, Any]]
JSONSender = Callable[[str], JSONResponse]


@dataclass(frozen=True)
class LLMCrossoverProposal:
    reasoning: str
    attn_sources: Tuple[str, ...]
    mlp_sources: Tuple[str, ...]
    mutations: Tuple[MutationEdit, ...]
    hypothesis: str = ""
    evidence_used: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMCrossoverOutcome:
    child: Policy
    crossover: CrossoverResult
    proposed_mutations: Tuple[MutationEdit, ...]
    budget_repair: Optional[BudgetRepairResult]
    proposal: Optional[LLMCrossoverProposal]
    prompt: Dict[str, Any]
    raw_response: Optional[str]
    fallback_used: bool
    fallback_reason: Optional[str]


def build_crossover_prompt(
    bounds: PolicyBounds,
    parent_a: EvaluatedPolicy,
    parent_b: EvaluatedPolicy,
    *,
    objective: Mapping[str, Any],
    experience_context: Optional[Mapping[str, Any]] = None,
    max_mutations: int = 4,
) -> Dict[str, Any]:
    """Build the controlled prompt shared by faithful and experience variants."""

    prompt: Dict[str, Any] = {
        "task": (
            "Act as a crossover operator. Inherit every gene from parent A or B, "
            "then optionally propose a few one-level mutations. Do not output a full "
            "free-form child policy."
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
            "A": _evaluated_parent_to_dict(parent_a),
            "B": _evaluated_parent_to_dict(parent_b),
        },
        "rules": [
            f"attn_sources and mlp_sources must each have exactly {bounds.num_layers} entries",
            "every source entry must be exactly A or B",
            f"mutations may contain at most {max_mutations} entries",
            "each mutation delta must be exactly -1 or +1",
            "mutations must respect the supplied per-layer bounds",
            "the effective child must differ from both parents when a novel child is feasible",
            "do not merely copy the higher-fitness parent",
            "prefer a testable structural hypothesis over arbitrary mixing",
            "return one JSON object and no markdown",
        ],
        "response_schema": {
            "reasoning": "short explanation",
            "hypothesis": "testable structural hypothesis",
            "attn_sources": ["A", "B", "... one per layer"],
            "mlp_sources": ["A", "B", "... one per layer"],
            "mutations": [
                {"layer": 0, "module": "attn or mlp", "delta": 1}
            ],
            "evidence_used": ["IDs or concise evidence labels"],
        },
    }
    if experience_context is not None:
        prompt["retrieved_experience"] = dict(experience_context)
    return prompt


def _evaluated_parent_to_dict(parent: EvaluatedPolicy) -> Dict[str, Any]:
    return {
        "source_id": parent.source_id,
        "policy": parent.policy.to_dict(),
        "fitness": parent.fitness,
        "kl": parent.kl,
        "ppl": parent.ppl,
        "flops_reduction": parent.flops_reduction,
    }


def parse_crossover_response(
    response: JSONResponse,
    *,
    num_layers: int,
    max_mutations: int,
) -> Tuple[LLMCrossoverProposal, Optional[str]]:
    """Parse the model JSON; semantic bounds are enforced when compiling it."""

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
    attn_sources = _parse_sources(value.get("attn_sources"), num_layers, "attn_sources")
    mlp_sources = _parse_sources(value.get("mlp_sources"), num_layers, "mlp_sources")

    raw_mutations = value.get("mutations", [])
    if not isinstance(raw_mutations, list):
        raise ValueError("mutations must be a list")
    if len(raw_mutations) > max_mutations:
        raise ValueError(
            f"received {len(raw_mutations)} mutations; maximum is {max_mutations}"
        )
    mutations = []
    mutation_targets = set()
    for index, item in enumerate(raw_mutations):
        if not isinstance(item, Mapping):
            raise ValueError(f"mutations[{index}] must be an object")
        layer = item.get("layer")
        delta = item.get("delta")
        module = item.get("module")
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError(f"mutations[{index}].layer must be an integer")
        if module not in ("attn", "mlp"):
            raise ValueError(f"mutations[{index}].module must be attn or mlp")
        if isinstance(delta, bool) or not isinstance(delta, int) or delta not in (-1, 1):
            raise ValueError(f"mutations[{index}].delta must be -1 or +1")
        target = (layer, module)
        if target in mutation_targets:
            raise ValueError(f"mutations contains duplicate target {module}[{layer}]")
        mutation_targets.add(target)
        mutations.append(MutationEdit(layer=layer, module=module, delta=delta))

    evidence_used_raw = value.get("evidence_used", [])
    if not isinstance(evidence_used_raw, list):
        raise ValueError("evidence_used must be a list")
    proposal = LLMCrossoverProposal(
        reasoning=reasoning,
        hypothesis=str(value.get("hypothesis", "")).strip(),
        attn_sources=attn_sources,
        mlp_sources=mlp_sources,
        mutations=tuple(mutations),
        evidence_used=tuple(str(item) for item in evidence_used_raw),
    )
    return proposal, raw_response


def _parse_sources(value: Any, expected: int, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} must be a list with exactly {expected} entries")
    sources = tuple(str(item).upper() for item in value)
    if any(item not in ("A", "B") for item in sources):
        raise ValueError(f"{name} entries must be A or B")
    return sources


class LLMCrossoverOperator:
    """Generate a validated child, with random crossover as a safe fallback."""

    def __init__(
        self,
        bounds: PolicyBounds,
        sender: JSONSender,
        *,
        rng: Optional[random.Random] = None,
        max_mutations: int = 4,
        fallback_balanced_moves: int = 1,
    ) -> None:
        if max_mutations < 0:
            raise ValueError("max_mutations must be non-negative")
        self.bounds = bounds
        self.sender = sender
        self.rng = rng or random.Random()
        self.max_mutations = max_mutations
        self.fallback_balanced_moves = fallback_balanced_moves

    def generate(
        self,
        parent_a: EvaluatedPolicy,
        parent_b: EvaluatedPolicy,
        *,
        objective: Mapping[str, Any],
        archive: Optional[ExperienceArchive] = None,
        use_experience: bool = False,
        cost_fn: Optional[CostFunction] = None,
        target_budget: Optional[float] = None,
        budget_tolerance: float = 0.01,
    ) -> LLMCrossoverOutcome:
        if use_experience and archive is None:
            raise ValueError("use_experience=True requires an experience archive")
        experience_context = None
        if use_experience and archive is not None:
            experience_context = archive.build_prompt_context(
                parent_a.policy,
                parent_b.policy,
                target_flops_reduction=_optional_float(
                    objective.get("target_flops_reduction")
                ),
                flops_tolerance=_optional_float(objective.get("flops_tolerance")),
            )
        prompt = build_crossover_prompt(
            self.bounds,
            parent_a,
            parent_b,
            objective=objective,
            experience_context=experience_context,
            max_mutations=self.max_mutations,
        )
        forbidden_fingerprints = {
            parent_a.policy.fingerprint,
            parent_b.policy.fingerprint,
        }
        if archive is not None:
            forbidden_fingerprints.update(
                entry.policy.fingerprint for entry in archive.entries
            )

        raw_response: Optional[str] = None
        try:
            response = self.sender(json.dumps(prompt, ensure_ascii=False))
            raw_response = (
                response
                if isinstance(response, str)
                else json.dumps(response, ensure_ascii=False)
            )
            proposal, _ = parse_crossover_response(
                response,
                num_layers=self.bounds.num_layers,
                max_mutations=self.max_mutations,
            )
            crossover = compile_inheritance_crossover(
                parent_a.policy,
                parent_b.policy,
                self.bounds,
                proposal.attn_sources,
                proposal.mlp_sources,
            )
            mutated = apply_mutations(
                crossover.child,
                self.bounds,
                proposal.mutations,
                max_edits=self.max_mutations,
            )
            child, repair = self._repair_if_requested(
                mutated.child,
                cost_fn,
                target_budget,
                budget_tolerance,
                protected_refs=tuple(edit.ref for edit in proposal.mutations),
            )
            if repair is not None and not repair.satisfied:
                raise OperatorError(f"LLM child budget repair failed: {repair.reason}")
            if child.fingerprint in forbidden_fingerprints:
                raise OperatorError(
                    "LLM child duplicates a parent or previously evaluated policy"
                )
            return LLMCrossoverOutcome(
                child=child,
                crossover=crossover,
                proposed_mutations=proposal.mutations,
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
                forbidden_fingerprints,
            )

    def _repair_if_requested(
        self,
        child: Policy,
        cost_fn: Optional[CostFunction],
        target_budget: Optional[float],
        budget_tolerance: float,
        protected_refs: Sequence[GeneRef] = (),
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
            protected_refs=protected_refs,
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
        forbidden_fingerprints: Sequence[Tuple[int, ...]],
    ) -> LLMCrossoverOutcome:
        forbidden = set(forbidden_fingerprints)
        crossover = uniform_crossover(parent_a.policy, parent_b.policy, self.bounds, self.rng)
        child = crossover.child
        mutations: Tuple[MutationEdit, ...] = ()
        repair = None
        for _ in range(32):
            crossover = uniform_crossover(
                parent_a.policy, parent_b.policy, self.bounds, self.rng
            )
            child = crossover.child
            mutations = ()
            if self.fallback_balanced_moves > 0:
                try:
                    mutation_result = balanced_mutation(
                        child,
                        self.bounds,
                        self.rng,
                        num_moves=self.fallback_balanced_moves,
                    )
                    child = mutation_result.child
                    mutations = mutation_result.edits
                except OperatorError:
                    pass
            child, repair = self._repair_if_requested(
                child, cost_fn, target_budget, budget_tolerance
            )
            if (repair is None or repair.satisfied) and child.fingerprint not in forbidden:
                break
        if child.fingerprint in forbidden:
            reason += "; fallback could not find a novel child in 32 attempts"
        return LLMCrossoverOutcome(
            child=child,
            crossover=crossover,
            proposed_mutations=mutations,
            budget_repair=repair,
            proposal=None,
            prompt=prompt,
            raw_response=raw_response,
            fallback_used=True,
            fallback_reason=reason,
        )


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
