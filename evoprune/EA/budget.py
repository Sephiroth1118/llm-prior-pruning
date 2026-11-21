"""Cheap policy cost proxies and deterministic budget repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from .operators import MutationEdit
from .policy import GeneRef, Policy, PolicyBounds


CostFunction = Callable[[Policy], float]


@dataclass(frozen=True)
class BudgetRepairResult:
    policy: Policy
    cost: float
    target: float
    tolerance: float
    satisfied: bool
    edits: Tuple[MutationEdit, ...]
    reason: Optional[str] = None


def make_normalized_level_cost_fn(
    bounds: PolicyBounds,
    *,
    attn_weight: float = 1.0,
    mlp_weight: float = 1.0,
) -> CostFunction:
    """Return a monotonic [0, 1] budget proxy for the chosen compression levels."""

    if attn_weight < 0 or mlp_weight < 0 or attn_weight + mlp_weight <= 0:
        raise ValueError("cost weights must be non-negative and not both zero")
    weighted_genes = []
    for ref in bounds.iter_genes():
        gene_bounds = bounds.get(ref)
        if gene_bounds.width:
            weight = attn_weight if ref.module == "attn" else mlp_weight
            if weight:
                weighted_genes.append((ref, gene_bounds, weight))
    denominator = sum(weight for _, _, weight in weighted_genes)
    if denominator == 0:
        raise ValueError("bounds contain no mutable genes with positive weight")

    def cost(policy: Policy) -> float:
        bounds.validate(policy)
        total = sum(
            weight
            * (policy.get(ref) - gene_bounds.minimum)
            / gene_bounds.width
            for ref, gene_bounds, weight in weighted_genes
        )
        return total / denominator

    return cost


def make_transformer_proxy_flops_cost_fn(
    bounds: PolicyBounds,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
) -> CostFunction:
    """Return a normalized parameter/FLOPs proxy matching the warm-start weights."""

    attn_weight, mlp_weight = transformer_proxy_module_weights(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    return make_normalized_level_cost_fn(
        bounds, attn_weight=attn_weight, mlp_weight=mlp_weight
    )


def transformer_proxy_module_weights(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
) -> Tuple[float, float]:
    """Return the attention/MLP weights shared by budget and diversity logic."""

    if min(hidden_size, intermediate_size, num_attention_heads) <= 0:
        raise ValueError("transformer dimensions must be positive")
    kv_heads = num_key_value_heads or num_attention_heads
    if kv_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError("invalid attention head configuration")
    head_dim = hidden_size // num_attention_heads
    kv_hidden = kv_heads * head_dim
    attn_weight = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_hidden
    mlp_weight = 3 * hidden_size * intermediate_size
    return float(attn_weight), float(mlp_weight)


def repair_policy_budget(
    policy: Policy,
    bounds: PolicyBounds,
    cost_fn: CostFunction,
    *,
    target: float,
    tolerance: float,
    max_steps: Optional[int] = None,
    protected_refs: Iterable[GeneRef] = (),
) -> BudgetRepairResult:
    """Greedily take deterministic one-level steps toward a target cost.

    The repair never reports success unless the returned policy is within the
    requested tolerance. Ties prefer attention, then lower layer index.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    bounds.validate(policy)
    if max_steps is None:
        max_steps = sum(bounds.get(ref).width for ref in bounds.iter_genes())
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    protected = frozenset(protected_refs)
    current = policy
    edits = []
    current_cost = float(cost_fn(current))
    if abs(current_cost - target) <= tolerance:
        return BudgetRepairResult(
            current, current_cost, target, tolerance, True, tuple(edits)
        )

    for _ in range(max_steps):
        current_error = abs(current_cost - target)
        direction = 1 if current_cost < target else -1
        candidates = []
        for ref in bounds.iter_genes():
            candidate_value = current.get(ref) + direction
            if not bounds.get(ref).contains(candidate_value):
                continue
            neighbor = current.replace(ref, candidate_value)
            neighbor_cost = float(cost_fn(neighbor))
            neighbor_error = abs(neighbor_cost - target)
            if neighbor_error + 1e-15 < current_error:
                module_order = 0 if ref.module == "attn" else 1
                candidates.append(
                    (
                        1 if ref in protected else 0,
                        neighbor_error,
                        module_order,
                        ref.layer,
                        neighbor_cost,
                        neighbor,
                        ref,
                    )
                )
        if not candidates:
            return BudgetRepairResult(
                current,
                current_cost,
                target,
                tolerance,
                False,
                tuple(edits),
                "no legal one-level edit improves the target error",
            )
        _, _, _, _, current_cost, current, chosen_ref = min(
            candidates, key=lambda x: x[:4]
        )
        edits.append(MutationEdit(chosen_ref.layer, chosen_ref.module, direction))
        if abs(current_cost - target) <= tolerance:
            return BudgetRepairResult(
                current, current_cost, target, tolerance, True, tuple(edits)
            )

    return BudgetRepairResult(
        current,
        current_cost,
        target,
        tolerance,
        False,
        tuple(edits),
        f"maximum repair steps ({max_steps}) reached",
    )
