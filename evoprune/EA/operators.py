"""Deterministic evolutionary operators for structured-pruning policies."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

from .mutation_kernel import MutationKernel, MutationRoute, TRANSFER_KINDS
from .policy import GeneRef, ModuleKind, Policy, PolicyBounds


class OperatorError(ValueError):
    """Raised when an operator proposal is malformed or infeasible."""


@dataclass(frozen=True)
class MutationEdit:
    layer: int
    module: ModuleKind
    delta: int

    @property
    def ref(self) -> GeneRef:
        return GeneRef(self.layer, self.module)


@dataclass(frozen=True)
class CrossoverResult:
    child: Policy
    attn_sources: Tuple[str, ...]
    mlp_sources: Tuple[str, ...]


@dataclass(frozen=True)
class MutationResult:
    child: Policy
    edits: Tuple[MutationEdit, ...]


@dataclass(frozen=True)
class WeightedMutationResult:
    child: Policy
    edits: Tuple[MutationEdit, ...]
    requested_strength: float
    actual_strength: float
    budget_before: float
    budget_after: float
    strength_reached: bool
    reason: Optional[str] = None
    route_names: Tuple[str, ...] = ()


def _validate_parents(parent_a: Policy, parent_b: Policy, bounds: PolicyBounds) -> None:
    if parent_a.num_layers != parent_b.num_layers:
        raise OperatorError("parents have different layer counts")
    if parent_a.num_layers != bounds.num_layers:
        raise OperatorError("parent layer count does not match bounds")
    for name, parent in (("parent_a", parent_a), ("parent_b", parent_b)):
        try:
            bounds.validate(parent)
        except ValueError as exc:
            raise OperatorError(f"{name} is outside bounds: {exc}") from exc


def _normalize_sources(sources: Sequence[str], expected: int, name: str) -> Tuple[str, ...]:
    if len(sources) != expected:
        raise OperatorError(f"{name} must contain exactly {expected} entries")
    normalized = tuple(str(source).upper() for source in sources)
    invalid = sorted(set(normalized) - {"A", "B"})
    if invalid:
        raise OperatorError(f"{name} contains invalid parent labels: {invalid}")
    return normalized


def compile_inheritance_crossover(
    parent_a: Policy,
    parent_b: Policy,
    bounds: PolicyBounds,
    attn_sources: Sequence[str],
    mlp_sources: Sequence[str],
) -> CrossoverResult:
    """Compile an explicit A/B inheritance mask into a validated child."""

    _validate_parents(parent_a, parent_b, bounds)
    attn_mask = _normalize_sources(attn_sources, bounds.num_layers, "attn_sources")
    mlp_mask = _normalize_sources(mlp_sources, bounds.num_layers, "mlp_sources")
    child = Policy(
        attn_levels=tuple(
            parent_a.attn_levels[i] if source == "A" else parent_b.attn_levels[i]
            for i, source in enumerate(attn_mask)
        ),
        mlp_levels=tuple(
            parent_a.mlp_levels[i] if source == "A" else parent_b.mlp_levels[i]
            for i, source in enumerate(mlp_mask)
        ),
    )
    return CrossoverResult(child, attn_mask, mlp_mask)


def uniform_crossover(
    parent_a: Policy,
    parent_b: Policy,
    bounds: PolicyBounds,
    rng: random.Random,
) -> CrossoverResult:
    """Independently inherit every gene from either parent with equal probability."""

    attn_sources = tuple(rng.choice(("A", "B")) for _ in range(bounds.num_layers))
    mlp_sources = tuple(rng.choice(("A", "B")) for _ in range(bounds.num_layers))
    return compile_inheritance_crossover(
        parent_a, parent_b, bounds, attn_sources, mlp_sources
    )


def apply_mutations(
    policy: Policy,
    bounds: PolicyBounds,
    edits: Iterable[MutationEdit],
    *,
    max_edits: int = 4,
    max_abs_delta: int = 1,
) -> MutationResult:
    """Apply a small, bounded edit list and reject invalid intermediate policies."""

    try:
        bounds.validate(policy)
    except ValueError as exc:
        raise OperatorError(f"input policy is outside bounds: {exc}") from exc
    edits_tuple = tuple(edits)
    if len(edits_tuple) > max_edits:
        raise OperatorError(f"received {len(edits_tuple)} edits; maximum is {max_edits}")

    child = policy
    for edit in edits_tuple:
        if isinstance(edit.delta, bool) or not isinstance(edit.delta, int):
            raise OperatorError("mutation delta must be an integer")
        if edit.delta == 0 or abs(edit.delta) > max_abs_delta:
            raise OperatorError(
                f"mutation delta must be non-zero with |delta| <= {max_abs_delta}"
            )
        try:
            old_value = child.get(edit.ref)
            gene_bounds = bounds.get(edit.ref)
        except (IndexError, ValueError) as exc:
            raise OperatorError(f"invalid mutation target {edit.ref}: {exc}") from exc
        new_value = old_value + edit.delta
        if not gene_bounds.contains(new_value):
            raise OperatorError(
                f"mutation sends {edit.ref} from {old_value} to {new_value}, outside "
                f"[{gene_bounds.minimum}, {gene_bounds.maximum}]"
            )
        child = child.replace(edit.ref, new_value)

    return MutationResult(child=child, edits=edits_tuple)


def _feasible_balanced_pairs(
    policy: Policy, bounds: PolicyBounds, module: ModuleKind
) -> Tuple[Tuple[GeneRef, GeneRef], ...]:
    decrease = tuple(
        ref
        for ref in bounds.iter_genes(module)
        if policy.get(ref) > bounds.get(ref).minimum
    )
    increase = tuple(
        ref
        for ref in bounds.iter_genes(module)
        if policy.get(ref) < bounds.get(ref).maximum
    )
    return tuple((down, up) for down in decrease for up in increase if down != up)


def balanced_mutation(
    policy: Policy,
    bounds: PolicyBounds,
    rng: random.Random,
    *,
    num_moves: int = 1,
    module: Optional[ModuleKind] = None,
) -> MutationResult:
    """Move one level between two layers of the same module type.

    Each move changes one gene by -1 and another by +1. Consequently the raw
    level sum for attention and MLP is preserved independently.
    """

    if num_moves < 0:
        raise OperatorError("num_moves must be non-negative")
    if module not in (None, "attn", "mlp"):
        raise OperatorError(f"unknown module type: {module}")
    try:
        bounds.validate(policy)
    except ValueError as exc:
        raise OperatorError(f"input policy is outside bounds: {exc}") from exc

    child = policy
    applied = []
    allowed_modules: Tuple[ModuleKind, ...] = (module,) if module else ("attn", "mlp")
    for move_index in range(num_moves):
        feasible = {
            kind: _feasible_balanced_pairs(child, bounds, kind)
            for kind in allowed_modules
        }
        feasible = {kind: pairs for kind, pairs in feasible.items() if pairs}
        if not feasible:
            if move_index == 0 and num_moves:
                raise OperatorError("no feasible balanced mutation exists")
            break
        chosen_module = rng.choice(tuple(sorted(feasible)))
        down, up = rng.choice(feasible[chosen_module])
        move_edits = (
            MutationEdit(down.layer, down.module, -1),
            MutationEdit(up.layer, up.module, +1),
        )
        child = apply_mutations(child, bounds, move_edits, max_edits=2).child
        applied.extend(move_edits)

    return MutationResult(child=child, edits=tuple(applied))


def _proxy_level_costs(
    policy: Policy,
    bounds: PolicyBounds,
    cost_fn: Callable[[Policy], float],
) -> Dict[GeneRef, float]:
    base_cost = float(cost_fn(policy))
    costs = {}
    for ref in bounds.iter_genes():
        gene_bounds = bounds.get(ref)
        if gene_bounds.width <= 0:
            continue
        delta = 1 if policy.get(ref) < gene_bounds.maximum else -1
        neighbor = policy.replace(ref, policy.get(ref) + delta)
        level_cost = abs(float(cost_fn(neighbor)) - base_cost)
        if level_cost > 0.0:
            costs[ref] = level_cost
    return costs


def _weighted_change_mass(
    parent: Policy,
    child: Policy,
    level_costs: Dict[GeneRef, float],
) -> float:
    return 0.5 * sum(
        level_cost * abs(child.get(ref) - parent.get(ref))
        for ref, level_cost in level_costs.items()
    )


def _legal_refs(
    policy: Policy,
    bounds: PolicyBounds,
    module: ModuleKind,
    delta: int,
    *,
    exclude: Iterable[GeneRef] = (),
) -> Tuple[GeneRef, ...]:
    excluded = frozenset(exclude)
    return tuple(
        ref
        for ref in bounds.iter_genes(module)
        if ref not in excluded
        and bounds.get(ref).contains(policy.get(ref) + delta)
    )


def _weighted_ref_choice(
    refs: Sequence[GeneRef],
    rng: random.Random,
    kernel: Optional[MutationKernel],
    delta: int,
) -> GeneRef:
    if kernel is None:
        return rng.choice(refs)
    return rng.choices(
        refs,
        weights=[kernel.gene_weight(ref, delta) for ref in refs],
        k=1,
    )[0]


def _route_refs(
    policy: Policy,
    bounds: PolicyBounds,
    module: ModuleKind,
    layers: Sequence[int],
    delta: int,
    *,
    exclude: Iterable[GeneRef] = (),
) -> Tuple[GeneRef, ...]:
    excluded = frozenset(exclude)
    return tuple(
        GeneRef(layer, module)
        for layer in layers
        if GeneRef(layer, module) not in excluded
        and bounds.get(GeneRef(layer, module)).contains(
            policy.get(GeneRef(layer, module)) + delta
        )
    )


def _sample_route_ref(refs, route: MutationRoute, endpoint: str, rng: random.Random):
    if not getattr(route, f"{endpoint}_layer_weights"):
        return rng.choice(refs)
    return rng.choices(
        refs, weights=[route.endpoint_weight(endpoint, ref.layer) for ref in refs], k=1
    )[0]


def _sample_route_transfer(
    policy: Policy,
    bounds: PolicyBounds,
    rng: random.Random,
    level_costs: Dict[GeneRef, float],
    big_module: ModuleKind,
    route: MutationRoute,
) -> Optional[Tuple[Policy, Tuple[MutationEdit, ...]]]:
    """Sample one explicit donor-to-receiver route and balance its FLOPs mass."""

    if route.donor_module == route.receiver_module:
        donors = _route_refs(
            policy, bounds, route.donor_module, route.endpoint_layers("donor"), -1
        )
        if not donors:
            return None
        donor = _sample_route_ref(donors, route, "donor", rng)
        after_donor = policy.replace(donor, policy.get(donor) - 1)
        receivers = _route_refs(
            after_donor,
            bounds,
            route.receiver_module,
            route.endpoint_layers("receiver"),
            1,
            exclude=(donor,),
        )
        if not receivers:
            return None
        receiver = _sample_route_ref(receivers, route, "receiver", rng)
        return after_donor.replace(receiver, after_donor.get(receiver) + 1), (
            MutationEdit(donor.layer, donor.module, -1),
            MutationEdit(receiver.layer, receiver.module, 1),
        )

    big_is_donor = route.donor_module == big_module
    big_module_for_route = route.donor_module if big_is_donor else route.receiver_module
    big_endpoint = "donor" if big_is_donor else "receiver"
    small_endpoint = "receiver" if big_is_donor else "donor"
    big_layers = route.endpoint_layers(big_endpoint)
    big_delta = -1 if big_is_donor else 1
    small_module = route.receiver_module if big_is_donor else route.donor_module
    small_layers = route.endpoint_layers(small_endpoint)
    small_delta = -big_delta
    big_refs = tuple(
        ref
        for ref in _route_refs(
            policy, bounds, big_module_for_route, big_layers, big_delta
        )
        if ref in level_costs
    )
    if not big_refs:
        return None
    big_ref = _sample_route_ref(big_refs, route, big_endpoint, rng)
    child = policy.replace(big_ref, policy.get(big_ref) + big_delta)
    edits = [MutationEdit(big_ref.layer, big_ref.module, big_delta)]
    target_mass = level_costs[big_ref]
    small_mass = 0.0

    while small_mass + 1e-15 < target_mass:
        small_refs = tuple(
            ref
            for ref in _route_refs(
                child, bounds, small_module, small_layers, small_delta
            )
            if ref in level_costs
        )
        if not small_refs:
            return None
        small_ref = _sample_route_ref(small_refs, route, small_endpoint, rng)
        level_cost = level_costs[small_ref]
        remaining = target_mass - small_mass
        if level_cost > remaining and rng.random() >= remaining / level_cost:
            break
        child = child.replace(small_ref, child.get(small_ref) + small_delta)
        edits.append(MutationEdit(small_ref.layer, small_ref.module, small_delta))
        small_mass += level_cost
        if level_cost >= remaining:
            break

    if len(edits) == 1:
        return None
    return child, tuple(edits)


def _sample_same_module_transfer(
    policy: Policy,
    bounds: PolicyBounds,
    rng: random.Random,
    module: ModuleKind,
    kernel: Optional[MutationKernel] = None,
) -> Optional[Tuple[Policy, Tuple[MutationEdit, ...]]]:
    down_refs = _legal_refs(policy, bounds, module, -1)
    if not down_refs:
        return None
    down_ref = _weighted_ref_choice(down_refs, rng, kernel, -1)
    after_down = policy.replace(down_ref, policy.get(down_ref) - 1)
    up_refs = _legal_refs(after_down, bounds, module, 1, exclude=(down_ref,))
    if not up_refs:
        return None
    up_ref = _weighted_ref_choice(up_refs, rng, kernel, 1)
    child = after_down.replace(up_ref, after_down.get(up_ref) + 1)
    return child, (
        MutationEdit(down_ref.layer, down_ref.module, -1),
        MutationEdit(up_ref.layer, up_ref.module, 1),
    )


def _sample_cross_module_transfer(
    policy: Policy,
    bounds: PolicyBounds,
    rng: random.Random,
    level_costs: Dict[GeneRef, float],
    *,
    big_module: ModuleKind,
    big_delta: int,
    kernel: Optional[MutationKernel] = None,
) -> Optional[Tuple[Policy, Tuple[MutationEdit, ...]]]:
    small_module: ModuleKind = "mlp" if big_module == "attn" else "attn"
    big_refs = tuple(
        ref
        for ref in _legal_refs(policy, bounds, big_module, big_delta)
        if ref in level_costs
    )
    if not big_refs:
        return None
    big_ref = _weighted_ref_choice(big_refs, rng, kernel, big_delta)
    child = policy.replace(big_ref, policy.get(big_ref) + big_delta)
    edits = [MutationEdit(big_ref.layer, big_ref.module, big_delta)]
    target_mass = level_costs[big_ref]
    small_delta = -big_delta
    small_mass = 0.0

    while small_mass + 1e-15 < target_mass:
        small_refs = tuple(
            ref
            for ref in _legal_refs(child, bounds, small_module, small_delta)
            if ref in level_costs
        )
        if not small_refs:
            return None
        small_ref = _weighted_ref_choice(small_refs, rng, kernel, small_delta)
        level_cost = level_costs[small_ref]
        remaining = target_mass - small_mass
        if level_cost > remaining and rng.random() >= remaining / level_cost:
            break
        child = child.replace(small_ref, child.get(small_ref) + small_delta)
        edits.append(MutationEdit(small_ref.layer, small_ref.module, small_delta))
        small_mass += level_cost
        if level_cost >= remaining:
            break

    if len(edits) == 1:
        return None
    return child, tuple(edits)


def weighted_budget_transfer_mutation(
    policy: Policy,
    bounds: PolicyBounds,
    cost_fn: Callable[[Policy], float],
    rng: random.Random,
    *,
    target_strength: float,
    budget_tolerance: float = 0.001,
    max_attempts: int = 2048,
    kernel: Optional[MutationKernel] = None,
) -> WeightedMutationResult:
    """Randomly move proxy FLOPs mass while preserving the parent's budget.

    Same-module transfers exchange one level between two layers. Cross-module
    transfers exchange one level of the more expensive module for a
    stochastically rounded number of levels from the cheaper module.
    """

    if target_strength < 0.0:
        raise OperatorError("target_strength must be non-negative")
    if budget_tolerance < 0.0:
        raise OperatorError("budget_tolerance must be non-negative")
    if max_attempts <= 0:
        raise OperatorError("max_attempts must be positive")
    try:
        bounds.validate(policy)
    except ValueError as exc:
        raise OperatorError(f"input policy is outside bounds: {exc}") from exc
    if kernel is not None and kernel.num_layers != bounds.num_layers:
        raise OperatorError("mutation kernel layer count does not match bounds")

    budget_before = float(cost_fn(policy))
    level_costs = _proxy_level_costs(policy, bounds, cost_fn)
    if target_strength == 0.0:
        return WeightedMutationResult(
            policy, (), 0.0, 0.0, budget_before, budget_before, True
        )
    if not level_costs:
        return WeightedMutationResult(
            policy,
            (),
            target_strength,
            0.0,
            budget_before,
            budget_before,
            False,
            "policy bounds contain no mutable positive-cost genes",
        )

    module_costs = {
        module: [
            level_costs[ref]
            for ref in bounds.iter_genes(module)
            if ref in level_costs
        ]
        for module in ("attn", "mlp")
    }
    big_module: ModuleKind = max(
        ("attn", "mlp"),
        key=lambda module: (
            sum(module_costs[module]) / len(module_costs[module])
            if module_costs[module]
            else 0.0
        ),
    )
    strength_tolerance = min(level_costs.values()) / 2.0
    child = policy
    edits = []
    route_names = []
    actual_strength = 0.0

    for _ in range(max_attempts):
        if abs(actual_strength - target_strength) <= strength_tolerance:
            break
        route = None
        if kernel is not None and kernel.routes:
            route = rng.choices(
                kernel.routes,
                weights=[candidate.weight for candidate in kernel.routes],
                k=1,
            )[0]
            proposal = _sample_route_transfer(
                child, bounds, rng, level_costs, big_module, route
            )
        else:
            transfer_kind = (
                rng.choice(TRANSFER_KINDS)
                if kernel is None
                else rng.choices(
                    TRANSFER_KINDS,
                    weights=[kernel.transfer_weight(kind) for kind in TRANSFER_KINDS],
                    k=1,
                )[0]
            )
            if transfer_kind in ("same_attention", "same_mlp"):
                proposal = _sample_same_module_transfer(
                    child,
                    bounds,
                    rng,
                    "attn" if transfer_kind == "same_attention" else "mlp",
                    kernel,
                )
            else:
                proposal = _sample_cross_module_transfer(
                    child,
                    bounds,
                    rng,
                    level_costs,
                    big_module=big_module,
                    big_delta=(
                        1
                        if transfer_kind == "compress_expensive_release_cheap"
                        else -1
                    ),
                    kernel=kernel,
                )
        if proposal is None:
            continue
        proposed_child, proposed_edits = proposal
        proposed_budget = float(cost_fn(proposed_child))
        if abs(proposed_budget - budget_before) > budget_tolerance + 1e-15:
            continue
        proposed_strength = _weighted_change_mass(
            policy, proposed_child, level_costs
        )
        if proposed_strength <= actual_strength + 1e-15:
            continue
        if proposed_strength > target_strength + strength_tolerance:
            continue
        if (
            abs(proposed_strength - target_strength) + 1e-15
            >= abs(actual_strength - target_strength)
        ):
            continue
        child = proposed_child
        edits.extend(proposed_edits)
        if route is not None:
            route_names.append(route.name)
        actual_strength = proposed_strength

    budget_after = float(cost_fn(child))
    strength_reached = abs(actual_strength - target_strength) <= strength_tolerance
    reason = None
    if not strength_reached:
        reason = "no legal budget-preserving transfer moved closer to requested strength"
    return WeightedMutationResult(
        child=child,
        edits=tuple(edits),
        requested_strength=target_strength,
        actual_strength=actual_strength,
        budget_before=budget_before,
        budget_after=budget_after,
        strength_reached=strength_reached,
        reason=reason,
        route_names=tuple(route_names),
    )
