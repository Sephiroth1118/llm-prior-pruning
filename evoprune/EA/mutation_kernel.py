"""Search distributions for FLOPs-aware policy mutation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch

from .policy import GeneRef, ModuleKind, Policy


TRANSFER_KINDS = (
    "same_attention",
    "same_mlp",
    "release_expensive_compress_cheap",
    "compress_expensive_release_cheap",
)


@dataclass(frozen=True)
class MutationRoute:
    """An auditable compression transfer from donor layers to receiver layers."""

    name: str
    donor_module: ModuleKind
    donor_layers: Tuple[int, ...]
    receiver_module: ModuleKind
    receiver_layers: Tuple[int, ...]
    weight: float = 1.0
    donor_layer_weights: Tuple[float, ...] = ()
    receiver_layer_weights: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("mutation route name must be non-empty")
        if self.donor_module not in {"attn", "mlp"}:
            raise ValueError(f"invalid donor module: {self.donor_module}")
        if self.receiver_module not in {"attn", "mlp"}:
            raise ValueError(f"invalid receiver module: {self.receiver_module}")
        for field_name in ("donor_layers", "receiver_layers"):
            layers = tuple(dict.fromkeys(int(layer) for layer in getattr(self, field_name)))
            # Soft endpoints may have no preferred layers; their full weights
            # still provide sampling support. Hard endpoints require support.
            endpoint_weights = getattr(self, field_name.replace("_layers", "_layer_weights"))
            if (not layers and not endpoint_weights) or any(layer < 0 for layer in layers):
                raise ValueError(f"{field_name} must contain non-negative layer indices")
            object.__setattr__(self, field_name, layers)
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("mutation route weight must be finite and positive")
        object.__setattr__(self, "weight", float(self.weight))
        for name in ("donor_layer_weights", "receiver_layer_weights"):
            values = tuple(float(value) for value in getattr(self, name))
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError("route layer weights must be finite and positive")
            object.__setattr__(self, name, values)
        if bool(self.donor_layer_weights) != bool(self.receiver_layer_weights):
            raise ValueError("soft routes require weights for both endpoints")

    def endpoint_layers(self, endpoint: str) -> Tuple[int, ...]:
        weights = getattr(self, f"{endpoint}_layer_weights")
        return tuple(range(len(weights))) if weights else getattr(self, f"{endpoint}_layers")

    def endpoint_weight(self, endpoint: str, layer: int) -> float:
        weights = getattr(self, f"{endpoint}_layer_weights")
        return weights[layer] if weights else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "donor": {
                "module": self.donor_module,
                "layers": list(self.donor_layers),
                **({"layer_weights": list(self.donor_layer_weights)} if self.donor_layer_weights else {}),
            },
            "receiver": {
                "module": self.receiver_module,
                "layers": list(self.receiver_layers),
                **({"layer_weights": list(self.receiver_layer_weights)} if self.receiver_layer_weights else {}),
            },
            "weight": self.weight,
        }


def _positive_tuple(values: Iterable[float], expected: int, name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


@dataclass(frozen=True)
class MutationKernel:
    """Factorized gene-direction and transfer-type mutation weights."""

    attn_compress: Tuple[float, ...]
    attn_decompress: Tuple[float, ...]
    mlp_compress: Tuple[float, ...]
    mlp_decompress: Tuple[float, ...]
    transfer_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    source: str = "uniform"
    version: int = 0
    hypothesis: str = "uniform mutation"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    routes: Tuple[MutationRoute, ...] = ()

    def __post_init__(self) -> None:
        num_layers = len(self.attn_compress)
        if num_layers <= 0:
            raise ValueError("mutation kernel must contain at least one layer")
        for name in (
            "attn_compress",
            "attn_decompress",
            "mlp_compress",
            "mlp_decompress",
        ):
            object.__setattr__(
                self,
                name,
                _positive_tuple(getattr(self, name), num_layers, name),
            )
        object.__setattr__(
            self,
            "transfer_weights",
            _positive_tuple(self.transfer_weights, len(TRANSFER_KINDS), "transfer_weights"),
        )
        routes = tuple(self.routes)
        if any(not isinstance(route, MutationRoute) for route in routes):
            raise ValueError("routes must contain MutationRoute values")
        if len({route.name for route in routes}) != len(routes):
            raise ValueError("mutation route names must be unique")
        for route in routes:
            for endpoint in ("donor", "receiver"):
                values = getattr(route, f"{endpoint}_layer_weights")
                if values and len(values) != num_layers:
                    raise ValueError("route layer weights must match the kernel layer count")
            if any(layer >= num_layers for layer in route.donor_layers + route.receiver_layers):
                raise ValueError("mutation route layer is outside the kernel layer range")
            if (
                route.donor_module == route.receiver_module
                and len(
                    {
                        (route.donor_module, donor, receiver)
                        for donor in route.endpoint_layers("donor")
                        for receiver in route.endpoint_layers("receiver")
                        if donor != receiver
                    }
                )
                == 0
            ):
                raise ValueError("same-module route needs at least one distinct donor/receiver pair")
        object.__setattr__(self, "routes", routes)

    @property
    def num_layers(self) -> int:
        return len(self.attn_compress)

    @classmethod
    def uniform(cls, num_layers: int) -> "MutationKernel":
        ones = (1.0,) * int(num_layers)
        return cls(ones, ones, ones, ones)

    @classmethod
    def uniform_routes(cls, num_layers: int) -> "MutationKernel":
        layers = tuple(range(int(num_layers)))
        ones = (1.0,) * int(num_layers)
        routes = tuple(
            MutationRoute(
                f"uniform_{donor}_to_{receiver}",
                donor,
                layers,
                receiver,
                layers,
            )
            for donor in ("attn", "mlp")
            for receiver in ("attn", "mlp")
        )
        return cls(ones, ones, ones, ones, routes=routes)

    def gene_weight(self, ref: GeneRef, delta: int) -> float:
        if delta not in (-1, 1):
            raise ValueError("kernel gene direction must be -1 or +1")
        direction = "compress" if delta > 0 else "decompress"
        return float(getattr(self, f"{ref.module}_{direction}")[ref.layer])

    def transfer_weight(self, kind: str) -> float:
        try:
            return float(self.transfer_weights[TRANSFER_KINDS.index(kind)])
        except ValueError as exc:
            raise ValueError(f"unknown transfer kind: {kind}") from exc

    def js_divergence_from_uniform(self) -> float:
        """Mean JS divergence over the distributions the sampler actually normalizes."""

        if self.routes:
            left = self._route_distribution()
            right = self.uniform_routes(self.num_layers)._route_distribution()
            keys = set(left) | set(right)
            return self._js_divergence(
                tuple(left.get(key, 0.0) for key in keys),
                tuple(right.get(key, 0.0) for key in keys),
            )
        groups = (
            self.attn_compress,
            self.attn_decompress,
            self.mlp_compress,
            self.mlp_decompress,
            self.transfer_weights,
        )
        return sum(
            self._js_divergence(values, (1.0,) * len(values)) for values in groups
        ) / len(groups)

    def total_variation_distance(self, other: "MutationKernel") -> float:
        """Mean TV distance across the five actual weighted sampling choices."""

        if self.num_layers != other.num_layers:
            raise ValueError("mutation kernels have different layer counts")
        if self.routes or other.routes:
            left = (
                self._route_distribution()
                if self.routes
                else self.uniform_routes(self.num_layers)._route_distribution()
            )
            right = (
                other._route_distribution()
                if other.routes
                else self.uniform_routes(self.num_layers)._route_distribution()
            )
            keys = set(left) | set(right)
            return 0.5 * sum(
                abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys
            )
        left_groups = (
            self.attn_compress,
            self.attn_decompress,
            self.mlp_compress,
            self.mlp_decompress,
            self.transfer_weights,
        )
        right_groups = (
            other.attn_compress,
            other.attn_decompress,
            other.mlp_compress,
            other.mlp_decompress,
            other.transfer_weights,
        )
        distances = []
        for left, right in zip(left_groups, right_groups):
            left_total, right_total = sum(left), sum(right)
            distances.append(
                0.5
                * sum(
                    abs(a / left_total - b / right_total)
                    for a, b in zip(left, right)
                )
            )
        return sum(distances) / len(distances)

    @staticmethod
    def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
        left_total, right_total = sum(left), sum(right)
        p = tuple(value / left_total for value in left)
        q = tuple(value / right_total for value in right)
        midpoint = tuple((a + b) / 2.0 for a, b in zip(p, q))

        def kl(values: Sequence[float], reference: Sequence[float]) -> float:
            return sum(a * math.log(a / b) for a, b in zip(values, reference) if a > 0.0)

        return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)

    def _route_distribution(self) -> Dict[Tuple[str, int, str, int], float]:
        routes = self.routes or self.uniform_routes(self.num_layers).routes
        route_total = sum(route.weight for route in routes)
        distribution: Dict[Tuple[str, int, str, int], float] = {}
        for route in routes:
            if route.donor_layer_weights:
                for donor in route.endpoint_layers("donor"):
                    receivers = [
                        receiver for receiver in route.endpoint_layers("receiver")
                        if route.donor_module != route.receiver_module or donor != receiver
                    ]
                    receiver_total = sum(route.endpoint_weight("receiver", receiver) for receiver in receivers)
                    for receiver in receivers:
                        pair = (route.donor_module, donor, route.receiver_module, receiver)
                        mass = (
                            route.weight / route_total
                            * route.endpoint_weight("donor", donor) / sum(route.donor_layer_weights)
                            * route.endpoint_weight("receiver", receiver) / receiver_total
                        )
                        distribution[pair] = distribution.get(pair, 0.0) + mass
                continue
            pairs = tuple(
                (route.donor_module, donor, route.receiver_module, receiver)
                for donor in route.donor_layers
                for receiver in route.receiver_layers
                if route.donor_module != route.receiver_module or donor != receiver
            )
            pair_mass = route.weight / route_total / len(pairs)
            for pair in pairs:
                distribution[pair] = distribution.get(pair, 0.0) + pair_mass
        return distribution

    def _flat_weights(self) -> Tuple[float, ...]:
        return (
            self.attn_compress
            + self.attn_decompress
            + self.mlp_compress
            + self.mlp_decompress
            + self.transfer_weights
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "hypothesis": self.hypothesis,
            "attn_compress_weights": list(self.attn_compress),
            "attn_decompress_weights": list(self.attn_decompress),
            "mlp_compress_weights": list(self.mlp_compress),
            "mlp_decompress_weights": list(self.mlp_decompress),
            "transfer_weights": dict(zip(TRANSFER_KINDS, self.transfer_weights)),
            "routes": [route.to_dict() for route in self.routes],
            "js_divergence_from_uniform": self.js_divergence_from_uniform(),
            "behavioral_total_variation_from_uniform": self.total_variation_distance(
                self.uniform_routes(self.num_layers) if self.routes else self.uniform(self.num_layers)
            ),
            "metadata": dict(self.metadata),
        }


def _transition(entry: Mapping[str, Any], num_layers: int):
    parent = entry.get("mutation_parent_policy")
    parent_reward = entry.get("mutation_parent_reward")
    child_reward = entry.get("reward")
    if not parent or parent_reward is None or child_reward is None or entry.get("error"):
        return None
    try:
        parent_policy = Policy.from_mapping(parent)
        child_policy = Policy.from_mapping(entry)
        delta = [
            child_policy.attn_levels[index] - parent_policy.attn_levels[index]
            for index in range(num_layers)
        ] + [
            child_policy.mlp_levels[index] - parent_policy.mlp_levels[index]
            for index in range(num_layers)
        ]
        reward_delta = float(child_reward) - float(parent_reward)
    except (KeyError, TypeError, ValueError):
        return None
    if len(parent_policy.attn_levels) != num_layers or not any(delta):
        return None
    if not math.isfinite(reward_delta):
        return None
    return delta, reward_delta, entry


def _scaled_weights(
    coefficients: torch.Tensor,
    *,
    temperature: float,
    min_weight: float,
    max_weight: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    centered = coefficients - coefficients.mean()
    scale = centered.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) <= 1e-12:
        ones = (1.0,) * int(coefficients.numel())
        return ones, ones
    z_scores = torch.clamp(centered / scale, -4.0, 4.0)
    compress = torch.exp(z_scores / temperature).clamp(min_weight, max_weight)
    decompress = torch.exp(-z_scores / temperature).clamp(min_weight, max_weight)
    return tuple(compress.tolist()), tuple(decompress.tolist())


def build_online_ridge_kernel(
    history: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    expensive_module: str,
    version: int,
    evidence_limit: int = 80,
    ridge: float = 1.0,
    temperature: float = 1.0,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    fit_target: str = "reward_delta",
    model: str = "linear",
    quadratic_ridge: float = 10.0,
    maxima: Sequence[int] | None = None,
    coefficient_mode: str = "learned",
    permutation_seed: int = 0,
) -> MutationKernel:
    """Fit gene directions; transfer-kind statistics always use reward deltas."""

    if fit_target not in {"reward_delta", "negative_kl_delta"}:
        raise ValueError(f"invalid online fit target: {fit_target}")
    if model not in {"linear", "quadratic"}:
        raise ValueError(f"invalid online model: {model}")
    if coefficient_mode not in {"learned", "permuted"}:
        raise ValueError(f"invalid coefficient mode: {coefficient_mode}")
    if coefficient_mode == "permuted" and (model != "linear" or fit_target != "negative_kl_delta"):
        raise ValueError("coefficient permutation requires the linear M0 KL target")
    if model == "quadratic":
        if fit_target != "negative_kl_delta":
            raise ValueError("M1 requires negative_kl_delta")
        if not math.isfinite(quadratic_ridge) or quadratic_ridge <= 0:
            raise ValueError("quadratic_ridge must be finite and positive")
        if (maxima is None or len(maxima) != 2*num_layers
                or any(not math.isfinite(v) or v < 0 for v in maxima)):
            raise ValueError("M1 requires finite nonnegative policy maxima for every gene")
    if ridge <= 0.0 or temperature <= 0.0:
        raise ValueError("ridge and temperature must be positive")
    if not 0.0 < min_weight <= max_weight:
        raise ValueError("kernel weight bounds are invalid")
    transitions = [
        item
        for item in (_transition(entry, num_layers) for entry in history)
        if item is not None
    ][-int(evidence_limit):]
    if len(transitions) < 2:
        uniform = MutationKernel.uniform(num_layers)
        return MutationKernel(
            uniform.attn_compress,
            uniform.attn_decompress,
            uniform.mlp_compress,
            uniform.mlp_decompress,
            source="online",
            version=version,
            hypothesis="insufficient transition evidence; uniform kernel",
            metadata={"sample_count": len(transitions), "fit_target": fit_target, "model": model},
        )

    design = torch.tensor([item[0] for item in transitions], dtype=torch.float64)
    targets = [item[1] for item in transitions]
    if fit_target == "negative_kl_delta":
        # Resolve against the full archive, not just the rolling fit window.
        parents = {row["history_id"]: row for row in history if "history_id" in row}
        targets = []
        for _, _, entry in transitions:
            parent = parents.get(entry.get("mutation_parent_history_id"))
            if parent is None:
                raise ValueError("M0 requires the parent history record to obtain raw KL")
            if Policy.from_mapping(parent) != Policy.from_mapping(entry["mutation_parent_policy"]):
                raise ValueError("M0 parent history policy does not match the recorded mutation parent")
            try:
                parent_kl, child_kl = float(parent["kl"]), float(entry["kl"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("M0 requires numeric raw parent and child KL") from exc
            if not math.isfinite(parent_kl) or not math.isfinite(child_kl):
                raise ValueError("M0 requires finite raw parent and child KL")
            targets.append(parent_kl - child_kl)
    rewards = torch.tensor(targets, dtype=torch.float64)
    rewards = rewards - rewards.mean()
    reward_scale = rewards.std(unbiased=False)
    if torch.isfinite(reward_scale) and float(reward_scale) > 1e-12:
        rewards = rewards / reward_scale
    identity = torch.eye(2 * num_layers, dtype=torch.float64)
    coefficients = torch.linalg.solve(
        design.T @ design + float(ridge) * identity,
        design.T @ rewards,
    )
    model_metadata = {"model": model}
    if coefficient_mode == "permuted":
        # Local RNG: never consume mutation/global RNG draws. Index lists map
        # destination layer -> source coefficient, independently within modules.
        rng_key = f"m0-permutation:{permutation_seed}:{version}"
        rng = random.Random(rng_key)
        permutations = {}
        learned = coefficients.tolist()
        permuted = []
        for module, offset in (("attn", 0), ("mlp", num_layers)):
            indices = list(range(num_layers))
            rng.shuffle(indices)
            permutations[module] = indices
            permuted.extend(learned[offset+i] for i in indices)
        coefficients = torch.tensor(permuted, dtype=torch.float64)
        model_metadata.update(
            coefficient_mode=coefficient_mode, permutation_seed=permutation_seed,
            permutation_rng_key=rng_key, coefficient_permutations=permutations,
            learned_coefficients=learned, compiled_coefficients=permuted,
        )
    if model == "quadratic":
        # Shadow M0 uses precisely the same rows and target preprocessing.
        shadow_beta = coefficients.tolist()
        parent_states = torch.tensor([
            list(Policy.from_mapping(item[2]["mutation_parent_policy"]).attn_levels)
            + list(Policy.from_mapping(item[2]["mutation_parent_policy"]).mlp_levels)
            for item in transitions
        ], dtype=torch.float64)
        # A fixed-at-zero gene contributes zero; avoid division by zero for it.
        levels = torch.tensor(maxima, dtype=torch.float64).clamp_min(1.0)
        quadratic = (2*parent_states*design + design.square()) / levels
        full_design = torch.cat((design, quadratic), dim=1)
        penalties = torch.tensor([ridge]*(2*num_layers) + [quadratic_ridge]*(2*num_layers),
                                 dtype=torch.float64)
        fitted = torch.linalg.solve(full_design.T @ full_design + torch.diag(penalties),
                                    full_design.T @ rewards)
        model_metadata.update(
            beta=fitted[:2*num_layers].tolist(), gamma=fitted[2*num_layers:].tolist(),
            maxima=list(maxima), shadow_beta=shadow_beta, quadratic_ridge=quadratic_ridge,
            target_mean=sum(targets)/len(targets),
            target_scale=float(reward_scale) if float(reward_scale) > 1e-12 else 1.0,
            min_weight=min_weight, max_weight=max_weight,
            weight_scope="unconditioned_template", weight_refresh="once_per_parent",
        )
    attn_compress, attn_decompress = _scaled_weights(
        coefficients[:num_layers],
        temperature=temperature,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    mlp_compress, mlp_decompress = _scaled_weights(
        coefficients[num_layers:],
        temperature=temperature,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    transfer_scores = {kind: [] for kind in TRANSFER_KINDS}
    for _, reward_delta, entry in transitions:
        edits = entry.get("mutation_edits") or []
        modules = {edit.get("module") for edit in edits}
        if modules == {"attn"}:
            kind = "same_attention"
        elif modules == {"mlp"}:
            kind = "same_mlp"
        elif modules == {"attn", "mlp"}:
            expensive_delta = sum(
                int(edit.get("delta", 0))
                for edit in edits
                if edit.get("module") == expensive_module
            )
            kind = (
                "compress_expensive_release_cheap"
                if expensive_delta > 0
                else "release_expensive_compress_cheap"
            )
        else:
            continue
        transfer_scores[kind].append(reward_delta)
    means = torch.tensor(
        [
            sum(transfer_scores[kind]) / len(transfer_scores[kind])
            if transfer_scores[kind]
            else 0.0
            for kind in TRANSFER_KINDS
        ],
        dtype=torch.float64,
    )
    transfer_weights, _ = _scaled_weights(
        means,
        temperature=temperature,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    if model == "quadratic":
        # These are placeholders, not representative/average M1 gene weights.
        # The candidate builder must condition this template on its actual parent.
        attn_compress = attn_decompress = mlp_compress = mlp_decompress = (1.0,)*num_layers
    return MutationKernel(
        attn_compress,
        attn_decompress,
        mlp_compress,
        mlp_decompress,
        transfer_weights=transfer_weights,
        source=("online_permuted" if coefficient_mode == "permuted" else
                "online_m1" if model == "quadratic" else "online"),
        version=version,
        hypothesis=("M0 module-wise coefficient permutation control; transfer learning unchanged"
                    if coefficient_mode == "permuted" else
                    "M1 quadratic KL-improvement model; gene weights compiled once per parent"
                    if model == "quadratic" else
                    "ridge direction fitted from observed mutation reward deltas"
                    if fit_target == "reward_delta" else
                    "M0 ridge gene direction fitted from parent KL - child KL; transfer kinds use reward"),
        metadata={
            **model_metadata,
            "fit_target": fit_target,
            "transfer_fit_target": "reward_delta",
            "fit_history_ids": [item[2].get("history_id") for item in transitions],
            "raw_fit_targets": targets,
            "sample_count": len(transitions),
            "ridge": ridge,
            "temperature": temperature,
            "evidence_limit": evidence_limit,
        },
    )


def condition_online_kernel(kernel: MutationKernel | None, parent: Policy) -> MutationKernel | None:
    """Compile M1 once at the starting parent; leave every other kernel untouched."""
    if kernel is None or kernel.metadata.get("model") != "quadratic" or "beta" not in kernel.metadata:
        return kernel
    m = kernel.metadata
    x = torch.tensor(parent.attn_levels + parent.mlp_levels, dtype=torch.float64)
    beta, gamma = (torch.tensor(m[key], dtype=torch.float64) for key in ("beta", "gamma"))
    levels = torch.tensor(m["maxima"], dtype=torch.float64).clamp_min(1.0)
    if x.numel() != beta.numel():
        raise ValueError("M1 parent/model dimensions differ")
    plus = beta + gamma*(2*x+1)/levels
    minus = -beta + gamma*(-2*x+1)/levels
    n = kernel.num_layers
    options = dict(temperature=m["temperature"], min_weight=m["min_weight"], max_weight=m["max_weight"])
    ac, mc = (_scaled_weights(scores, **options)[0] for scores in (plus[:n], plus[n:]))
    ad, md = (_scaled_weights(scores, **options)[0] for scores in (minus[:n], minus[n:]))
    return replace(kernel, attn_compress=ac, attn_decompress=ad, mlp_compress=mc, mlp_decompress=md,
                   metadata={**m, "weight_scope":"parent", "compiled_parent":parent.to_dict()})


def online_prediction_audit(kernel: MutationKernel, parent: Policy, child: Policy) -> Dict[str, Any]:
    """Frozen prospective predictions: never uses the child's evaluated quality."""
    m = kernel.metadata
    x = torch.tensor(parent.attn_levels + parent.mlp_levels, dtype=torch.float64)
    d = torch.tensor(child.attn_levels + child.mlp_levels, dtype=torch.float64) - x
    levels = torch.tensor(m["maxima"], dtype=torch.float64).clamp_min(1.0)
    linear = float(torch.dot(torch.tensor(m["beta"], dtype=torch.float64), d))
    quadratic = float(torch.dot(torch.tensor(m["gamma"], dtype=torch.float64), (2*x*d+d.square())/levels))
    shadow = float(torch.dot(torch.tensor(m["shadow_beta"], dtype=torch.float64), d))
    # Undo label preprocessing for diagnostic regression predictions only.
    # The target mean is an intercept here, not part of the action-weight scores.
    return dict(version=kernel.version, target="parent_kl_minus_child_kl",
                m1_standardized=linear+quadratic, linear_contribution=linear,
                quadratic_contribution=quadratic, shadow_m0_standardized=shadow,
                m1_predicted_improvement=(linear+quadratic)*m["target_scale"]+m["target_mean"],
                shadow_m0_predicted_improvement=shadow*m["target_scale"]+m["target_mean"])
