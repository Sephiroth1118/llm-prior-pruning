"""Canonical immutable representation of an pruning policy."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Iterator, Literal, Mapping, Sequence, Tuple


ModuleKind = Literal["attn", "mlp"]


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True, order=True)
class GeneRef:
    """A single attention or MLP decision at one transformer layer."""

    layer: int
    module: ModuleKind

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("gene layer must be non-negative")
        if self.module not in ("attn", "mlp"):
            raise ValueError("gene module must be 'attn' or 'mlp'")


@dataclass(frozen=True)
class GeneBounds:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(
                f"invalid gene bounds: minimum={self.minimum} exceeds maximum={self.maximum}"
            )

    @property
    def width(self) -> int:
        return self.maximum - self.minimum

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class Policy:
    """Attention and MLP compression levels for every transformer layer."""

    attn_levels: Tuple[int, ...]
    mlp_levels: Tuple[int, ...]

    def __post_init__(self) -> None:
        attn = tuple(_require_integer(value, "attn level") for value in self.attn_levels)
        mlp = tuple(_require_integer(value, "mlp level") for value in self.mlp_levels)
        if not attn:
            raise ValueError("a policy must contain at least one layer")
        if len(attn) != len(mlp):
            raise ValueError("attn_levels and mlp_levels must have the same length")
        object.__setattr__(self, "attn_levels", attn)
        object.__setattr__(self, "mlp_levels", mlp)

    @classmethod
    def from_arrays(cls, attn_levels: Sequence[int], mlp_levels: Sequence[int]) -> "Policy":
        return cls(tuple(attn_levels), tuple(mlp_levels))

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "Policy":
        nested = obj.get("policy")
        if isinstance(nested, Mapping):
            obj = nested

        if "attn_levels" in obj and "mlp_levels" in obj:
            return cls.from_arrays(obj["attn_levels"], obj["mlp_levels"])

        layers = obj.get("layers")
        if not isinstance(layers, Mapping):
            raise ValueError("policy mapping must contain arrays or a 'layers' mapping")
        num_layers = _require_integer(obj.get("num_layers", len(layers)), "num_layers")
        try:
            attn = [layers[str(index)]["attn_level"] for index in range(num_layers)]
            mlp = [layers[str(index)]["mlp_level"] for index in range(num_layers)]
        except (KeyError, TypeError) as exc:
            raise ValueError("policy layers are incomplete or malformed") from exc
        return cls.from_arrays(attn, mlp)

    @property
    def num_layers(self) -> int:
        return len(self.attn_levels)

    @property
    def fingerprint(self) -> Tuple[int, ...]:
        return self.attn_levels + self.mlp_levels

    def get(self, ref: GeneRef) -> int:
        values = self.attn_levels if ref.module == "attn" else self.mlp_levels
        if ref.layer >= len(values):
            raise IndexError(f"layer {ref.layer} is outside a {self.num_layers}-layer policy")
        return values[ref.layer]

    def replace(self, ref: GeneRef, value: int) -> "Policy":
        value = _require_integer(value, "replacement level")
        attn = list(self.attn_levels)
        mlp = list(self.mlp_levels)
        values = attn if ref.module == "attn" else mlp
        if ref.layer >= len(values):
            raise IndexError(f"layer {ref.layer} is outside a {self.num_layers}-layer policy")
        values[ref.layer] = value
        return Policy.from_arrays(attn, mlp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attn_levels": list(self.attn_levels),
            "mlp_levels": list(self.mlp_levels),
        }

    def to_db_policy(self) -> Dict[str, Any]:
        return {
            "policy_type": "db_level_policy",
            "num_layers": self.num_layers,
            "layers": {
                str(index): {
                    "attn_level": self.attn_levels[index],
                    "mlp_level": self.mlp_levels[index],
                }
                for index in range(self.num_layers)
            },
        }


@dataclass(frozen=True)
class PolicyBounds:
    attn: Tuple[GeneBounds, ...]
    mlp: Tuple[GeneBounds, ...]

    def __post_init__(self) -> None:
        if not self.attn:
            raise ValueError("policy bounds must contain at least one layer")
        if len(self.attn) != len(self.mlp):
            raise ValueError("attention and MLP bounds must have the same length")

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "PolicyBounds":
        num_layers = _require_integer(obj.get("num_layers"), "num_layers")
        layers = obj.get("layers")
        if not isinstance(layers, Mapping):
            raise ValueError("policy bounds must contain a 'layers' mapping")

        attn = []
        mlp = []
        for index in range(num_layers):
            try:
                layer = layers[str(index)]
                attn.append(
                    GeneBounds(
                        _require_integer(layer["attn_min"], "attn_min"),
                        _require_integer(layer["attn_max"], "attn_max"),
                    )
                )
                mlp.append(
                    GeneBounds(
                        _require_integer(layer["mlp_min"], "mlp_min"),
                        _require_integer(layer["mlp_max"], "mlp_max"),
                    )
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"missing or malformed bounds for layer {index}") from exc
        return cls(tuple(attn), tuple(mlp))

    @property
    def num_layers(self) -> int:
        return len(self.attn)

    def get(self, ref: GeneRef) -> GeneBounds:
        values = self.attn if ref.module == "attn" else self.mlp
        if ref.layer >= len(values):
            raise IndexError(f"layer {ref.layer} is outside {self.num_layers}-layer bounds")
        return values[ref.layer]

    def iter_genes(self, module: ModuleKind | None = None) -> Iterator[GeneRef]:
        modules = (module,) if module is not None else ("attn", "mlp")
        for current_module in modules:
            for layer in range(self.num_layers):
                yield GeneRef(layer=layer, module=current_module)

    def validate(self, policy: Policy) -> None:
        if policy.num_layers != self.num_layers:
            raise ValueError(
                f"policy has {policy.num_layers} layers but bounds require {self.num_layers}"
            )
        violations = []
        for ref in self.iter_genes():
            value = policy.get(ref)
            gene_bounds = self.get(ref)
            if not gene_bounds.contains(value):
                violations.append(
                    f"{ref.module}[{ref.layer}]={value} is outside "
                    f"{gene_bounds.minimum}..{gene_bounds.maximum}"
                )
        if violations:
            raise ValueError("invalid policy: " + "; ".join(violations))

    def to_compact_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "layers": {
                str(index): {
                    "attn_min": self.attn[index].minimum,
                    "attn_max": self.attn[index].maximum,
                    "mlp_min": self.mlp[index].minimum,
                    "mlp_max": self.mlp[index].maximum,
                }
                for index in range(self.num_layers)
            },
        }
