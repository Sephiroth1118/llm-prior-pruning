"""Compact retrieval of evaluated policies and local edit evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .metrics import normalized_l1_distance
from .policy import GeneRef, Policy, PolicyBounds


@dataclass(frozen=True)
class EvaluatedPolicy:
    policy: Policy
    fitness: float
    kl: Optional[float] = None
    ppl: Optional[float] = None
    flops_reduction: Optional[float] = None
    source_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluatedPolicy":
        score = value.get("fitness", value.get("reward"))
        if score is None:
            raise ValueError("evaluated policy requires fitness or reward")
        return cls(
            policy=Policy.from_mapping(value),
            fitness=float(score),
            kl=float(value["kl"]) if value.get("kl") is not None else None,
            ppl=float(value["ppl"]) if value.get("ppl") is not None else None,
            flops_reduction=(
                float(value["flops_reduction"])
                if value.get("flops_reduction") is not None
                else None
            ),
            source_id=str(value["source_id"]) if value.get("source_id") else None,
        )


@dataclass(frozen=True)
class LocalEditEvidence:
    base_id: Optional[str]
    variant_id: Optional[str]
    gene: GeneRef
    base_level: int
    variant_level: int
    fitness_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_id": self.base_id,
            "variant_id": self.variant_id,
            "layer": self.gene.layer,
            "module": self.gene.module,
            "base_level": self.base_level,
            "variant_level": self.variant_level,
            "fitness_delta": self.fitness_delta,
            "confidence": "high",
            "basis": "policies differ at exactly one gene",
        }


class ExperienceArchive:
    """Deduplicated evaluated-policy archive used to build LLM context."""

    def __init__(
        self, bounds: PolicyBounds, entries: Iterable[EvaluatedPolicy] = ()
    ) -> None:
        self.bounds = bounds
        by_fingerprint: Dict[Tuple[int, ...], EvaluatedPolicy] = {}
        for entry in entries:
            bounds.validate(entry.policy)
            existing = by_fingerprint.get(entry.policy.fingerprint)
            if existing is None or entry.fitness > existing.fitness:
                by_fingerprint[entry.policy.fingerprint] = entry
        self.entries = tuple(by_fingerprint.values())

    def nearest_to_parents(
        self,
        parent_a: Policy,
        parent_b: Policy,
        *,
        limit: int = 6,
        target_flops_reduction: Optional[float] = None,
        flops_tolerance: Optional[float] = None,
    ) -> Tuple[EvaluatedPolicy, ...]:
        candidates = list(self.entries)
        if target_flops_reduction is not None and flops_tolerance is not None:
            candidates = [
                entry
                for entry in candidates
                if entry.flops_reduction is not None
                and abs(entry.flops_reduction - target_flops_reduction) <= flops_tolerance
            ]
        ranked = sorted(
            candidates,
            key=lambda entry: (
                min(
                    normalized_l1_distance(entry.policy, parent_a, self.bounds),
                    normalized_l1_distance(entry.policy, parent_b, self.bounds),
                ),
                -entry.fitness,
                entry.policy.fingerprint,
            ),
        )
        return tuple(ranked[: max(0, limit)])

    def single_gene_evidence(self, *, limit: int = 8) -> Tuple[LocalEditEvidence, ...]:
        evidence: List[LocalEditEvidence] = []
        for index, left in enumerate(self.entries):
            for right in self.entries[index + 1 :]:
                differences = [
                    ref
                    for ref in self.bounds.iter_genes()
                    if left.policy.get(ref) != right.policy.get(ref)
                ]
                if len(differences) != 1:
                    continue
                ref = differences[0]
                if left.policy.get(ref) <= right.policy.get(ref):
                    base, variant = left, right
                else:
                    base, variant = right, left
                evidence.append(
                    LocalEditEvidence(
                        base_id=base.source_id,
                        variant_id=variant.source_id,
                        gene=ref,
                        base_level=base.policy.get(ref),
                        variant_level=variant.policy.get(ref),
                        fitness_delta=variant.fitness - base.fitness,
                    )
                )
        evidence.sort(
            key=lambda item: (
                -abs(item.fitness_delta),
                item.gene.layer,
                item.gene.module,
                item.base_level,
            )
        )
        return tuple(evidence[: max(0, limit)])

    def build_prompt_context(
        self,
        parent_a: Policy,
        parent_b: Policy,
        *,
        neighbor_limit: int = 6,
        evidence_limit: int = 8,
        target_flops_reduction: Optional[float] = None,
        flops_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        neighbors = self.nearest_to_parents(
            parent_a,
            parent_b,
            limit=neighbor_limit,
            target_flops_reduction=target_flops_reduction,
            flops_tolerance=flops_tolerance,
        )
        return {
            "nearby_evaluations": [
                {
                    "source_id": entry.source_id,
                    "policy": entry.policy.to_dict(),
                    "fitness": entry.fitness,
                    "kl": entry.kl,
                    "ppl": entry.ppl,
                    "flops_reduction": entry.flops_reduction,
                }
                for entry in neighbors
            ],
            "high_confidence_local_edits": [
                item.to_dict() for item in self.single_gene_evidence(limit=evidence_limit)
            ],
        }
