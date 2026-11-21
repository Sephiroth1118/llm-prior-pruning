"""Distances used for trust-region selection and diversity reporting."""

from __future__ import annotations

from .policy import Policy, PolicyBounds


def _validate_pair(left: Policy, right: Policy, bounds: PolicyBounds) -> None:
    bounds.validate(left)
    bounds.validate(right)


def normalized_l1_distance(left: Policy, right: Policy, bounds: PolicyBounds) -> float:
    """Mean normalized L1 distance over genes that can actually vary."""

    _validate_pair(left, right, bounds)
    contributions = []
    for ref in bounds.iter_genes():
        width = bounds.get(ref).width
        if width <= 0:
            continue
        contributions.append(abs(left.get(ref) - right.get(ref)) / float(width))
    if not contributions:
        return 0.0
    return sum(contributions) / len(contributions)


def hamming_distance(left: Policy, right: Policy, bounds: PolicyBounds) -> float:
    """Fraction of all attention/MLP genes whose levels are different."""

    _validate_pair(left, right, bounds)
    refs = list(bounds.iter_genes())
    if not refs:
        return 0.0
    return sum(left.get(ref) != right.get(ref) for ref in refs) / float(len(refs))
