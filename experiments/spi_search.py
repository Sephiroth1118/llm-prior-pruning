"""Configurable extension of SerialSearch (evoprune/serial_search.py).

With rounds=[{'source':'llm','n':2}]*2 (or random), refine_budget=16 and a fixed EA strength of
0.0075, this reproduces SerialSearch.run exactly: the same per-branch RNG streams, the same
round-robin parents, the same selection and the same EA RNG (random.Random(seed)).

New knobs
  rounds            any number of structure rounds, any number of structures per round
  schedule          EA-phase mutation strength: fixed / anneal (geometric decay) / adapt (1/5 rule)
  refine_strength   strength used inside the short structural refinements (default as original)
"""
import math
import random

import common  # noqa: F401  (puts the repository root on sys.path)
from evoprune import serial_search as core
StructureDecoder = core.StructureDecoder
SerialSearch = core.SerialSearch


class StrengthMutator:
    """The reference mutate() with a mutable target strength (budget_tolerance and kernel unchanged)."""

    def __init__(self, operator, policy_cls, bounds, decoder, layers, strength=0.0075, tolerance=0.001):
        self.operator, self.policy_cls, self.bounds, self.decoder = operator, policy_cls, bounds, decoder
        self.layers, self.strength, self.tolerance = layers, strength, tolerance
        self.kernel = None                 # None = the uniform mutation; a MutationKernel = GP-guided
        self.edits_by_child = {}           # child tuple -> list of {layer, module, delta} (GP training rows)

    def cost(self, policy):
        return self.decoder.budget(policy.fingerprint)

    def __call__(self, policy, rng):
        parent = self.policy_cls.from_arrays(list(policy[:self.layers]), list(policy[self.layers:]))
        result = self.operator(parent, self.bounds, self.cost, rng, target_strength=self.strength,
                               budget_tolerance=self.tolerance, kernel=self.kernel)
        child = tuple(result.child.fingerprint)
        self.edits_by_child[child] = [{"layer": int(e.layer), "module": str(e.module), "delta": int(e.delta)} for e in result.edits]
        return child


class FixedSchedule:
    def __init__(self, start=0.0075, **_):
        self.value = start

    def strength(self, evaluations):
        return self.value

    def update(self, successes, children):
        pass

    def describe(self):
        return {"type": "fixed", "start": self.value}


class AnnealSchedule:
    """Geometric decay from `start` at `begin` evaluations to `end` at `finish` evaluations."""

    def __init__(self, start=0.0075, end=0.0026, begin=80, finish=1000, **_):
        self.start, self.end, self.begin, self.finish = start, end, begin, finish

    def strength(self, evaluations):
        t = min(1.0, max(0.0, (evaluations - self.begin) / max(1, self.finish - self.begin)))
        return self.start * (self.end / self.start) ** t

    def update(self, successes, children):
        pass

    def describe(self):
        return {"type": "anneal", "start": self.start, "end": self.end, "begin": self.begin, "finish": self.finish}


class AdaptSchedule:
    """Rechenberg-style 1/5 rule per generation on the fraction of children better than their parent."""

    def __init__(self, start=0.0075, lo=0.0026, hi=0.015, target=0.2, up=1.25, down=0.85, **_):
        self.value, self.lo, self.hi, self.target, self.up, self.down = start, lo, hi, target, up, down
        self.start = start

    def strength(self, evaluations):
        return self.value

    def update(self, successes, children):
        if not children:
            return
        rate = successes / children
        if rate > self.target:
            self.value = min(self.hi, self.value * self.up)
        elif rate < self.target:
            self.value = max(self.lo, self.value * self.down)

    def describe(self):
        return {"type": "adapt", "start": self.start, "lo": self.lo, "hi": self.hi, "target": self.target,
                "up": self.up, "down": self.down}


SCHEDULES = {"fixed": FixedSchedule, "anneal": AnnealSchedule, "adapt": AdaptSchedule}


def make_schedule(spec):
    spec = dict(spec or {"schedule": "fixed"})
    kind = spec.pop("schedule", "fixed")
    return SCHEDULES[kind](**spec)


class ResearchSearch(SerialSearch):
    def __init__(self, evaluate, mutator, decoder, propose, seed, *, rounds, schedule, mu=8, batch=8,
                 refine_budget=16, refine_strength=0.0075):
        super().__init__(evaluate, mutator, decoder, propose, seed, mu=mu, batch=batch, refine_budget=refine_budget)
        self.mutator, self.rounds, self.schedule, self.refine_strength = mutator, list(rounds), schedule, refine_strength
        self.ea_log, self.fallback_log = [], []

    def offspring(self, population, count, rng):
        try:
            return super().offspring(population, count, rng)
        except RuntimeError as exc:
            # Only reached where SerialSearch would crash (no unique feasible
            # child in 256 tries, seen once at the adaptive upper strength): retry at the default strength.
            saved = self.mutator.strength
            self.fallback_log.append({"evaluations": len(self.entries), "kind": "offspring", "error": str(exc),
                                      "strength": saved})
            self.mutator.strength = min(saved, 0.0075)   # the default step; a large step is what fails here
            try:
                return super().offspring(population, count, rng)
            finally:
                self.mutator.strength = saved

    def before_generation(self):
        """Hook called before every EA generation (GPSearch refits its kernel here)."""

    def refine_branch(self, round_id, branch, template, checkpoint):
        """Verbatim the inner loop of SerialSearch.run (anchor + descendant-only refinement)."""
        self.mutator.strength = self.refine_strength
        self.mutator.kernel = None         # structural refinement is always the uniform operator
        rng = random.Random(self.seed * 10000 + 100 + branch)
        for attempt in range(32):
            try:
                anchor, decoding = self.decoder.decode(template, rng)
            except ValueError as exc:
                # Only reached where SerialSearch would crash: an undecodable
                # template becomes an annotated random fallback (same as an invalid LLM proposal).
                self.fallback_log.append({"branch": branch, "kind": "decode", "error": str(exc),
                                          "template": template})
                template = {**self.decoder.random_template(rng), "fallback": True, "decode_error": str(exc)}
                continue
            if anchor not in self.seen:
                break
        else:
            raise RuntimeError("Template failed to produce a new anchor")
        before = min(r["kl"] for r in self.entries)
        local = self.record([anchor], [None], "anchor", branch)
        start = local[0]["id"]
        checkpoint(self)
        while len(self.entries) - start < self.refine_budget:
            n = min(self.batch, self.refine_budget - (len(self.entries) - start))
            policies, parents = self.offspring(local, n, rng)
            children = self.record(policies, parents, "refine", branch)
            local = self.select(local + children)
            checkpoint(self)
        champion = min(local, key=self.rank)
        self.branches.append({"branch": branch, "round": round_id + 1, "template": template,
                              "decoding": decoding, "anchor_id": start, "best_id": champion["id"],
                              "anchor_kl": self.entries[start]["kl"], "refined_kl": champion["kl"],
                              "best_before": before, "improves_global": champion["kl"] < before,
                              "evaluations": self.refine_budget,
                              "refined_structure": self.decoder.describe(champion["policy"])})
        self.population = self.select(self.population + local)
        checkpoint(self)

    def run(self, initial, budget=1000, checkpoint=lambda state: None):
        structural = sum(spec["n"] for spec in self.rounds) * self.refine_budget
        if budget < len(initial) + structural:
            raise ValueError("Budget does not cover the structural trials")
        self.population = self.select(self.record(initial, [None] * len(initial), "initial"))
        checkpoint(self)
        branch = 0
        for round_id, spec in enumerate(self.rounds):
            self.phase = f"structure_round{round_id + 1}"
            templates, metadata = self.propose(round_id, self, spec)
            if len(templates) != spec["n"]:
                raise ValueError(f"Round {round_id + 1} needs exactly {spec['n']} structures")
            self.calls.append(metadata)
            for template in templates:
                self.refine_branch(round_id, branch, template, checkpoint)
                branch += 1
        self.phase = "ea"
        while len(self.entries) < budget:
            self.before_generation()
            self.mutator.strength = self.schedule.strength(len(self.entries))
            policies, parents = self.offspring(self.population, min(self.batch, budget - len(self.entries)), self.rng)
            rows = self.record(policies, parents, "ea")
            successes = sum(r["kl"] < self.entries[r["parent"]]["kl"] for r in rows)
            self.ea_log.append({"evaluations": len(self.entries), "strength": self.mutator.strength,
                                "successes": int(successes), "children": len(rows)})
            self.schedule.update(successes, len(rows))
            self.population = self.select(self.population + rows)
            checkpoint(self)
        self.phase = "complete"
        checkpoint(self)
        return self


class GPSearch(ResearchSearch):
    """ResearchSearch whose EA phase uses GP-guided mutation (gp_guide.GPGuide), refitted on a schedule.

    Identical to ResearchSearch until the first fit (uniform operator); children carry their edit
    lists so the guide can label transfer kinds exactly as the original implementation does.
    """

    def __init__(self, *args, guide=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.guide, self.gp_log = guide, []

    def record(self, policies, parents, phase, branch=None):
        rows = super().record(policies, parents, phase, branch)
        for row in rows:
            if row.get("parent") is not None:
                row["mutation_edits"] = self.mutator.edits_by_child.get(tuple(row["policy"]))
        return rows

    def before_generation(self):
        if self.guide is None:
            return
        n = len(self.entries)
        if self.guide.due(n):
            kernel = self.guide.fit(self.entries, n)
            if kernel is not None:
                self.mutator.kernel = kernel
            self.gp_log.append(dict(self.guide.log[-1], active=self.mutator.kernel is not None))


class AlphabetDecoder(StructureDecoder):
    """StructureDecoder with a configurable weight alphabet (the base class fixes WEIGHTS at module level)."""

    def __init__(self, remaining_costs, dense_total, band=(.500, .502), weights=core.WEIGHTS):
        super().__init__(remaining_costs, dense_total, band)
        self.weights = tuple(float(w) for w in weights)
        if len(set(self.weights)) != len(self.weights) or 0.0 not in self.weights or min(self.weights) < 0:
            raise ValueError("alphabet must be distinct, non-negative and contain 0")

    def validate(self, template):
        import numpy as np
        weights = template.get("weights")
        if not isinstance(weights, list) or len(weights) != 8:
            raise ValueError("A structure needs eight cell weights")
        if any(type(w) not in (int, float) or float(w) not in self.weights for w in weights):
            raise ValueError("Weights must belong to the shared finite alphabet")
        w = np.asarray(weights, dtype=float)[self.cells]
        if self.removed[w > 0, -1].sum() < self.target:
            raise ValueError("Positive-weight cells cannot supply the target removal budget")
        return w

    def random_template(self, rng):
        for _ in range(1000):
            item = {"name": "random", "weights": [rng.choice(self.weights) for _ in range(8)],
                    "rationale": "Independent uniform draws from the shared weight alphabet"}
            try:
                self.validate(item)
                return item
            except ValueError:
                pass
        raise RuntimeError("Unable to sample a feasible structural template")
