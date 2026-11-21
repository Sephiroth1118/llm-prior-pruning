"""Budget-feasible structural anchors, short descendant-only refinement, then EA."""
import math
import random

import numpy as np

WEIGHTS = (0., .25, .5, 1., 2., 4.)


class StructureDecoder:
    def __init__(self, remaining_costs, dense_total, band=(.500, .502)):
        costs = np.asarray(remaining_costs, float)
        if costs.ndim != 2 or costs.shape[0] % 2 or np.any(np.diff(costs, axis=1) > 1e-8):
            raise ValueError('Expected monotone per-gene remaining-cost tables')
        self.removed = (costs[:, :1] - costs) / dense_total
        self.fractions = (costs[:, :1] - costs) / costs[:, :1]
        self.mass = costs[:, 0] / dense_total
        self.layers = len(costs) // 2
        self.cells = np.tile(np.minimum(np.arange(self.layers) * 4 // self.layers, 3), 2)
        self.cells[self.layers:] += 4
        self.band, self.target = band, sum(band) / 2

    def budget(self, policy):
        return float(self.removed[np.arange(len(policy)), policy].sum())

    def feasible(self, policy):
        return (len(policy) == len(self.mass) and
                all(isinstance(v, (int, np.integer)) and 0 <= v < self.removed.shape[1] for v in policy) and
                self.band[0] <= self.budget(policy) <= self.band[1])

    def validate(self, template):
        weights = template.get('weights')
        if not isinstance(weights, list) or len(weights) != 8:
            raise ValueError('A structure needs eight cell weights')
        if any(type(w) not in (int, float) or w not in WEIGHTS for w in weights):
            raise ValueError('Weights must belong to the shared finite alphabet')
        w = np.asarray(weights)[self.cells]
        if self.removed[w > 0, -1].sum() < self.target:
            raise ValueError('Positive-weight cells cannot supply the target removal budget')
        return w

    def random_template(self, rng):
        for _ in range(1000):
            item = {'name': 'random', 'weights': [rng.choice(WEIGHTS) for _ in range(8)],
                    'rationale': 'Independent uniform draws from the shared weight alphabet'}
            try:
                self.validate(item)
                return item
            except ValueError:
                pass
        raise RuntimeError('Unable to sample a feasible structural template')

    def decode(self, template, rng):
        """Scale relative removal FRACTIONS, stochastically round, then repair discretization.

        Zero weights protect genes at the anchor only. All later mutations are unrestricted
        except for the common absolute budget band. No KL or outcome is read here.
        """
        w = self.validate(template)
        lo, hi = 0., 1.
        maximum = self.fractions[:, -1]
        while np.dot(np.minimum(hi * w, maximum), self.mass) < self.target:
            hi *= 2
        for _ in range(64):
            mid = (lo + hi) / 2
            if np.dot(np.minimum(mid * w, maximum), self.mass) < self.target:
                lo = mid
            else:
                hi = mid
        desired = np.minimum(hi * w, maximum)
        policy = []
        for row, value in zip(self.fractions, desired):
            upper = min(int(np.searchsorted(row, value, side='left')), len(row) - 1)
            lower = max(0, upper - 1)
            p = (value - row[lower]) / (row[upper] - row[lower]) if row[upper] > row[lower] else 0.
            policy.append(upper if rng.random() < p else lower)
        before = self.budget(policy)
        edits = 0
        for _ in range(self.removed.size):
            current = self.budget(policy)
            if self.band[0] <= current <= self.band[1]:
                actual = self.fractions[np.arange(len(policy)), policy]
                return tuple(policy), {'flops_reduction': current, 'rounding_budget': before,
                                       'repair_edits': edits, 'profile_rmse': float(np.sqrt(np.mean((actual-desired)**2))),
                                       'desired_cell_fractions': [float(desired[self.cells == k].mean()) for k in range(8)],
                                       'actual_cell_fractions': [float(actual[self.cells == k].mean()) for k in range(8)]}
            direction = 1 if current < self.target else -1
            choices = []
            for i, old in enumerate(policy):
                new = old + direction
                if not w[i] or not 0 <= new < self.removed.shape[1]:
                    continue
                cost = current + self.removed[i, new] - self.removed[i, old]
                if abs(cost-self.target) >= abs(current-self.target) - 1e-15:
                    continue
                # Choose the least distortion per unit budget moved, with seeded tie breaking.
                distortion = ((self.fractions[i, new]-desired[i])**2 -
                              (self.fractions[i, old]-desired[i])**2)
                choices.append((distortion / max(abs(cost-current), 1e-15), rng.random(), i, new))
            if not choices:
                # Unequal discrete costs can require moving two genes in opposite directions.
                # A one-coordinate greedy repair can stall despite a nearby feasible policy.
                moves = []
                for i, old in enumerate(policy):
                    if not w[i]:
                        continue
                    for new in range(self.removed.shape[1]):
                        if new == old:
                            continue
                        delta = self.removed[i,new]-self.removed[i,old]
                        dist = (self.fractions[i,new]-desired[i])**2-(self.fractions[i,old]-desired[i])**2
                        moves.append((i,new,delta,dist))
                if not moves:
                    break
                m = np.asarray(moves)
                pair_budget = current+m[:,2,None]+m[None,:,2]
                feasible = ((pair_budget >= self.band[0]) & (pair_budget <= self.band[1]) &
                            (m[:,0,None] != m[None,:,0]))
                ii,jj = np.nonzero(feasible)
                if not len(ii):
                    break
                costs = m[ii,3]+m[jj,3]
                candidates = np.flatnonzero(np.isclose(costs,costs.min(),rtol=0,atol=1e-14))
                pick = int(rng.choice(candidates.tolist()))
                for index in (ii[pick],jj[pick]):
                    i,new,_,_ = moves[index]; policy[i] = new
                edits += 2
                continue
            _, _, i, new = min(choices)
            policy[i] = new
            edits += 1
        raise ValueError('Discrete rounding could not reach the absolute budget band')

    def describe(self, policy):
        frac = self.fractions[np.arange(len(policy)), policy]
        removed = self.removed[np.arange(len(policy)), policy]
        return {'cell_removed_fractions': [float(frac[self.cells == k].mean()) for k in range(8)],
                'cell_budget_shares': [float(removed[self.cells == k].sum()/removed.sum()) for k in range(8)]}


class SerialSearch:
    def __init__(self, evaluate, mutate, decoder, propose, seed, *, mu=8, batch=8, refine_budget=16):
        self.evaluate, self.mutate, self.decoder, self.propose = evaluate, mutate, decoder, propose
        self.seed, self.mu, self.batch, self.refine_budget = seed, mu, batch, refine_budget
        self.rng = random.Random(seed)
        self.entries, self.population, self.branches, self.calls = [], [], [], []
        self.seen = set()
        self.phase = 'initial'

    @staticmethod
    def rank(row):
        return row['kl'], tuple(row['policy'])

    def select(self, rows):
        return sorted({tuple(r['policy']): r for r in rows}.values(), key=self.rank)[:self.mu]

    def record(self, policies, parents, phase, branch=None):
        if len(set(policies)) != len(policies) or any(p in self.seen or not self.decoder.feasible(p) for p in policies):
            raise ValueError('Duplicate or infeasible evaluation')
        results = self.evaluate(policies)
        if len(results) != len(policies):
            raise ValueError('Evaluation batch length mismatch')
        rows = []
        for policy, parent, result in zip(policies, parents, results):
            kl = float(result['kl'])
            if result.get('error') or not math.isfinite(kl) or kl < 0 or abs(result['reward']+kl) > 1e-12:
                raise ValueError('Invalid raw-KL result')
            if abs(result['flops_reduction']-self.decoder.budget(policy)) > 1e-12:
                raise ValueError('Evaluator and decoder FLOPs disagree')
            row = {'id': len(self.entries), 'evaluations': len(self.entries)+1, 'policy': list(policy),
                   **result, 'parent': parent, 'phase': phase, 'branch': branch,
                   'origin': (self.entries[parent]['origin'] if parent is not None else
                              f'structure{branch}' if branch is not None else 'initial')}
            self.entries.append(row); rows.append(row); self.seen.add(policy)
        return rows

    def offspring(self, population, count, rng):
        policies, parents, seen = [], [], set(self.seen)
        for slot in range(count):
            # Same top-eight round-robin parent selection in both structural arms.
            parent = population[slot % len(population)]
            for _ in range(256):
                child = tuple(self.mutate(tuple(parent['policy']), rng))
                if child not in seen and self.decoder.feasible(child):
                    policies.append(child); parents.append(parent['id']); seen.add(child)
                    break
            else:
                raise RuntimeError('No unique feasible mutation within 256 attempts')
        return policies, parents

    def run(self, initial, budget=320, checkpoint=lambda state: None):
        if budget < len(initial)+4*self.refine_budget:
            raise ValueError('Budget does not cover four structural trials')
        self.population = self.select(self.record(initial, [None]*len(initial), 'initial'))
        checkpoint(self)
        for round_id in range(2):
            self.phase = f'structure_round{round_id+1}'
            templates, metadata = self.propose(round_id, self)
            if len(templates) != 2:
                raise ValueError('Exactly two structures per round')
            self.calls.append(metadata)
            for slot, template in enumerate(templates):
                branch = round_id*2+slot
                rng = random.Random(self.seed*10000+100+branch)
                for attempt in range(32):
                    anchor, decoding = self.decoder.decode(template, rng)
                    if anchor not in self.seen:
                        break
                else:
                    raise RuntimeError('Template failed to produce a new anchor')
                before = min(r['kl'] for r in self.entries)
                local = self.record([anchor], [None], 'anchor', branch)
                start = local[0]['id']
                checkpoint(self)
                while len(self.entries)-start < self.refine_budget:
                    n = min(self.batch, self.refine_budget-(len(self.entries)-start))
                    policies, parents = self.offspring(local, n, rng)
                    children = self.record(policies, parents, 'refine', branch)
                    local = self.select(local+children)
                    checkpoint(self)
                champion = min(local, key=self.rank)
                self.branches.append({'branch': branch, 'round': round_id+1, 'template': template,
                                      'decoding': decoding, 'anchor_id': start, 'best_id': champion['id'],
                                      'anchor_kl': self.entries[start]['kl'], 'refined_kl': champion['kl'],
                                      'best_before': before, 'improves_global': champion['kl'] < before,
                                      'evaluations': self.refine_budget,
                                      'refined_structure': self.decoder.describe(champion['policy'])})
                # No global elite can parent the short refinement. Merge only after its full allowance.
                self.population = self.select(self.population+local)
                checkpoint(self)
        self.phase = 'ea'
        while len(self.entries) < budget:
            policies, parents = self.offspring(self.population, min(self.batch, budget-len(self.entries)), self.rng)
            self.population = self.select(self.population+self.record(policies, parents, 'ea'))
            checkpoint(self)
        self.phase = 'complete'
        checkpoint(self)
        return self

    def summary(self):
        champion = min(self.entries, key=self.rank)
        curve = np.minimum.accumulate([r['kl'] for r in self.entries])
        return {'evaluations': len(self.entries), 'best_kl': champion['kl'], 'best_entry': champion['id'],
                'best_origin': champion['origin'], 'auc_mean': float(curve[16:].mean()) if len(curve)>16 else None,
                'checkpoints': {str(n): float(curve[n-1]) for n in (16,48,80,160,320,500,1000) if len(curve)>=n}}
