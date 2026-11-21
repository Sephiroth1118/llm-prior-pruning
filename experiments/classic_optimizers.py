#!/usr/bin/env python3
"""Classical black-box optimizers as baselines for the allocation search .

All optimizers share: the same 16 initial policies as the other arms, the same evaluator and cache, the
hard FLOPs band of the decoder, and a generic greedy repair that projects any proposal onto the band.
Proposals are integer vectors (2L genes, K levels each); continuous optimizers (DE, PSO, GP-BO's pool)
round before repair. Duplicates of already evaluated policies are perturbed and repaired again.

Optimizers: random (fresh feasible draws), tpe (hyperopt-style univariate Parzen estimators over levels),
gpbo (Matern GP + expected improvement over a sampled candidate pool, batch by Kriging believer),
de (DE/rand/1/bin), pso (constriction PSO), ga (tournament + uniform crossover + per-gene mutation, mu+lambda),
cmaes ((mu/mu_w, lambda)-CMA-ES with default parameters).
No third-party optimizer package is available in the environment, so everything is implemented here.
"""
import math
import random

import numpy as np


class Space:
    """Band-constrained integer space with greedy repair; wraps the StructureDecoder tables."""

    def __init__(self, decoder, rng, random_feasible=None):
        self.d = decoder
        self.removed = np.asarray(decoder.removed, float)
        self.n, self.K = self.removed.shape
        self.lo, self.hi = decoder.band
        self.target = decoder.target
        self.rng = rng
        self._random_feasible = random_feasible
        self.repair_log = {"single": 0, "pair": 0, "failed": 0, "fallback_random": 0, "duplicates": 0}

    def budget(self, p):
        return float(self.removed[np.arange(self.n), list(p)].sum())

    def feasible(self, p):
        return self.lo <= self.budget(p) <= self.hi

    def clip_round(self, x):
        return [int(min(self.K - 1, max(0, round(float(v))))) for v in x]

    def repair(self, p, max_steps=600):
        p = [int(v) for v in p]
        for _ in range(max_steps):
            b = self.budget(p)
            if self.lo <= b <= self.hi:
                return tuple(int(v) for v in p)
            cur = abs(b - self.target)
            best = None
            order = list(range(self.n))
            self.rng.shuffle(order)
            for i in order:
                for v in (p[i] - 1, p[i] + 1):
                    if 0 <= v < self.K:
                        nb = b - self.removed[i, p[i]] + self.removed[i, v]
                        dist = abs(nb - self.target)
                        if dist < cur - 1e-15 and (best is None or dist < best[0] - 1e-15):
                            best = (dist, i, v)
            if best is not None:
                p[best[1]] = best[2]
                self.repair_log["single"] += 1
                continue
            # pair move: two single-level changes in opposite directions
            moves = [(i, v, self.removed[i, v] - self.removed[i, p[i]]) for i in range(self.n)
                     for v in (p[i] - 1, p[i] + 1) if 0 <= v < self.K]
            m = np.asarray([[i, v, d] for i, v, d in moves], float)
            nb = b + m[:, 2, None] + m[None, :, 2]
            ok = (nb >= self.lo) & (nb <= self.hi) & (m[:, 0, None] != m[None, :, 0])
            ii, jj = np.nonzero(ok)
            if not len(ii):
                self.repair_log["failed"] += 1
                return None
            k = self.rng.randrange(len(ii))
            for idx in (ii[k], jj[k]):
                p[int(m[idx, 0])] = int(m[idx, 1])
            self.repair_log["pair"] += 1
        self.repair_log["failed"] += 1
        return None

    def random_feasible(self):
        if self._random_feasible is not None:
            return self._random_feasible()
        for _ in range(100):
            p = self.repair([self.rng.randrange(self.K) for _ in range(self.n)])
            if p is not None:
                return p
        raise RuntimeError("could not draw a random feasible policy")

    def unique(self, p, seen, attempts=30):
        """Repair p; if the result was evaluated already, perturb one or two genes and repair again."""
        q = self.repair(p) if p is not None else None
        for _ in range(attempts):
            if q is not None and q not in seen:
                return q
            self.repair_log["duplicates"] += q is not None
            base = list(q) if q is not None else self.random_feasible()
            for _ in range(self.rng.choice((1, 2))):
                i = self.rng.randrange(self.n)
                base[i] = min(self.K - 1, max(0, base[i] + self.rng.choice((-1, 1))))
            q = self.repair(base)
        self.repair_log["fallback_random"] += 1
        for _ in range(100):
            q = self.random_feasible()
            if q not in seen:
                return q
        raise RuntimeError("search space exhausted")


class RandomSearch:
    name = "random"

    def __init__(self, space, seed, draws=None, **_):
        self.space, self.rng = space, random.Random(seed)
        self.draws = list(draws or [])           # pre-generated feasible draws (same generator as the initial design)

    def ask(self, count, seen):
        out = []
        while len(out) < count:
            p = self.draws.pop(0) if self.draws else self.space.random_feasible()
            if p not in seen and p not in out:
                out.append(p)
        return out

    def tell(self, policy, kl):
        pass


class TPE:
    """Hyperopt-style TPE with independent categorical Parzen estimators per gene (optuna is not installed)."""
    name = "tpe"

    def __init__(self, space, seed, gamma=0.25, candidates=64, prior=1.0, **_):
        self.space, self.rng = space, np.random.RandomState(seed)
        self.gamma, self.candidates, self.prior = gamma, candidates, prior
        self.X, self.y = [], []

    def tell(self, policy, kl):
        self.X.append(list(policy)); self.y.append(float(kl))

    def ask(self, count, seen):
        out = []
        for _ in range(count):
            X, y = np.asarray(self.X, int), np.asarray(self.y, float)
            n_good = max(1, int(math.ceil(self.gamma * len(y))))
            order = np.argsort(y)
            good, bad = X[order[:n_good]], X[order[n_good:]]
            K, n = self.space.K, self.space.n
            l = np.full((n, K), self.prior); g = np.full((n, K), self.prior)
            for row in good:
                l[np.arange(n), row] += 1
            for row in bad:
                g[np.arange(n), row] += 1
            l /= l.sum(1, keepdims=True); g /= g.sum(1, keepdims=True)
            cands = np.stack([[self.rng.choice(K, p=l[i]) for i in range(n)] for _ in range(self.candidates)])
            score = np.log(l[np.arange(n), cands] / g[np.arange(n), cands]).sum(1)
            best = cands[int(np.argmax(score))]
            p = self.space.unique(list(best), seen | set(out))
            out.append(p)
        return out


class GPBO:
    """GP (Matern 5/2 + noise) on scaled levels, EI over a candidate pool of random draws and local moves."""
    name = "gpbo"

    def __init__(self, space, seed, pool_random=1000, pool_local=1000, top=10, **_):
        self.space, self.rng = space, random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.pool_random, self.pool_local, self.top = pool_random, pool_local, top
        self.X, self.y = [], []

    def tell(self, policy, kl):
        self.X.append(list(policy)); self.y.append(float(kl))

    def _pool(self, seen):
        pool = set()
        for _ in range(self.pool_random):
            pool.add(self.space.random_feasible())
        order = np.argsort(self.y)[: self.top]
        for _ in range(self.pool_local):
            base = list(self.X[int(self.rng.choice(order))])
            for _ in range(self.rng.choice((1, 2, 3, 4))):
                i = self.rng.randrange(self.space.n)
                base[i] = min(self.space.K - 1, max(0, base[i] + self.rng.choice((-1, 1))))
            q = self.space.repair(base)
            if q is not None:
                pool.add(q)
        return [p for p in pool if p not in seen]

    def ask(self, count, seen):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from scipy.stats import norm
        X = np.asarray(self.X, float) / (self.space.K - 1)
        y = np.log(np.asarray(self.y, float))
        mu_y, sd_y = y.mean(), y.std() + 1e-9
        Xf, yf = X.copy(), (y - mu_y) / sd_y
        pool = self._pool(seen)
        P = np.asarray(pool, float) / (self.space.K - 1)
        out = []
        for _ in range(count):
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                + WhiteKernel(1e-2, (1e-5, 1e0))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=1, random_state=self.np_rng)
            gp.fit(Xf, yf)
            m, s = gp.predict(P, return_std=True)
            best = yf.min()
            z = (best - m) / np.maximum(s, 1e-9)
            ei = (best - m) * norm.cdf(z) + s * norm.pdf(z)
            ei[[i for i, p in enumerate(pool) if p in out]] = -np.inf
            k = int(np.argmax(ei))
            p = pool[k]
            out.append(p)
            Xf = np.vstack([Xf, P[k]]); yf = np.append(yf, m[k])      # Kriging believer for the rest of the batch
        return [self.space.unique(list(p), seen | set(out[:i])) if p in seen else p for i, p in enumerate(out)]


class DE:
    """DE/rand/1/bin on the continuous relaxation; trials are rounded and repaired; greedy selection per index."""
    name = "de"

    def __init__(self, space, seed, F=0.5, CR=0.9, **_):
        self.space, self.rng, self.F, self.CR = space, np.random.RandomState(seed), F, CR
        self.pop, self.fit, self.pending = [], [], []

    def tell(self, policy, kl):
        policy = tuple(policy)
        if self.pending and self.pending[0][1] == policy:          # trial for index i
            i = self.pending.pop(0)[0]
            if kl <= self.fit[i]:
                self.pop[i], self.fit[i] = np.asarray(policy, float), kl
        else:                                                       # initial population
            self.pop.append(np.asarray(policy, float)); self.fit.append(kl)

    def ask(self, count, seen):
        out = []
        n = len(self.pop)
        for i in range(min(count, n)):
            a, b, c = self.rng.choice([j for j in range(n) if j != i], 3, replace=False)
            v = self.pop[a] + self.F * (self.pop[b] - self.pop[c])
            mask = self.rng.rand(self.space.n) < self.CR
            mask[self.rng.randint(self.space.n)] = True
            trial = np.where(mask, v, self.pop[i])
            p = self.space.unique(self.space.clip_round(trial), seen | set(out))
            out.append(p); self.pending.append((i, p))
        return out


class PSO:
    """Constriction PSO (Clerc-Kennedy chi=0.729, c1=c2=2.05); positions are rounded and repaired for evaluation."""
    name = "pso"

    def __init__(self, space, seed, chi=0.7298, c1=2.05, c2=2.05, **_):
        self.space, self.rng, self.chi, self.c1, self.c2 = space, np.random.RandomState(seed), chi, c1, c2
        self.x, self.v, self.pbest, self.pfit, self.pending = [], [], [], [], []
        self.gbest, self.gfit = None, math.inf

    def tell(self, policy, kl):
        policy = tuple(policy)
        if self.pending and self.pending[0][1] == policy:
            i = self.pending.pop(0)[0]
            self.x[i] = np.asarray(policy, float)
            if kl < self.pfit[i]:
                self.pbest[i], self.pfit[i] = np.asarray(policy, float), kl
        else:
            self.x.append(np.asarray(policy, float)); self.v.append(np.zeros(self.space.n))
            self.pbest.append(np.asarray(policy, float)); self.pfit.append(kl)
        if kl < self.gfit:
            self.gbest, self.gfit = np.asarray(policy, float), kl

    def ask(self, count, seen):
        out = []
        vmax = (self.space.K - 1) / 2
        for i in range(min(count, len(self.x))):
            r1, r2 = self.rng.rand(self.space.n), self.rng.rand(self.space.n)
            self.v[i] = self.chi * (self.v[i] + self.c1 * r1 * (self.pbest[i] - self.x[i]) + self.c2 * r2 * (self.gbest - self.x[i]))
            self.v[i] = np.clip(self.v[i], -vmax, vmax)
            x = self.x[i] + self.v[i]
            p = self.space.unique(self.space.clip_round(x), seen | set(out))
            out.append(p); self.pending.append((i, p))
        return out


class GA:
    """Classic GA: tournament (k=3) parents, uniform crossover, per-gene mutation 1/n, mu+lambda survival (mu=16)."""
    name = "ga"

    def __init__(self, space, seed, mu=16, tournament=3, **_):
        self.space, self.rng, self.mu, self.k = space, random.Random(seed), mu, tournament
        self.pop = []

    def tell(self, policy, kl):
        self.pop = sorted(self.pop + [(kl, tuple(policy))])[: self.mu]

    def ask(self, count, seen):
        out = []
        for _ in range(count):
            a = min(self.rng.sample(self.pop, self.k))[1]
            b = min(self.rng.sample(self.pop, self.k))[1]
            child = [x if self.rng.random() < 0.5 else y for x, y in zip(a, b)]
            for i in range(self.space.n):
                if self.rng.random() < 1.0 / self.space.n:
                    child[i] = self.rng.randrange(self.space.K)
            p = self.space.unique(child, seen | set(out))
            out.append(p)
        return out


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES (Hansen 2016, "The CMA Evolution Strategy: A Tutorial", default parameters) on the
    continuous relaxation. The mean starts at the weighted recombination of the best
    mu initial policies and sigma at a quarter of the level range, with C = I. A sample is clipped to the level box,
    rounded and repaired before evaluation; the box-clipped continuous sample enters the update with the divergence of
    its repaired policy. A generation is updated once all its samples are evaluated."""
    name = "cmaes"

    def __init__(self, space, seed, sigma0=None, **_):
        self.space, self.rng = space, np.random.RandomState(seed)
        n = self.n = space.n
        self.lam = 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float((self.w ** 2).sum())
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.sigma = float(sigma0) if sigma0 is not None else 0.25 * (space.K - 1)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.mean, self.gen = None, 0
        self.initial, self.pending = [], {}            # policy -> [clipped sample, divergence or None]

    def _box(self, x):
        return np.clip(x, 0.0, self.space.K - 1.0)

    def tell(self, policy, kl):
        policy = tuple(policy)
        if self.mean is None:                          # the shared initial design
            self.initial.append((kl, np.asarray(policy, float)))
            return
        if policy in self.pending:
            self.pending[policy][1] = kl
            if all(v[1] is not None for v in self.pending.values()):
                self._update()

    def _start(self):
        best = sorted(self.initial, key=lambda t: t[0])[:self.mu]
        self.mean = sum(wi * x for wi, (_, x) in zip(self.w, best))

    def _update(self):
        rows = sorted(self.pending.values(), key=lambda v: v[1])
        self.pending = {}
        if len(rows) < self.lam:                       # truncated last generation at the end of the budget
            return
        old = self.mean
        y = np.array([(x - old) / self.sigma for x, _ in rows[:self.mu]])
        yw = self.w @ y
        self.mean = old + self.sigma * yw
        invsqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.gen += 1
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrt @ yw)
        hs = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))

    def ask(self, count, seen):
        if self.mean is None:
            self._start()
        out = []
        for _ in range(count):
            x = self._box(self.mean + self.sigma * (self.B @ (self.D * self.rng.randn(self.n))))
            p = self.space.unique(self.space.clip_round(x), seen | set(out))
            out.append(p)
            self.pending[tuple(p)] = [x, None]
        return out


OPTIMIZERS = {c.name: c for c in (RandomSearch, TPE, GPBO, DE, PSO, GA, CMAES)}
BATCH = {"random": 16, "tpe": 4, "gpbo": 4, "de": 16, "pso": 16, "ga": 16, "cmaes": 16}


def run_search(optimizer, initial, budget, evaluate, space, checkpoint=lambda state: None):
    """Evaluate the initial design, then ask/tell until `budget` unique feasible evaluations are recorded."""
    class State:
        pass
    state = State()
    state.entries, state.phase, state.seen = [], "initial", set()

    def record(policies, phase):
        results = evaluate(policies)
        for p, r in zip(policies, results):
            kl = float(r["kl"])
            row = {"id": len(state.entries), "evaluations": len(state.entries) + 1, "policy": list(p), **r,
                   "parent": None, "phase": phase, "branch": None, "origin": "initial" if phase == "initial" else optimizer.name}
            state.entries.append(row); state.seen.add(tuple(p))
            optimizer.tell(tuple(p), kl)

    record(list(initial), "initial")
    checkpoint(state)
    state.phase = "search"
    batch = BATCH[optimizer.name]
    while len(state.entries) < budget:
        n = min(batch, budget - len(state.entries))
        policies = optimizer.ask(n, state.seen)
        assert len(policies) == n and len(set(policies)) == n and all(p not in state.seen and space.feasible(p) for p in policies)
        record(policies, "search")
        checkpoint(state)
    state.phase = "complete"
    checkpoint(state)
    return state
