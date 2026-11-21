"""Frozen scikit-learn surrogate (Gaussian process) that guides the mutation operator.

FAMILY and GUIDANCE select the surrogate and whether average finite differences are compiled at fit
time. The program is loaded in process by evoprune-based experiments (see experiments/gp_guide.py)
and is written to be independent of the installed scikit-learn version.
"""
import warnings

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.svm import SVR
import torch

FAMILY = "gp"
GUIDANCE = True


def _fit(x, y, family):
    if family == "rf":
        model = RandomForestRegressor(n_estimators=128, max_depth=5,
                                      min_samples_leaf=4, max_features=1/3,
                                      n_jobs=1, random_state=0)
    elif family == "svr":
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale",
                    tol=1e-3, max_iter=10000, cache_size=64)
        # BaseLibSVM.fit draws a seed even for SVR; avoid the global EA/worker RNG.
        # The inherited attribute exists although SVR omits it from its constructor.
        model.random_state = 0
    elif family == "gbdt":
        # Default least-squares loss is 'ls' in worker 0.23, 'squared_error' in 1.x.
        model = GradientBoostingRegressor(n_estimators=64, learning_rate=0.05,
            max_depth=2, min_samples_leaf=4, subsample=1.0, max_features=None,
            n_iter_no_change=None, random_state=0)
    elif family == "gp":
        distances = pdist(x)
        length = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) *
                   Matern(length_scale=length, length_scale_bounds=(1e-3, 1e3), nu=2.5) +
                   WhiteKernel(0.01, (1e-6, 1e1)),
            alpha=1e-8, normalize_y=False, n_restarts_optimizer=0, random_state=0)
    else:
        raise ValueError("unknown family")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    if family == "svr" and model.fit_status_ != 0:
        raise ValueError("SVR solver did not converge within 10000 iterations")
    return model, sorted({str(w.message) for w in caught})


def _export(model, family):
    if family == "gp":
        return dict(train=model.X_train_.tolist(), alpha=model.alpha_.tolist(),
                    amplitude=float(model.kernel_.k1.k1.constant_value),
                    length=float(model.kernel_.k1.k2.length_scale),
                    noise=float(model.kernel_.k2.noise_level))
    if family == "svr":
        return dict(support_vectors=model.support_vectors_.tolist(),
                    dual_coef=model.dual_coef_[0].tolist(), intercept=float(model.intercept_[0]),
                    gamma=float(model._gamma), solver_random_state=model.random_state,
                    parameters=model.get_params())
    if family not in ("rf", "gbdt"):
        raise ValueError("unknown family")
    # JSON arrays avoid pickle and support the worker's fresh namespace per call.
    estimators = model.estimators_ if family == "rf" else model.estimators_[:, 0]
    result = {"trees": [dict(left=e.tree_.children_left.tolist(),
                           right=e.tree_.children_right.tolist(),
                           feature=e.tree_.feature.tolist(),
                           threshold=e.tree_.threshold.tolist(),
                           value=e.tree_.value[:, 0, 0].tolist()) for e in estimators]}
    if family == "gbdt":
        result.update(initial=float(model.init_.constant_[0, 0]),
                      learning_rate=float(model.learning_rate), parameters=model.get_params())
    return result


def _predict(state, delta):
    x = np.asarray(delta, dtype=np.float64) / np.asarray(state["spans"])
    if x.ndim != 2 or x.shape[1] != len(state["spans"]) or not np.isfinite(x).all():
        raise ValueError("invalid transition matrix")
    if not len(x):
        return np.empty(0)
    fit = state["fit"]
    if state["family"] == "gp":
        d = cdist(x / fit["length"], np.asarray(fit["train"]) / fit["length"])
        t = np.sqrt(5.0) * d
        kernel = fit["amplitude"] * (1 + t + t*t/3) * np.exp(-t)
        result = kernel @ np.asarray(fit["alpha"])
    elif state["family"] == "svr":
        support = np.asarray(fit["support_vectors"]).reshape(-1, x.shape[1])
        result = np.exp(-fit["gamma"] * cdist(x, support, "sqeuclidean")) @ np.asarray(fit["dual_coef"])
        result += fit["intercept"]
    elif state["family"] in ("rf", "gbdt"):
        # sklearn trees cast their input to float32 before comparing thresholds.
        x = x.astype(np.float32)
        boosting = state["family"] == "gbdt"
        result = np.full(len(x), fit["initial"]) if boosting else np.zeros(len(x))
        for tree in fit["trees"]:
            left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
            feature, threshold = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
            nodes = np.zeros(len(x), dtype=np.int64)
            for _ in range(5):
                active = np.flatnonzero(left[nodes] != -1)
                if not len(active):
                    break
                node = nodes[active]
                nodes[active] = np.where(x[active, feature[node]] <= threshold[node],
                                         left[node], right[node])
            if np.any(left[nodes] != -1):
                raise ValueError("tree exceeds the fixed depth")
            values = np.asarray(tree["value"])[nodes]
            result += fit["learning_rate"] * values if boosting else values
        if not boosting:
            result /= len(fit["trees"])
    else:
        raise ValueError("unknown family")
    return state["mean"] + state["scale"] * result


def _weights(values, schema):
    # Identical operations to M0's forward conversion, applied to each direction.
    values = torch.tensor(values, dtype=torch.float64)
    centered = values - values.mean()
    scale = centered.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) <= 1e-12:
        return [1.0] * len(values)
    ref = schema["reference"]
    return torch.exp(torch.clamp(centered/scale, -4, 4)/ref["temperature"]).clamp(
        ref["min_weight"], ref["max_weight"]).tolist()


def _scores(state, x):
    base = _predict(state, x)
    queries = []
    for sign in (1, -1):
        for j in range(x.shape[1]):
            query = x.copy()
            query[:, j] += sign
            queries.append(query)
    predictions = _predict(state, np.concatenate(queries)).reshape(2, x.shape[1], len(x))
    return np.mean(predictions-base, axis=2).tolist()


def fit_model(data, schema):
    n = schema["num_layers"]
    x = np.asarray([np.subtract(r["child"], r["parent"]) for r in data], dtype=np.float64)
    y = np.asarray([r["gain"] for r in data], dtype=np.float64)
    if len(data) < 2 or x.shape != (len(data), 2*n) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid training data")
    spans = np.maximum(np.subtract(schema["maxima"], schema["minima"]), 1)
    mean, scale = float(y.mean()), float(y.std())
    scale = scale if scale > 1e-12 else 1.0
    model, messages = _fit(x/spans, (y-mean)/scale, FAMILY)
    state = dict(family=FAMILY, spans=spans.tolist(), mean=mean, scale=scale,
                 fit=_export(model, FAMILY), warnings=messages, sample_count=len(data))
    groups = [[r["reward_delta"] for r in data if r["transfer_kind"] == kind]
              for kind in schema["transfer_kinds"]]
    # Whole-offspring labels: this prior does not identify atomic-action rewards.
    observed = [float(np.mean(g)) for g in groups if g]
    prior = float(np.mean(observed)) if observed else 0.0
    shrink = 3.0
    transfers = [(float(np.sum(g)) + shrink * prior) / (len(g) + shrink) if g
                 else prior for g in groups]
    state["transfer_diagnostics"] = dict(counts=[len(g) for g in groups],
                                         prior=prior, shrink=shrink)
    state["kernel"] = {k: [1.0]*n for k in
                       ("attn_compress", "attn_decompress", "mlp_compress", "mlp_decompress")}
    state["kernel"]["transfer_weights"] = _weights(transfers, schema)
    if GUIDANCE:
        scores = np.asarray(_scores(state, x))
        values = [scores[0, :n], scores[1, :n], scores[0, n:], scores[1, n:]]
        for key, value in zip(tuple(state["kernel"])[:4], values):
            state["kernel"][key] = _weights(value, schema)
        # Sensitivities may query off the legal edit manifold; no causal claim.
        children = np.asarray([r["child"] for r in data])
        costs = np.asarray(schema["level_costs"])
        budget = x @ costs
        state["score_diagnostics"] = dict(
            raw_scores=scores.tolist(), group_std=[float(v.std()) for v in values],
            exact_zero_fraction=float(np.mean(scores == 0)),
            query_out_of_bounds_fraction=float(np.mean([
                (children[:, j]+s < schema["minima"][j]) |
                (children[:, j]+s > schema["maxima"][j]) for s in (1, -1) for j in range(2*n)])),
            query_outside_parent_tolerance_fraction=float(np.mean([
                np.abs(budget+s*costs[j]) > 0.001 + 1e-12 for s in (1, -1) for j in range(2*n)])))
    return state


def make_kernel(state, parent, schema):
    return state["kernel"]


def predict_gain(state, parents, children, schema):
    if len(parents) != len(children):
        raise ValueError("parent/child batch sizes differ")
    return _predict(state, np.asarray(children, dtype=np.float64)-np.asarray(parents)).tolist()
