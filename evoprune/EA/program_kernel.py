"""Training data and validation for history-adaptive (program-defined) mutation kernels."""
import json
import math

from .mutation_kernel import MutationKernel, TRANSFER_KINDS, _transition
from .policy import Policy

API_VERSION = 1
WEIGHT_KEYS = ("attn_compress", "attn_decompress", "mlp_compress", "mlp_decompress", "transfer_weights")


class ProgramError(RuntimeError):
    pass


class ProgramAbort(BaseException):
    """Bypass the old online-fitter's Exception -> keep-previous-kernel fallback."""


def json_copy(value):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > 2**20:
        raise ProgramError("program input/state exceeds 1 MiB")
    return json.loads(encoded)


def training_data(history, num_layers, evidence_limit, expensive_module, *,
                  quality_key="kl", higher_is_better=False):
    """Select recent transitions; default to KL reduction, or an explicit objective."""
    rows = [item for entry in history if (item := _transition(entry, num_layers)) is not None]
    rows = rows[-int(evidence_limit):]
    archive = {entry["history_id"]: entry for entry in history if "history_id" in entry}
    data, ids = [], []
    for delta, reward_delta, entry in rows:
        parent = archive.get(entry.get("mutation_parent_history_id"))
        if parent is None or Policy.from_mapping(parent) != Policy.from_mapping(entry["mutation_parent_policy"]):
            raise ProgramError("missing or inconsistent parent history")
        parent_quality, child_quality = float(parent[quality_key]), float(entry[quality_key])
        gain = child_quality-parent_quality if higher_is_better else parent_quality-child_quality
        if not math.isfinite(gain):
            raise ProgramError("nonfinite parent/child quality: " + quality_key)
        edits = entry.get("mutation_edits", [])
        modules = {edit.get("module") for edit in edits}
        kind = None
        if modules == {"attn"}:
            kind = "same_attention"
        elif modules == {"mlp"}:
            kind = "same_mlp"
        elif modules == {"attn", "mlp"}:
            expensive_delta = sum(int(edit.get("delta", 0)) for edit in edits
                                  if edit.get("module") == expensive_module)
            kind = ("compress_expensive_release_cheap" if expensive_delta > 0
                    else "release_expensive_compress_cheap")
        p = Policy.from_mapping(parent)
        x = list(p.attn_levels+p.mlp_levels)
        data.append(dict(parent=x, child=[a+b for a, b in zip(x, delta)], gain=gain,
                         reward_delta=reward_delta, transfer_kind=kind))
        ids.append(entry.get("history_id"))
    return json_copy(data), ids


def validate_kernel(value, n, *, allow_transfer_mapping=False, **kwargs):
    if not isinstance(value, dict) or set(value) != set(WEIGHT_KEYS):
        raise ProgramError("make_kernel must return exactly the five weight arrays")
    values, normalizations = {}, []
    for key in WEIGHT_KEYS:
        array = value[key]
        if key == "transfer_weights" and allow_transfer_mapping and isinstance(array, dict):
            missing, extra = set(TRANSFER_KINDS)-set(array), set(array)-set(TRANSFER_KINDS)
            if missing or extra:
                raise ProgramError(f"transfer_weights: expected exactly {list(TRANSFER_KINDS)}; "
                                   f"missing={sorted(missing)}, extra={sorted(map(str, extra))}")
            array = [array[kind] for kind in TRANSFER_KINDS]
            normalizations.append(dict(field=key, action="mapping_to_schema_ordered_list", order=list(TRANSFER_KINDS)))
        expected = 4 if key == "transfer_weights" else n
        if not isinstance(array, list):
            raise ProgramError(f"{key}: expected list of length {expected}; got {type(array).__name__}")
        if len(array) != expected:
            raise ProgramError(f"{key}: expected list of length {expected}; got list of length {len(array)}")
        for index, v in enumerate(array):
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ProgramError(f"{key}[{index}]: expected positive finite int/float; got {v!r} ({type(v).__name__})")
        total = sum(array)
        if not math.isfinite(total) or min(array)/total <= 0:
            raise ProgramError("weights cannot be normalized numerically: " + key)
        values[key] = tuple(array)
    if normalizations:
        kwargs["metadata"] = {**kwargs.get("metadata", {}), "output_normalizations": normalizations}
    return MutationKernel(**values, **kwargs)
