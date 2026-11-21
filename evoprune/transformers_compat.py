#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


def patch_transformers_pruning_utils():
    """Restore pruning helper aliases expected by older remote model code."""
    import torch
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(modeling_utils, "prune_linear_layer") and hasattr(pytorch_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer

    if hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size, device="cpu")
        for head in heads:
            shifted_head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[shifted_head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), dtype=torch.long, device="cpu")[mask].long()
        return heads, index

    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None, *args, **kwargs):
            cache = cls()
            if past_key_values is None:
                return cache
            for layer_idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                key_states, value_states = layer_past[:2]
                cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            legacy_cache = []
            for layer in self.layers:
                legacy_cache.append((layer.keys, layer.values))
            return tuple(legacy_cache)

        DynamicCache.to_legacy_cache = to_legacy_cache
