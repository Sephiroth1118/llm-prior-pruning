"""Parameter count of a masked model after physical removal of the pruned channels."""

def retained_parameters(model, sync):
    """Count actual retained projections from synchronized GQA ownership, including biases."""
    attn = {x['layer_idx']: x for x in sync['attention']}
    mlp = {x['layer_idx']: x for x in sync['mlp']}
    total = sum(p.numel() for p in model.parameters())
    for i, layer in enumerate(model.model.layers):
        q = attn[i]['active_query_head_count'] * layer.self_attn.head_dim
        kv = attn[i]['active_kv_group_count'] * layer.self_attn.head_dim
        m = mlp[i]['active_mlp_dim_count']
        for linear, width, input_axis in [(layer.self_attn.q_proj, q, False), (layer.self_attn.k_proj, kv, False),
                (layer.self_attn.v_proj, kv, False), (layer.self_attn.o_proj, q, True),
                (layer.mlp.gate_proj, m, False), (layer.mlp.up_proj, m, False), (layer.mlp.down_proj, m, True)]:
            total -= linear.weight.numel() - projection_count(linear, width, input_axis, include_bias=False)
            if linear.bias is not None and not input_axis:
                total -= linear.bias.numel() - width
    return total


def projection_count(linear, width, input_axis=False, include_bias=True):
    count = width * linear.weight.shape[0 if input_axis else 1]
    if include_bias and linear.bias is not None:
        count += linear.bias.numel() if input_axis else width
    return count


