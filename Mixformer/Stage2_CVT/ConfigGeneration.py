def make_preprocessor_config(s_hw, t_hw, in_c, out_c, p_size, p_pad, p_stride, use_cls):
    config = {'search_hw': s_hw, 'target_hw': t_hw, 'in_c': in_c, 'out_c': out_c,
              'patch_size': p_size, 'patch_padding': p_pad, 'patch_stride': p_stride, "use_cls": use_cls}
    return config

def make_dw_qkv_config(embd_d, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads, use_cls):
    config = {'embd_d': embd_d, 'kernel_size': ker_size, 'padding_q': pad_q, 'stride_q': stride_q,
              'padding_kv': pad_kv, 'stride_kv': stride_kv, 'num_heads': n_heads, "use_cls": use_cls}
    return config

def make_mha_config(embd_d, n_heads, ff_scale):
    config = {'embd_d': embd_d, 'num_heads': n_heads, 'ff_scale': ff_scale}
    return config

def make_mam_config(embd_d, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads, ff_scale, use_cls):
    config = {'embd_d': embd_d}
    dwqkv = make_dw_qkv_config(embd_d, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads, use_cls)
    config['depthwise_qkv'] = dwqkv
    config['attention'] = make_mha_config(embd_d, n_heads, ff_scale)
    return config

def make_stage_config(s_hw, t_hw, in_c, out_c, p_size, p_pad, p_stride, ker_size,
                      pad_q, stride_q, pad_kv, stride_kv, n_heads, ff_scale, n_mam, use_cls):
    prep = make_preprocessor_config(s_hw, t_hw, in_c, out_c, p_size, p_pad, p_stride, use_cls)
    config = {'num_mams': n_mam, 'preprocessor': prep}
    config['mam'] = make_mam_config(out_c, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads,
                                    ff_scale, use_cls)
    return config

def make_mask_head_config(c):
    config = {'channels': c}
    return config

def make_class_head_config(d, inner_d):
    config = {'embd_d': d, 'inner_dim': inner_d}
    return config

def make_transformer_config(size_type='large'):
    num_stages = 2
    lst_stg = num_stages - 1
    if size_type == 'medium':
        embds, n_heads, num_mams = [48, 72], [2, 3], [3, 6]
    elif size_type == 'large':
        embds, n_heads, num_mams = [72, 108], [2, 3], [4, 8]
    else:
        raise ValueError(f'Invalid size type {size_type}')
    config = {'proj_channels': embds[0], 'num_stages': num_stages}
    config['stage_0'] = make_stage_config(64, 48, embds[0], embds[0], 3, 1, 2, 3, 1, 1, 1, 2, n_heads[0],
                                          3, num_mams[0], False)
    config['stage_1'] = make_stage_config(32, 24, embds[0], embds[1], 3, 1, 2, 3, 1, 1, 1, 2, n_heads[1],
                                          3, num_mams[1], True)
    config['mask_head'] = make_mask_head_config(embds[lst_stg])
    config['class_head'] = make_class_head_config(embds[lst_stg], 256)
    return config
