def make_preprocessor_config(c, embd_d, s_h, s_w, p_size, p_pad, p_stride, cls_inp_sz):
    config = {'channels': c, 'embed_dim': embd_d, 'search_inp_h': s_h, 'search_inp_w': s_w,
              'patch_size': p_size, 'patch_padding': p_pad, 'patch_stride': p_stride,
              'cls_input_size': cls_inp_sz}
    new_size = lambda x: (x - p_size + 2 * p_pad) // p_stride + 1
    config.update({'search_ous_h': new_size(s_h), 'search_ous_w': new_size(s_w)})
    return config

def make_dw_qkv_config(embd_d, s_h, s_w, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads):
    config = {'embed_dim': embd_d, 'search_inp_h': s_h, 'search_inp_w': s_w, 'kernel_size': ker_size,
              'padding_q': pad_q, 'stride_q': stride_q, 'padding_kv': pad_kv, 'stride_kv': stride_kv,
              'num_heads': n_heads}
    new_sz_q = lambda x: (x - ker_size + 2 * pad_q) // stride_q + 1
    config.update({'search_q_h': new_sz_q(s_h), 'search_q_w': new_sz_q(s_w)})
    new_sz_kv = lambda x: (x - ker_size + 2 * pad_kv) // stride_kv + 1
    config.update({'search_kv_h': new_sz_kv(s_h), 'search_kv_w': new_sz_kv(s_w)})
    return config

def make_mha_config(embd_d, s_h, s_w, t_q_h, t_q_w, t_kv_h, t_kv_w, n_heads, ff_scale):
    config = {'embed_dim': embd_d, 'search_inp_h': s_h, 'search_inp_w': s_w, 'search_q_h': t_q_h,
              'search_q_w': t_q_w, 'search_kv_h': t_kv_h, 'search_kv_w': t_kv_w, 'num_heads': n_heads,
              'ff_scale': ff_scale}
    return config

def make_mam_config(embd_d, s_h, s_w, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads, ff_scale):
    config = {'embed_dim': embd_d}
    dwqkv = make_dw_qkv_config(embd_d, s_h, s_w, ker_size, pad_q, stride_q, pad_kv, stride_kv, n_heads)
    config['depthwise_qkv'] = dwqkv
    config['attention'] = make_mha_config(embd_d, s_h, s_w, dwqkv['search_q_h'], dwqkv['search_q_w'],
                                          dwqkv['search_kv_h'], dwqkv['search_kv_w'], n_heads, ff_scale)
    return config

def make_stage_config(c, embd_d, s_h, s_w, n_mam, p_size, p_pad, p_stride, ker_size,
                      pad_q, stride_q, pad_kv, stride_kv, n_heads, ff_scale):
    prep = make_preprocessor_config(c, embd_d, s_h, s_w, p_size, p_pad, p_stride)
    config = {'channels': c, 'embed_dim': embd_d, 'search_inp_h': s_h, 'search_inp_w': s_w,
              'preprocessor': prep}
    config['mam'] = make_mam_config(embd_d, prep['search_ous_h'], prep['search_ous_w'], ker_size,
                                    pad_q, stride_q, pad_kv, stride_kv, n_heads, ff_scale)
    config.update({'search_ous_h': prep['search_ous_h'], 'search_ous_w': prep['search_ous_w'],
                   'num_mam_blocks': n_mam})
    return config

def make_mask_head_config(c, s_h, s_w, s_inis_h, s_inis_w):
    config = {'channels': c, 'search_inp_h': s_h, 'search_inp_w': s_w, 'search_ous_h': s_inis_h,
              'search_ous_w': s_inis_w}
    return config

def make_mixformer_config(size_type='medium'):
    num_stages = 2
    lst_stg = num_stages - 1
    if size_type == 'small':
        embds, n_heads, num_mams = [32, 64], [1, 2], [2, 4]
    elif size_type == 'medium':
        embds, n_heads, num_mams = [48, 72], [2, 3], [3, 6]
    elif size_type == 'large':
        embds, n_heads, num_mams = [64, 96], [3, 4], [4, 8]
    else:
        raise ValueError(f'Invalid size type {size_type}')
    config = {'search_inp_h': 48, 'search_inp_w': 48, 'num_stages': num_stages}
    config['stage_0'] = make_stage_config(3, embds[0], 48, 48, num_mams[0], 5, 2, 2, 3, 1, 1, 1, 2,
                                          n_heads[0], 3)
    config['stage_1'] = make_stage_config(embds[0], embds[1], 24, 24, num_mams[1], 3, 1, 2, 3, 1, 1, 1, 2,
                                          n_heads[1], 3)
    config['classes_head'] = make_mask_head_config(embds[lst_stg], config[f'stage_{lst_stg}']['search_ous_h'],
                                                   config[f'stage_{lst_stg}']['search_ous_w'], 64, 64)
    config.update({'search_ous_h': config[f'stage_{lst_stg}']['search_ous_h'],
                   'search_ous_w': config[f'stage_{lst_stg}']['search_ous_w'],
                   'out_embed_dim': embds[lst_stg]})
    return config







