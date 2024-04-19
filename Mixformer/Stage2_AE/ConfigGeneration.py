def make_ae_config():
    embed_dim = 256
    config = {"embeddings": embed_dim}
    coder_cfg = {"part1_channels": 64, "part2_channels": 80, "part3_channels": 96, "target_h": 48,
               "target_w": 48, "search_h": 64, "search_w": 64, "embeddings": embed_dim,
               "target_out_h": 12, "target_out_w": 12, "search_out_h": 16, "search_out_w": 16,
               "reduce_size": 64}
    coder_cfg['block1'] = {"channels": 64, "mid_channels": 64}
    coder_cfg['block2'] = {"channels": 80, "mid_channels": 80}
    coder_cfg['block3'] = {"channels": 96, "mid_channels": 96}
    config["encoder"] = coder_cfg
    config["decoder"] = coder_cfg
    return config
