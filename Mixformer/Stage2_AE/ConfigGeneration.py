def make_ae_config():
    embed_dim = 256
    config = {"embeddings": embed_dim}
    coder_cfg = {"part1_channels": 64, "part2_channels": 88, "part3_channels": 112, "target_h": 48,
               "target_w": 48, "search_h": 64, "search_w": 64, "pos_enc_channels": 32, "type_channels": 64,
               "embeddings": embed_dim}
    coder_cfg['block1'] = {"channels": 64, "mid_channels": 64}
    coder_cfg['block2'] = {"channels": 88, "mid_channels": 88}
    coder_cfg['block3'] = {"channels": 112, "mid_channels": 112}
    config["encoder"] = coder_cfg
    config["decoder"] = coder_cfg
    return config
