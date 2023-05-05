default_pnc_config = dict(
    name = "default",
    learning_rate = 0.0001,
    lr_gamma = 0.9,
    clipping = 0.5,
    amortizer = dict(
        device = None,
        trial_encoder_type = "attention",
        encoder = dict(
            traj_sz = 5,
            stat_sz = 12,
            batch_norm = True,
            traj_encoder_type = "transformer",
            transformer = dict(
                num_latents = 4,
                n_block = 2,
                query_sz = 8,
                out_sz = 8,
                head_sz = 8,
                n_head = 4,
                attn_dropout = 0.4,
                res_dropout = 0.4,
                max_freq = 10,
                n_freq_bands = 2,
                max_step = 102,
            ),
            mlp = dict(
                feat_sz = 64,
                out_sz = 24,
                depth = 2,
            ),
        ),
        trial_encoder = dict(
            attention = dict(
                num_latents = 4,
                n_block = 2,
                query_sz = 32,
                out_sz = 32,
                head_sz = 8,
                n_head = 4,
                attn_dropout = 0.4,
                res_dropout = 0.4,
            )
        ),
        invertible = dict(
            param_sz = 4,
            n_block = 5,
            act_norm = True,
            invert_conv = True,
            batch_norm = False,
            block = dict(
                permutation = False,
                head_depth = 2,
                head_sz = 32,
                cond_sz = 32,
                feat_sz = 32,
                depth = 2,
            )
        )
    ),
    simulator = dict(
        seed = None,
        targeted_params = ["nv", "sigmav", "csigma", "max_th"],
        param_symbol = ["$n_v$", "$\sigma_v$", "$c_\sigma$", "$T_{h,max}$"],
        base_params = [0.245, 0.169, 0.14902, 2.5], # CHI'22 baseline (measured values)
        prior = "log-uniform",
    ),
)