default_typing_config = dict(
    name = "default",
    learning_rate = 0.00005,
    lr_gamma = 0.9,
    clipping = 1.0,
    point_estimation = False,
    amortizer = dict(
        device = None,
        trial_encoder_type = "attention",
        encoder = dict(
            traj_sz = 0,
            stat_sz = 5,
            batch_norm = True,
            mlp = dict(
                feat_sz = 16,
                out_sz = 32,
                depth = 2,
            ),
        ),
        trial_encoder = dict(
            attention = dict(
                num_latents = 2,
                query_sz = 32,
                out_sz = 32,
                n_block = 2,
                head_sz = 32,
                n_head = 2,
                attn_dropout = 0.2,
                res_dropout = 0.2,
            )
        ),
        invertible = dict(
            param_sz = 3,
            n_block = 5,
            act_norm = True,
            invert_conv = True,
            batch_norm = False,
            block = dict(
                permutation = False,
                head_depth = 1,
                head_sz = 32,
                cond_sz = 32,
                feat_sz = 16,
                depth = 2,
            )
        ),
        linear = dict( # For point_estimation
            in_sz = 32,
            out_sz = 3,
            hidden_sz = 512,
            hidden_depth = 2,
            batch_norm = False,
            activation = "relu",
        ),
    ),
    simulator = dict(
        seed = None,
        variable_params = True,
        targeted_params = ["obs_prob", "who_alpha", "who_k"],
        param_symbol = ["$p_{\mathit{obs}}$", "$\\alpha$", "$k$"],
        base_params = [0.7, 0.6, 0.12], # CHI'21 baseline (manually fitted)
        param_distr = dict(
            distr = ["uniform", "truncnorm", "truncnorm"],
            minv = [0.0, 0.4, 0.04],
            maxv = [1.0, 0.9, 0.20],
            mean = [0.7, 0.6, 0.12],
            std = [0.0, 0.3, 0.08],
        ),
        concat_layers = [0,],
        embed_net_arch = [16, 4],
        use_uniform = False,
    )
)