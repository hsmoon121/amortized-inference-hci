default_menu_config = dict(
    name = "default",
    learning_rate = 0.00005,
    lr_gamma = 0.9,
    clipping = 1.0,
    amortizer = dict(
        device = None,
        trial_encoder_type = None,
        encoder = dict(
            traj_sz = 0,
            stat_sz = 4,
            batch_norm = True,
            mlp = dict(
                feat_sz = 16,
                out_sz = 32,
                depth = 4,
            ),
        ),
        invertible = dict(
            param_sz = 4,
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
        )
    ),
    simulator = dict(
        seed = None,
        variable_params = True,
        variant = 3,
        targeted_params = ["d_fix", "d_sel", "p_rec", "p_sem"],
        param_symbol = [
            "$d_{\mathit{fix}}$",
            "$d_{\mathit{sel}}$",
            "$p_{\mathit{rec}}$",
            "$p_{\mathit{sem}}$"
        ],
        base_params = [2.80, 0.29, 0.69, 0.93], # CHI'17 baseline (result of ABC)
        param_distr = dict(
            distr = ["truncnorm", "truncnorm", "uniform", "uniform"],
            minv = [0.0, 0.0, 0.0, 0.0],
            maxv = [6.0, 1.0, 1.0, 1.0],
            mean = [3.0, 0.3, 0.0, 0.0],
            std = [1.0, 0.3, 0.0, 0.0],
        ),
        use_uniform = False,
    )
)