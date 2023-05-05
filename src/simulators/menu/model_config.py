menu_qnet_config = dict(
    name = "menu_dqn",
    obs_size = 18, # 8 (relevance) + 9 (focus, one-hot) + 1 (quit)
    act_size = 9, # 8 (menu-items) + 1 (quit)
    z_size = 4,
    hidden_size = 64,
    hidden_depth = 2,
    no_embed = True,
    concat_pos = [0, 1, 2],
    batch_norm = False,
    activation = "relu",
    mid_sz = None,
    mid_d = None,
    embed_size = None,
    embed_act = None,
    device = "cpu",
)