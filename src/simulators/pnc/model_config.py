import numpy as np

MEAN_PARAMS = dict(
    nv=0.245,
    np=0.047,
    sigmav=0.169,
    csigma=0.14902,
    cmu=0.385,
    nu=15.766,
    delta=0.399,
)

STD_PARAMS = dict(
    nv=0.056,
    np=0.033,
    sigmav=0.082,
    csigma=0.08372,
    cmu=0.142,
    nu=5.376,
    delta=0.0143,
)

CONFIG = dict(
    Tp=0.1,
    max_th=2.5,
    forearm=0.257,
    fixed=False,
    Th=0.1 + (np.arange(25.0) * 0.1),
    K=np.arange(2.0) * 1,
    INTERVAL=0.05,
    p=1,
    WINDOW_WIDTH=0.52704,
    WINDOW_HEIGHT=0.29646,
    MOUSE_GAIN=10.40,

    LOW=np.array(
        [
            -1,
            -1,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            -1,
            -1,
            -0.5,
            -0.5,
            0.0096,
            -1,
            -1,
        ]
    ),
    HIGH=np.array(
        [
            1,
            1,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            1,
            1,
            0.5,
            0.5,
            0.024,
            1,
            1,
        ]
    ),
)

pnc_qnet_config = dict(
    name = "pnc_dqn",
    obs_size = 11,
    act_size = 50,
    z_size = 4,
    hidden_size = 64,
    hidden_depth = 2,
    no_embed = True,
    concat_pos = [1, 2],
    mid_sz = None,
    mid_d = None,
    embed_size = None,
    embed_act = "relu",
    device = "cpu",
)