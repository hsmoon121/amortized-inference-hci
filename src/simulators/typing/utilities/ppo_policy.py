import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import make_proba_distribution

from .utils import get_device


class ModulatedMlpExtractor(nn.Module):
    """
    MlpExtractor class (described below) enabling being conditioned on task (simulatioin) parameters

    Constructs an MLP that receives directly the observations
    (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    Examples of ``net_arch''
    A network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network
    --> [55, dict(vf=[255, 255], pi=[128])]
    A simple shared network topology with two layers of size 128 --> [128, 128]

    Examples of ``concat_layers''
    --> [0, dict(vf=[0, 1], pi=[0])]

    Adapted from Stable Baselines.
    :param feature_dim: Dimension of the feature vector
    :param sim_param_dim: Dimension of task (simulation) paramters
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param embed_net_arch: The specification of the embedding networks (if use).
    :param concat_layers: Which layer to be concatenated with embedding params
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        sim_param_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        embed_net_arch: Optional[List[int]] = None,
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network

        self.activation_fn = activation_fn
        self.sim_param_dim = sim_param_dim
        self.concat_layers = concat_layers
        self.concat_pi_layers, self.concat_vf_layers = list(), list()
        if embed_net_arch is None:
            self.embed_dim = self.sim_param_dim
        else:
            self.embed_dim = embed_net_arch[-1]

        # Iterate through the shared layers and build the shared parts of the network
        self.shared_layers = nn.ModuleList([])
        last_layer_dim_shared = feature_dim - self.sim_param_dim
        for i, unit in enumerate(net_arch):
            if isinstance(unit, int):  # Check that this is a shared layer
                n_in = feature_dim - self.sim_param_dim if i == 0 else net_arch[i - 1]
                n_in += self.embed_dim if i in self.concat_layers else 0
                last_layer_dim_shared = net_arch[i]
                self.shared_layers.append(nn.Linear(n_in, last_layer_dim_shared))
            else:
                assert isinstance(unit, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in unit:
                    assert isinstance(unit["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = unit["pi"]

                if "vf" in unit:
                    assert isinstance(unit["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = unit["vf"]

                if isinstance(self.concat_layers[-1], dict):
                    unit_for_concat = self.concat_layers[-1]
                    if "pi" in unit_for_concat:
                        self.concat_pi_layers = unit_for_concat["pi"]
                    if "vf" in unit_for_concat:
                        self.concat_vf_layers = unit_for_concat["vf"]
                break  # From here on the network splits up in policy and value network

        if isinstance(net_arch[-1], dict):
            if (len(net_arch) - 1) in self.concat_layers:
                last_layer_dim_shared += self.embed_dim
        else:
            if len(net_arch) in self.concat_layers:
                last_layer_dim_shared += self.embed_dim

        # Build the non-shared part of the network
        self.pi_layers, self.vf_layers = nn.ModuleList([]), nn.ModuleList([])
        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        for i in range(len(policy_only_layers)):
            n_in = last_layer_dim_shared if i == 0 else policy_only_layers[i - 1]
            n_in += self.embed_dim if i in self.concat_pi_layers else 0
            last_layer_dim_pi = policy_only_layers[i]
            self.pi_layers.append(nn.Linear(n_in, last_layer_dim_pi))

        for i in range(len(value_only_layers)):
            n_in = last_layer_dim_shared if i == 0 else value_only_layers[i - 1]
            n_in += self.embed_dim if i in self.concat_vf_layers else 0
            last_layer_dim_vf = value_only_layers[i]
            self.vf_layers.append(nn.Linear(n_in, last_layer_dim_vf))

        if len(policy_only_layers) in self.concat_pi_layers:
            last_layer_dim_pi += self.embed_dim
        if len(value_only_layers) in self.concat_vf_layers:
            last_layer_dim_vf += self.embed_dim

        self.embed_nets = nn.ModuleList([])
        if isinstance(self.concat_layers[-1], dict):
            n_concat = len(self.concat_layers) - 1 + len(self.concat_pi_layers) + len(self.concat_vf_layers)
        else:
            n_concat = len(self.concat_layers)
        for _ in range(n_concat):
            embed_net = create_mlp(
                self.sim_param_dim,
                self.embed_dim,
                embed_net_arch[:-1],
                self.activation_fn
            ) if embed_net_arch is not None else []
            self.embed_nets.append(nn.Sequential(*embed_net))

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.device = get_device(device)
        self.to(self.device)

    def _forward_shared_net(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        obs_param, sim_param = features.to(self.device).split(
            [self.feature_dim - self.sim_param_dim, self.sim_param_dim],
            -1,
        )
        x = obs_param
        for i in range(len(self.shared_layers)):
            if i in self.concat_layers:
                embed = self.embed_nets[self.concat_layers.index(i)](sim_param)
                x = th.cat([x, embed], dim=-1)
            x = self.shared_layers[i](x)
            x = self.activation_fn()(x)

        if len(self.shared_layers) in self.concat_layers:
            embed = self.embed_nets[self.concat_layers.index(len(self.shared_layers))](sim_param)
            x = th.cat([x, embed], dim=-1)

        return x, sim_param

    def _forward_pi_net(self, x: th.Tensor, sim_param: th.Tensor) -> th.Tensor:
        for i in range(len(self.pi_layers)):
            if i in self.concat_pi_layers:
                embed = self.embed_nets[
                    len(self.concat_layers) - 1 + self.concat_pi_layers.index(i)
                ](sim_param)
                x = th.cat([x, embed], dim=-1)
            x = self.pi_layers[i](x)
            x = self.activation_fn()(x)

        if len(self.pi_layers) in self.concat_pi_layers:
            embed = self.embed_nets[self.concat_pi_layers.index(len(self.pi_layers))](sim_param)
            x = th.cat([x, embed], dim=-1)

        return x
    
    def _forward_vf_net(self, x: th.Tensor, sim_param: th.Tensor) -> th.Tensor:
        for i in range(len(self.vf_layers)):
            if i in self.concat_vf_layers:
                embed = self.embed_nets[
                    len(self.concat_layers) - 1 + len(self.concat_pi_layers) + self.concat_vf_layers.index(i)
                ](sim_param)
                x = th.cat([x, embed], dim=-1)
            x = self.vf_layers[i](x)
            x = self.activation_fn()(x)
        
        if len(self.vf_layers) in self.concat_vf_layers:
            embed = self.embed_nets[self.concat_vf_layers.index(len(self.vf_layers))](sim_param)
            x = th.cat([x, embed], dim=-1)

        return x

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent, sim_param = self._forward_shared_net(features)
        return (
            self._forward_pi_net(shared_latent, sim_param),
            self._forward_vf_net(shared_latent, sim_param)
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self._forward_pi_net(*self._forward_shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self._forward_vf_net(*self._forward_shared_net(features))



class ModulatedActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    Modified to apply ModulatedMlpExtractor

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        sim_param_dim: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        embed_net_arch: Optional[List[int]] = None,
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = [dict(pi=[0], vf=[0])],
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.sim_param_dim = sim_param_dim
        self.net_arch = net_arch
        self.embed_net_arch = embed_net_arch
        self.concat_layers = concat_layers
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None

        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = ModulatedMlpExtractor(
            self.features_dim,
            sim_param_dim=self.sim_param_dim,
            net_arch=self.net_arch,
            embed_net_arch=self.embed_net_arch,
            concat_layers=self.concat_layers,
            activation_fn=self.activation_fn,
            device="auto",
        )