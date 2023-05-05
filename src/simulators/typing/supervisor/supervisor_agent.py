"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import os
from pathlib import Path
import csv
import tqdm
import random
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from datetime import datetime

import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ..abstract.agent import Agent
from .supervisor_agent_environment import SupervisorEnvironment
from ..utilities.ppo_policy import ModulatedActorCriticPolicy
from ..utilities.callbacks import ProgressBarManager


class SupervisorAgent(Agent):

    def __init__(
        self,
        layout_config,
        agent_params,
        train,
        verbose=False,
        variable_params=False,
        fixed_params=None,
        concat_layers=[0, 1, 2],
        embed_net_arch=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.layout_config = layout_config
        self.agent_params = agent_params
        self.train_model = train
        self.verbose = verbose
        self.variable_params = variable_params

        if fixed_params is not None:
            assert len(fixed_params) == 3
        self.fixed_params = fixed_params

        self.env_cls = SupervisorEnvironment
        self.env = self.env_cls(
            self.layout_config,
            self.agent_params,
            self.train_model,
            variable_params=self.variable_params,
            fixed_params=self.fixed_params
        )
        self.eval_env = None

        lr = 0.001 if agent_params is None else agent_params['supervisor']['learning_rate']
        n_units = 512 if agent_params is None else int(agent_params['supervisor']['n_units'])
        pre_load = False if agent_params is None else bool(agent_params['supervisor']['pre_load'])
        self.gpu = True if agent_params is None else bool(agent_params['supervisor']['gpu'])
        self.save_path = os.path.join(Path(__file__).parent.parent, "models/supervisor")
        self.output_path = os.path.join(Path(__file__).parent.parent, "outputs")

        self.episodes = 1000000 if agent_params is None else int(agent_params['supervisor']['episodes'])
        self.log_interval = 1000 if agent_params is None else int(agent_params['supervisor']['log_interval'])

        # Modification from original repo: Use stable-baselines3 for PPO training
        policy_cls = ModulatedActorCriticPolicy if variable_params else "MlpPolicy"
        policy_kwargs=dict(
            net_arch=[n_units, n_units],
            activation_fn=nn.ReLU,
        )
        if variable_params:
            policy_kwargs.update(
                sim_param_dim=3,
                concat_layers=concat_layers,
                embed_net_arch=embed_net_arch,
            )

        self.model = PPO(
            policy_cls,
            self.env,
            learning_rate=lr,
            gamma=1.00, # 0.99, # default value of chainerrl-ppo
            ent_coef=1e-2,
            vf_coef=1.0,
            max_grad_norm=0.2,
            verbose=self.verbose,
            tensorboard_log="data/board",
            policy_kwargs=policy_kwargs
        )

        if pre_load:
            ckpts = os.listdir(self.save_path)
            if os.path.exists(os.path.join(self.save_path, "best_model.zip")):
                self.load("best_model.zip")
            elif len(ckpts) > 0:
                ckpts_w_step = [f for f in ckpts if f.startswith("model_")]
                ckpts_w_step.sort()
                self.load(max(ckpts_w_step))
            else:
                print(f"[ simulator - no checkpoint ] start training from scratch.")

    def load(self, ckpt_name=None):
        model_path = os.path.join(self.save_path, ckpt_name)
        assert os.path.exists(model_path)
        import src
        import sys
        sys.modules["simulators"] = src.simulators
        self.model = PPO.load(model_path)
        print(f"[ simulator - loaded checkpoint ]\n\t{model_path}")

    def train(self, episodes):
        """
        Trains the model for given number of episodes.
        """
        freq = 500000 if self.variable_params else 100000

        checkpoint_callback = CheckpointCallback(
            save_freq=freq,
            save_path=self.save_path,
            name_prefix="model",
        )
                
        with ProgressBarManager(episodes) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
            print(f" [ Training start ] ")
            if self.variable_params:
                print(f" - free_params: variable")
            else:
                print(f" - free_params: {self.env.free_params}")

            self.model.learn(
                episodes,
                log_interval=self.log_interval,
                callback=[progress_callback, checkpoint_callback],
            )
            
        self.model.save(os.path.join(self.save_path, "model"))

    def act(self, obs, deterministic=False):
        obs_tensor = torch.Tensor(obs.reshape((1, -1))).to(self.model.policy.device)
        # Preprocess the observation if needed
        features = self.model.policy.extract_features(obs_tensor)
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        # Evaluate the values for the given observations
        distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
        action = distribution.get_actions(deterministic=deterministic)
        action = np.clip(action.cpu().detach().numpy(), self.env.action_space.low, self.env.action_space.high)
        return action

    def simulate(
        self,
        fixed_params,
        n_eval_sentence=1500,
        random_sample=False,
        return_info=False,
        verbose=False,
    ):
        if self.eval_env is None:
            self.eval_env = self.env_cls(
                self.layout_config,
                self.agent_params,
                False,
                self.variable_params,
                self.fixed_params,
            )
        self.eval_env.set_free_params(given_params=fixed_params)
        self.eval_env.update_free_params()
        self.eval_env.set_corpus(n_eval_sentence=n_eval_sentence, random_sample=random_sample)

        np.random.seed(datetime.now().microsecond)
        random.seed(datetime.now().microsecond)

        ep_rewards, ep_lengths, action_stat = self.eval_episodes(
            n_episodes=n_eval_sentence,
            return_action_stat=True,
            verbose=verbose
        )
        agg_df = pd.DataFrame(
            self.eval_env.sentence_test_data[1:],
            columns=self.eval_env.sentence_test_data[0]
        )
        outputs = np.array([
            agg_df["wpm"],
            agg_df["error.rate"],
            agg_df["bs"],
            agg_df["kspc"],
            agg_df["sentence.length"],
        ]).T

        if return_info:
            info = dict(
                ep_rewards=ep_rewards,
                ep_lengths=ep_lengths,
                action_stat=action_stat
            )
            return outputs, info
        else:
            return outputs  

    def eval_episodes(
        self,
        n_steps=None,
        n_episodes=None,
        max_episode_len=100,
        return_action_stat=False,
        verbose=False,
    ):
        assert (n_steps is not None) or (n_episodes is not None)
        assert self.eval_env is not None
        
        scores, steps = [], []
        terminate = False
        total_step = 0
        total_episode = 0
        reset = True
        actions = []

        while not terminate:
            if reset:
                obs = self.eval_env.reset()
                done = False
                score = 0
                episode_len = 0
                info = {}
            a = self.act(obs, deterministic=True)
            actions.append(a)
            if total_episode == n_episodes - 1:
                obs, r, done, info = self.eval_env.step(a, verbose=verbose)
            else:
                obs, r, done, info = self.eval_env.step(a)
            score += r
            episode_len += 1
            total_step += 1
            reset = (done or episode_len >= max_episode_len
                or info.get('need_reset', False))
            if reset:
                total_episode += 1
                scores.append(float(score))
                steps.append(float(episode_len))

            if n_steps is not None:
                terminate = (total_step >= n_steps)
            else:
                terminate = (total_episode >= n_episodes)

        if return_action_stat:
            action_arr = np.array(actions)
            action_stat = action_arr.reshape(-1, action_arr.shape[-1]).mean(axis=0)
            return scores, steps, action_stat
        else:
            return scores, steps

    def evaluate(self, sentence=None, batch=False, n_users=1, plot_hist=True, **kwargs):
        """
        Function to evaluate trained agent.
        :param sentence: sentence to type.
        :param batch: run evaluation in batch mode (several eval_sentences & several seeds).
        :param n_users: number of users to simulate (requires batch mode).
        """
        os.makedirs(self.output_path, exist_ok=True)
        eval_env = self.env_cls(self.layout_config, self.agent_params, False, self.variable_params, self.fixed_params)
        eval_env.set_corpus(n_eval_sentence=1500)
        if not (sentence == "" or sentence is None):
            eval_env.sentences = [sentence]
            eval_env.sentences_bkp = [sentence]
            eval_env.sentences_id = [0]
        max_episode_len = 100

        # Non-batch mode: evaluation with corpus & several seeds
        if batch:
            n_eval_sentence = len(eval_env.sentences)
            with tqdm.tqdm(total=n_eval_sentence * n_users) as pbar:
                steps, rewards = list(), list()
                for i in range(n_users):
                    eval_env.agent_id = i

                    # reinitialise random seed.
                    np.random.seed(datetime.now().microsecond)
                    random.seed(datetime.now().microsecond)

                    for _ in range(n_eval_sentence):
                        state = eval_env.reset()
                        complete = False
                        step, score = 0, 0
                        while not complete:
                            action = self.act(state)
                            state, reward, done, info = eval_env.step(action)
                            step += 1
                            score += reward
                            complete = (done or step >= max_episode_len
                                or info.get('need_reset', False))
                        pbar.update(1)
                        steps.append(step)
                        rewards.append(reward)
            
            print(f"\n[ Evaluation ] {len(steps)} trials")
            print(f" - average.ep_length {sum(steps) / len(steps)}")
            print(f" - average.ep_reward {sum(rewards) / len(rewards)}")

            with open(os.path.join(self.output_path, "SupervisorAgent_sentence_test.csv"), "w", newline="",
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.sentence_test_data)

            with open(os.path.join(self.output_path, "SupervisorAgent_Vision_Viz.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.eye_viz_log)

            with open(os.path.join(self.output_path, "SupervisorAgent_Finger_Viz.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.finger_viz_log)

            with open(os.path.join(self.output_path, "SupervisorAgent_Typing_Viz.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.typing_viz_log)

        # Non-batch mode: evaluation with ONE sentence
        else: 
            state = eval_env.reset()
            complete = False
            step = 0
            while not complete:
                action = self.act(state)
                state, reward, done, info = eval_env.step(action)
                complete = (done or step >= max_episode_len
                    or info.get('need_reset', False))

            with open(os.path.join(self.output_path, "SupervisorAgent_vision_test.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.eye_test_data)

            with open(os.path.join(self.output_path, "SupervisorAgent_finger_test.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.finger_test_data)

            with open(os.path.join(self.output_path, "SupervisorAgent_sentence_test.csv"), "w", newline="",
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(eval_env.sentence_test_data)

        self.save_senetence_agg_data(os.path.join(self.output_path, "SupervisorAgent_sentence_test.csv"))
        self.save_user_agg_data(os.path.join(self.output_path, "SupervisorAgent_sentence_test.csv"))
        if plot_hist:
            self.plot_histogram()

    def save_senetence_agg_data(self, filename):
        """
        generates sentence level aggregate data.
        :param filename: raw data file path.
        """
        data = pd.read_csv(filename, sep=',', encoding='utf-8')
        data = data.groupby("target.sentence").agg(['mean', 'std'])
        data.to_csv(os.path.join(self.output_path, "SupervisorAgent_sentence_aggregate.csv"), encoding='utf-8')

    def save_user_agg_data(self, filename):
        """
        generates user level aggregate data.
        :param filename: raw data file path.
        """
        data = pd.read_csv(filename, sep=',', encoding='utf-8')
        data = data.groupby("agent.id").agg(['mean', 'std'])
        data.to_csv(os.path.join(self.output_path, "SupervisorAgent_user_aggregate.csv"), encoding='utf-8')

    def plot_histogram(self, color="g", maxyticks=5, scalemax=None, dt=None, ax=None):
        columns = ["wpm", "iki", "error.rate", "bs", "kspc", "sentence.length"]
        figsize = (5, 2.5 * len(columns))
        data = pd.read_csv(os.path.join(self.output_path, "SupervisorAgent_sentence_test.csv"), sep=',', encoding='utf-8')

        fig = pl.figure(figsize=figsize)
        for idx, column in enumerate(columns):
            ax = pl.subplot(len(columns), 1, idx + 1)
            bins, bin_edges = np.histogram(data[column], density=True)

            width = bin_edges[1] - bin_edges[0]
            if ax is not None:
                ax.bar(bin_edges[:-1], bins, width=width, color=color, edgecolor="black", linewidth=1.0)
                ax.set_xlim(min(bin_edges) - width * 0.5, max(bin_edges) + width * 0.5)
            else:
                pl.bar(bin_edges[:-1], bins, width=width, color=color, edgecolor="black", linewidth=1.0)
                pl.xlim(min(bin_edges) - width * 0.5, max(bin_edges) + width * 0.5)

            if scalemax is None or dt is None:
                deltaticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
                yticks = None
                for dt in deltaticks:
                    if not max(bins) > 0.0:
                        break
                    yticks = np.arange(0, (int(max(bins) / dt) + 2) * dt, dt)
                    if len(yticks) <= maxyticks:
                        if ax is not None: ax.set_yticks(yticks)
                        else: pl.yticks(yticks)
                        break
            else:
                yticks = np.arange(0, scalemax + dt / 2.0, dt)
                if ax is not None: ax.set_yticks(yticks)
                else: pl.yticks(yticks)

            pl.title("{}\n(m={:.2f} std={:.2f})".format(column, np.mean(data[column]), np.std(data[column])))
            
        pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        fig.savefig(os.path.join(self.output_path, "histogram.pdf"))