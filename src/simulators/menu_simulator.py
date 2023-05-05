import os
from pathlib import Path
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from scipy.stats import truncnorm

from ..configs import default_menu_config
from ..nets.dqn import EmbeddingDQN
from .menu.menu_env import MenuSearchEnv
from .menu.model_config import menu_qnet_config


class MenuSimulator(object):
    """
    A simulator of menu search behavior
    Menu search model from CHI 2017 paper by Kangasrääsiö et al.
    """
    def __init__(self, config=None):
        if config is None:
            self.config = deepcopy(default_menu_config["simulator"])
        else:
            self.config = config
        if "seed" in self.config:
            self.seeding(self.config["seed"])
        else:
            self.seeding()

        self.env = MenuSearchEnv(
            variant=self.config["variant"],
            variable_params=self.config["variable_params"],
            seed=self.seed,
        )
        qnet_config = deepcopy(menu_qnet_config)
        self.qnet = EmbeddingDQN(
            qnet_config,
            name=qnet_config["name"],
            task_name="menu",
            model_path=os.path.join(Path(__file__).parent, "menu", "models")
        )
        self.qnet.load()
    
    def simulate(
        self,
        n_param=1,
        sim_per_param=10000,
        fixed_params=None,
        return_behavior=False,
        verbose=False,
    ):
        """
        Simulates human behavior based on given (or sampled) free parameters

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - sim_per_param: no. of simulation per parameter set
        - fixed_params: ndarray of free parameter (see below) sets to simulate
        =======
        Free params in menu search model (from CHI'17 paper)
        1) f_dur : truncnorm(min=0.0, max=6.0, mean=3.0, std=1.0)
        2) d_sel : truncnorm(min=0.0, max=1.0, mean=0.3, std=0.3)
        3) p_rec : uniform(min=0.0, max=1.0)
        4) p_sem : uniform(min=0.0, max=1.0)
        =======
        - return_behavior: outputs behavior information for every trial in dictionary form as well

        Outputs:
        - param_sampled: free parameter sets that are used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - outputs: behavioral outputs by simulation (see below)
            > ndarray with size ((n_param), (dim. of behavior output))
        =======
        Behavioral output
        1) mean value of task completion time when target is presented
        2) std value of task completion time when target is presented
        3) mean value of task completion time when target is absent
        4) std value of task completion time when target is absent
        =======
        """
        if fixed_params is None:
            param_sampled = self.sample_param_from_distr(n_param)
        elif len(np.array(fixed_params).shape) == 1:
            param_sampled = np.expand_dims(np.array(fixed_params), axis=0)
        else:
            param_sampled = np.array(fixed_params)

        max_duration = 3000
        outputs, res_df_list = list(), list()
        loop = tqdm(range(param_sampled.shape[0])) if verbose else range(param_sampled.shape[0])
        for i in loop:
            res = self._simulate_trial(
                simulation_ep=sim_per_param,
                fixed_params=param_sampled[i],
            )
            self.env.clean()
            res_df = pd.DataFrame.from_dict(res["sessions"])

            output = self._get_data(res_df, use_summary=True)
            outputs.append(output / max_duration)
            
            if return_behavior:
                res_df_list.append(res_df)

        outputs = np.array(outputs)
        if sim_per_param == 1:
            outputs = np.squeeze(outputs, axis=1)

        if return_behavior:
            res_df = pd.concat(res_df_list)
            return param_sampled, outputs, self._get_behavior(res_df, maxlen=15, exclude_outlier=True)
        else:
            return param_sampled, outputs
        
    def _simulate_trial(self, simulation_ep, fixed_params=None, fixed_variant=None, verbose=False):
        self.qnet.eval()
        self.env.training = False
        self.env.start_logging()

        with tqdm(total=simulation_ep, desc=f" Ep", disable=not verbose) as progress:
            for _ in range(simulation_ep):
                state = self.env.reset(fixed_variant=fixed_variant, fixed_params=fixed_params)
                score = 0
                done = False
                while not done:
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action)    
                    score += reward
                    state = next_state
                progress.update(1)

        dataset = self.env.log
        self.env.end_logging()
        self.env.training = True
        self.qnet.train()
        return dataset
    
    def act(self, state):
        self.qnet.eval()
        q_values = self.qnet(state).detach().cpu().numpy()
        action = np.argmax(q_values)
        self.qnet.train()
        return action
        
    def _get_data(self, df, maxlen=None, exclude_outlier=False, use_summary=False, get_both=False):
        ret_array = list()
        for _, sess in df.iterrows():
            if exclude_outlier:
                if maxlen is not None and len(sess["action_duration"]) >= maxlen:
                    continue
                if "reward" in sess.keys() and sess["reward"][-1] < 0:
                    continue
            ret_array.append([sum(sess["action_duration"]), int(sess["target_present"])])

        pre_tct = [o[0] for o in ret_array if o[1] == 1]
        abs_tct = [o[0] for o in ret_array if o[1] == 0]
        if len(abs_tct) == 0:
            abs_tct = [0.,]
        summary = np.array([
            np.mean(pre_tct),
            np.std(pre_tct),
            np.mean(abs_tct),
            np.std(abs_tct)
        ])
        if get_both:
            return np.array(ret_array), summary
        elif use_summary:
            return summary
        else:
            return np.array(ret_array)

    def _get_behavior(self, df, maxlen=None, exclude_outlier=True):
        behavior_dict = dict(
            pre_tct = list(),
            pre_nfix = list(),
            abs_tct = list(),
            abs_nfix = list(),
        )
        for _, sess in df.iterrows():
            if maxlen is not None and len(sess["action_duration"]) >= maxlen and exclude_outlier:
                continue
            num_saccades = len(sess["action"])
            if sess["action"][-1] == 8: num_saccades -= 1

            if sess["target_present"]:
                behavior_dict["pre_tct"].append(sum(sess["action_duration"]))
                behavior_dict["pre_nfix"].append(num_saccades)
            else:
                behavior_dict["abs_tct"].append(sum(sess["action_duration"]))
                behavior_dict["abs_nfix"].append(num_saccades)
        return behavior_dict

    def sample_param_from_distr(self, n_param):
        param_dim = len(self.config["param_distr"]["distr"])
        param_distr = self.config["param_distr"]
        param_sampled = list()
        for i in range(param_dim):
            if param_distr["distr"][i] == "truncnorm":
                param_sampled.append(truncnorm.rvs(
                    (param_distr["minv"][i] - param_distr["mean"][i]) / param_distr["std"][i],
                    (param_distr["maxv"][i] - param_distr["mean"][i]) / param_distr["std"][i],
                    param_distr["mean"][i],
                    param_distr["std"][i],
                    size=n_param,
                ))
            else: # "uniform"
                param_sampled.append(np.random.uniform(
                    low=param_distr["minv"][i],
                    high=param_distr["maxv"][i],
                    size=n_param,
                ))
        return np.array(param_sampled).T

    def seeding(self, seed=121):
        ## Not implemented
        self.seed = seed