import os.path
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import truncnorm
import warnings
warnings.simplefilter("ignore", UserWarning)

from ..configs import default_typing_config
from .typing.supervisor.supervisor_agent import SupervisorAgent


class TypingSimulator(object):
    """
    A simulator of touchscreen typing behavior
    Touchscreen typing model from CHI 2021 paper by Jokinen et al.
    """
    def __init__(self, config=None):
        if config is None:
            self.config = deepcopy(default_typing_config["simulator"])
        else:
            self.config = config
        if "seed" in self.config:
            self.seeding(self.config["seed"])
        else:
            self.seeding()

        config_path = os.path.join(Path(__file__).parent, "typing", "configs")
        with open(os.path.join(config_path, "config.yml"), "r") as file:
            config_file = yaml.load(file, Loader=yaml.FullLoader)
        with open(os.path.join(config_path, config_file["testing_config"]), "r") as file:
            test_config = yaml.load(file, Loader=yaml.FullLoader)

        self.agent = SupervisorAgent(
            config_file["device_config"],
            test_config,
            train=False,
            verbose=False,
            variable_params=self.config["variable_params"],
            fixed_params=self.config["base_params"],
            concat_layers=self.config["concat_layers"],
            embed_net_arch=self.config["embed_net_arch"],
        )
    
    def simulate(
        self,
        n_param=1,
        n_eval_sentence=1500,
        fixed_params=None,
        random_sample=False,
        return_info=False,
        verbose=False,
    ):
        """
        Simulates human behavior based on given (or sampled) free parameters

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - n_eval_sentence: no. of simulation (i.e., sentences) per parameter set
        - fixed_params: ndarray of free parameter (see below) sets to simulate
        =======
        Free params in menu search model (from CHI'21 paper)
        1) obs_prob:  uniform(min=0.0, max=1.0)
        2) who_alpha: trucnorm(mean=0.6, std=0.3, min=0.4, max=0.9)
        3) who_k:     trucnorm(mean=0.12, std=0.08, min=0.04, max=0.20)
        =======
        - random_sample: set random sentenes from corpus for evaluation (simulation)
        - return_info: return other details incl. episode rewards, lengths, stats, etc.

        Outputs:
        - param_sampled: free parameter sets that are used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - outputs_normalized: behavioral outputs by simulation with normalized values (see below)
            > ndarray with size ((n_param), (n_eval_sentence), (dim. of behavior output))
        =======
        Behavioral output (per each trial)
        1) WPM
        2) Error rate
        3) No. of backspacing
        4) KSPC
        5) Length of sentence
        =======
        """
        if fixed_params is None:
            param_sampled = self.sample_param_from_distr(n_param)
        elif len(np.array(fixed_params).shape) == 1:
            param_sampled = np.expand_dims(np.array(fixed_params), axis=0)
        else:
            param_sampled = np.array(fixed_params)

        outputs, infos = list(), list()
        loop = tqdm(range(param_sampled.shape[0])) if verbose else range(param_sampled.shape[0])
        for i in loop:
            output, info = self.agent.simulate(
                fixed_params=param_sampled[i],
                n_eval_sentence=n_eval_sentence,
                random_sample=random_sample,
                return_info=True,
            )
            outputs.append(output)
            infos.append(info)

        outputs_max = np.array([80, 20, 20, 2.0, 80])
        outputs_min = np.array([0, 0, 0, 0.5, 0])
        outputs_normalized = (np.array(outputs) - outputs_min) / (outputs_max - outputs_min) * 2 - 1
        
        if return_info:
            return param_sampled, outputs_normalized, infos
        else:
            return param_sampled, outputs_normalized

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