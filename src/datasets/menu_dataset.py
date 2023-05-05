import os, pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed

from ..simulators import MenuSimulator
from ..simulators.menu.observation import BaillyData
from ..configs import default_menu_config


class MenuUserDataset(object):
    """
    The class 'MenuUserDataset' downloads a dataset of user behavior related to menu search tasks.
    ==> From "Model of Visual Search and Selection Time in Linear Menus" (CHI 2014) by Bailly et al.

    It allows you to retrieve a sample of the data and manipulate it in different ways.
    """
    def __init__(self):
        """
        Initializes the class and retrieves the dataset.
        """

        self.fpath = os.path.join(Path(__file__).parent.parent.parent, "data", "menu", "datasets", "empirical_data.pkl")
        self._get_dataset()
        self.user_set = self.df["user_id"].unique()
        self.n_user = len(self.user_set)

    def _get_dataset(self):
        """
        Checks if the dataset file exists, loads it if it does, and downloads it if it doesn't.
        """
        if os.path.exists(self.fpath):
            self.df = pd.read_pickle(self.fpath)
        else:
            full_data = BaillyData(
                menu_type="Semantic",
                allowed_users=[],
                excluded_users=["S6"], # S6 has only 3 trials
                trials_per_user_present=9999,
                trials_per_user_absent=9999,
            ).get()
            self.df = pd.DataFrame.from_dict(full_data["sessions"])
            self.df.to_pickle(self.fpath)

    def _get_data(self, df, maxlen=None, exclude_outlier=False, use_summary=False):
        """
        Generates an array of the total duration of actions and whether the target was present for each session.
        It can return a summary version with mean and standard deviation or the complete data.
        """
        ret_array = list()
        for _, sess in df.iterrows():
            if exclude_outlier:
                if maxlen is not None and len(sess["action_duration"]) >= maxlen:
                    continue
                if "reward" in sess.keys() and sess["reward"][-1] < 0:
                    continue
            ret_array.append([sum(sess["action_duration"]), int(sess["target_present"])])

        if use_summary:
            pre_tct = [o[0] for o in ret_array if o[1] == 1]
            abs_tct = [o[0] for o in ret_array if o[1] == 0]
            assert len(abs_tct) > 0
            return np.array([
                np.mean(pre_tct),
                np.std(pre_tct),
                np.mean(abs_tct),
                np.std(abs_tct)
            ])
        else:
            return np.array(ret_array)

    def _get_behavior(self, df, maxlen=None, exclude_outlier=True):
        """
        Calculates the behavior dictionary containing the task completion time (_tct)
        and the number of fixations (_nfix) for both target-present and target-absent trials.
        """
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

    def sample(self, cross_valid=False):
        """
        Returns a sample of the processed data (a summary version with normalized).
        Optionally, it can also return the behavior for validation users if cross_valid is set to True.
        """
        # Use the train/valid dataset split following CHI'17 paper, 
        train_user = ["S22", "S6", "S41", "S7", "S5", "S8", "S20", "S36", "S24"]
        train_df = self.df.query("user_id in @train_user")
        if cross_valid:
            valid_df = self.df.query("user_id not in @train_user")

        output = self._get_data(train_df, use_summary=True)
        max_duration = 3000
        output_norm = output / max_duration

        if cross_valid:
            return output_norm, self._get_behavior(valid_df, maxlen=15, exclude_outlier=True)
        else:
            return output_norm
            
    def indiv_sample(self, cross_valid=False, for_pop=False):
        """
        Returns a set of individual samples for each user in the dataset.
        It also can return data with aggregated across users if for_pop is set to True.
        Additionally, it can return validation data if cross_valid is set to True.
        """
        outputs, valid_df_list = list(), list()
        max_duration = 3000
        train_df_list = list()
        
        for user in self.user_set:
            user_df = self.df.query("user_id == @user")
            pre_user_df = user_df.query("target_present == True")
            abs_user_df = user_df.query("target_present == False")

            if cross_valid:
                pre_n_trial = pre_user_df.shape[0] // 2
                abs_n_trial = abs_user_df.shape[0] // 2
            else:
                pre_n_trial = pre_user_df.shape[0]
                abs_n_trial = abs_user_df.shape[0]
            
            train_pre_user_df = pre_user_df.iloc[:pre_n_trial]
            train_abs_user_df = abs_user_df.iloc[:abs_n_trial]
            train_df_list.extend([train_pre_user_df, train_abs_user_df])
            train_df = pd.concat((train_pre_user_df, train_abs_user_df))

            if cross_valid:
                valid_pre_user_df = pre_user_df.iloc[pre_n_trial:]
                valid_abs_user_df = abs_user_df.iloc[abs_n_trial:]
                valid_df_list.append([valid_pre_user_df, valid_abs_user_df])

            indiv_output = self._get_data(train_df, use_summary=True)
            outputs.append(indiv_output / max_duration)
            
        if for_pop:
            train_df = pd.concat(train_df_list)
            output = self._get_data(train_df, use_summary=True)
            output_for_pop = output / max_duration

        if cross_valid:
            behavior_list = [
                self._get_behavior(pd.concat(d), maxlen=15, exclude_outlier=True)
                for d in valid_df_list
            ]
            if for_pop:
                return output_for_pop, behavior_list
            else:
                return outputs, behavior_list
        else:
            if for_pop:
                return output_for_pop
            else:
                return outputs


class MenuTrainDataset(object):
    """
    The class 'MenuTrainDataset' handles the creation and retrieval of a training dataset with menu search simulator.
    It allows you to sample data from the dataset and supports various configurations.
    """
    def __init__(self, total_sim=2**26, n_ep=256, sim_config=None):
        """
        Initializes with the total number of simulations, number of episodes, simulator configuration. 
        """
        if sim_config is None:
            self.sim_config = deepcopy(default_menu_config["simulator"])
        else:
            self.sim_config = sim_config

        self.sim_config["seed"] = 100
        self.total_sim = total_sim
        self.n_ep = n_ep
        self.n_param = total_sim // n_ep

        self.name = f"{self.total_sim//1000000}M_step_{self.n_ep}ep"
        if self.sim_config["use_uniform"]:
            self.name += "_uniform"

        self.fpath = os.path.join(Path(__file__).parent.parent.parent, "data", "menu", "datasets", f"train_{self.name}.pkl")

        self._get_dataset()

    def _get_dataset(self):
        """
        Checks if the dataset file exists, loads it if it does, and creates it if it doesn't.
        The creation process involves running simulations with the specified configuration
        and saving the dataset as a dictionary with parameters and summary stats.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ train dataset ] {self.fpath}")
            self.simulator = MenuSimulator(self.sim_config)

            def get_simul_res(simulator, i):
                """
                Runs simulations using the MenuSimulator object and returns the simulation results.
                Used in parallel processing to speed up the dataset creation process.
                """
                args = simulator.simulate(
                    1,
                    self.n_ep,
                    verbose=False,
                )
                return args

            num_cpus = psutil.cpu_count(logical=False)
            eps = Parallel(n_jobs=num_cpus - 1)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.n_param))
            )
            params_arr = np.concatenate([eps[i][0] for i in range(self.n_param)], axis=0)
            summary_arr = np.concatenate([eps[i][1] for i in range(self.n_param)], axis=0)

            self.dataset = dict(
                params=params_arr,
                summary_stats=summary_arr,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, batch_sz):
        """
        Returns a random sample from the dataset with the specified batch size (number of parameter sets)
        """
        indices = np.random.choice(self.n_param, batch_sz)
        return (
            np.array(self.dataset["params"][indices], dtype=np.float32),
            np.array(self.dataset["summary_stats"][indices, :], dtype=np.float32),
        )

 
class MenuValidDataset(object):
    """
    The class 'MenuValidDataset' handles the creation and retrieval of a validation dataset.
    """
    def __init__(self, n_param=200, sim_per_param=5000):
        """
        Initializes with the total number of parameter sets (n_param) and number of episodes (sim_per_param). 
        """
        self.n_param = n_param
        self.sim_per_param = sim_per_param
        self.sim_config = deepcopy(default_menu_config["simulator"])
        self.sim_config["seed"] = 121

        self.name = f"{self.n_param}_param_{self.sim_per_param}ep"
        if self.sim_config["use_uniform"]:
            self.name += "_uniform"
            
        self.fpath = os.path.join(Path(__file__).parent.parent.parent, "data", "menu", "datasets", f"valid_{self.name}.pkl")
        self._get_dataset()

    def _get_dataset(self):
        """
        Checks if the dataset file exists, loads it if it does, and creates it if it doesn't.
        The creation process involves running simulations with the specified configuration
        and saving the dataset as a dictionary with parameters and summary stats.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ valid dataset ] {self.fpath}")
            simulator = MenuSimulator(self.sim_config)
            args = simulator.simulate(
                self.n_param,
                self.sim_per_param,
                verbose=True,
            )
            self.dataset = dict(
                params=args[0],
                summary_stats=args[1],
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, n_user=None):
        """
        Returns a sample from the dataset with the specified number of users (sampled parameters and stats).
        If the number of users is not specified, it defaults to the total number of parameters in the dataset.
        """
        if n_user is None:
            n_user = self.n_param
        params = list()
        stats = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats.append(self.dataset["summary_stats"][user, :])
        return np.array(params, dtype=np.float32), np.array(stats, dtype=np.float32)