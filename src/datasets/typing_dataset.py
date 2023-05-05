import os, pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed

from ..simulators import TypingSimulator
from ..configs import default_typing_config


class TypingUserDataset(object):
    """
    The class 'TypingUserDataset' handles the creation and retrieval of an empirical dataset for touchscreen typing tasks.
    ==> From "How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers"
        (MobileHCI 2019) by Palin et al.
    ==> https://userinterfaces.aalto.fi/typing37k/data/csv_raw_and_processed.zip
    """
    def __init__(self, min_trials=11):
        self.min_trials = min_trials
        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/typing/datasets/empirical_data_trial_min{self.min_trials}.csv"
        )
        self.p_fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/typing/datasets/empirical_data_info_min{self.min_trials}.csv"
        )
        self._get_dataset()
        self.n_user = len(self.df["PARTICIPANT_ID"].unique())

    def _get_dataset(self):
        """
        Retrieve the dataset if it exists, otherwise it processes and creates the dataset from the raw data files.
        Use empirical data where participants used "english" & "qwerty" keyboard and typed with one thumb.
        Also, only data from people with more than specific trials (min_trials) were used.
        """
        if os.path.exists(self.fpath) and os.path.exists(self.p_fpath):
            self.df = pd.read_csv(self.fpath, index_col=0)
            self.p_df = pd.read_csv(self.p_fpath, index_col=0)
        else:
            p_header = pd.read_csv(
                "./data/typing/datasets/full_data/open_participants_header.csv",
                header=0,
                encoding="utf8"
            ).to_numpy()[:, 0]
            p_header_wo_lang = list(p_header[:1]) + list(p_header[3:])

            # Total participants: 37606
            p_lang = pd.read_csv(
                "./data/typing/datasets/full_data/open_participants_lang.csv",
                header=None,
                names=p_header[:3],
                encoding="utf8",
                index_col=False,
                sep=r'(?<=\d),|(?<="),',
                engine="python",
            )
            p = pd.read_csv(
                "./data/typing/datasets/full_data/open_participants_wo_lang.csv",
                header=None,
                names=p_header_wo_lang,
                encoding="utf8",
                on_bad_lines="skip",
                index_col=False,
                sep=r'(?<!\\""")(?<!Middle),',
                engine="python",
            )

            # Participants who use "en" keyboard: 35692
            en_p_list = p_lang[p_lang["BROWSER_LANGUAGE"].astype(str).str.contains("en")]["PARTICIPANT_ID"].to_numpy()

            # Participants with "en" / "qwerty" / "one thumb": 1057
            valid_layout = ["qwerty"]
            self.p_df = (p
                .query("LAYOUT in @valid_layout")
                .query("~FINGERS.str.contains('both')")
                .query("~FINGERS.str.contains('undefined')")
                .query("PARTICIPANT_ID in @en_p_list")
            )
            self.p_df.to_csv(self.p_fpath)
            valid_p_list = self.p_df["PARTICIPANT_ID"].to_numpy().astype(int)

            trials_header = pd.read_csv(
                "./data/typing/datasets/full_data/open_test_sections_header.csv",
                header=0,
                encoding="utf8"
            )
            trials = pd.read_csv(
                "./data/typing/datasets/full_data/open_test_sections.csv",
                header=None,
                names=trials_header.to_numpy()[:, 0],
                encoding="utf8",
                index_col=False,
                sep=",",
                engine="python",
            )

            valid_trials = (trials
                .query('PARTICIPANT_ID in @valid_p_list')
                .query('PR_SWYP == "0"')
                .query('PR_PRED == "0"')
                .query('PR_AUTO == "0"')
            )
            valid_ite_p_count = \
                valid_trials["PARTICIPANT_ID"].value_counts()
            valid_ite_p_list = valid_ite_p_count[
                valid_ite_p_count >= self.min_trials
            ].index.to_numpy().astype(int)

            self.df = valid_trials.query("PARTICIPANT_ID in @valid_ite_p_list")
            self.df.to_csv(self.fpath)

    def sample(self, n_user=None, cross_valid=False):
        """
        Sample the dataset for a given number of users (n_user),
        and optionally splits the data into training and validation sets if cross_valid is True.
        """
        if n_user is None:
            if cross_valid:
                n_user = self.n_user // 2
            else:
                n_user = self.n_user
        outputs = list()
        outputs_max = np.array([80, 20, 20, 2.0, 80])
        outputs_min = np.array([0, 0, 0, 0.5, 0])

        sampled_p = np.random.choice(self.df["PARTICIPANT_ID"].unique(), size=n_user, replace=False)
        train_p_stat = self.df.query("PARTICIPANT_ID in @sampled_p")
        if cross_valid:
            valid_p_stat = self.df.query("PARTICIPANT_ID not in @sampled_p")
        stat_arr = [train_p_stat, valid_p_stat] if cross_valid else [train_p_stat,]

        for p_stat in stat_arr:
            output = np.array([
                p_stat["WPM"],
                p_stat["ERROR_RATE"],
                p_stat["TS_BSP"],
                p_stat["TS_KSPC"],
                p_stat["TS_UILEN"],
            ]).T
            outputs.append((output - outputs_min) / (outputs_max - outputs_min) * 2 - 1)
        
        if cross_valid:
            return outputs[0], outputs[1]
        else:
            return outputs[0]

    def indiv_sample(self, n_user=None, cross_valid=False, for_pop=False):
        """
        Sample the dataset for individual users with a given number of users (n_user),
        and optionally splits the data into training and validation sets if cross_valid is True.
        The for_pop parameter, if set to True, concatenates the outputs for all users.
        """
        if n_user is None:
            n_user = self.n_user
        outputs, valid_outputs, user_info = list(), list(), list()
        outputs_max = np.array([80, 20, 20, 2.0, 80])
        outputs_min = np.array([0, 0, 0, 0.5, 0])

        sampled_p = np.random.choice(self.df["PARTICIPANT_ID"].unique(), size=n_user, replace=False)
        for i in range(n_user):
            target_id = sampled_p[i]
            p_stat = self.df.query("PARTICIPANT_ID == @target_id")
            if cross_valid:
                n_trial = p_stat.shape[0] // 2
            else:
                n_trial = p_stat.shape[0]

            train_p_stat = p_stat.iloc[:n_trial]
            valid_p_stat = p_stat.iloc[n_trial:]

            indiv_output = np.array([
                train_p_stat["WPM"],
                train_p_stat["ERROR_RATE"],
                train_p_stat["TS_BSP"],
                train_p_stat["TS_KSPC"],
                train_p_stat["TS_UILEN"],
            ]).T
            outputs.append((indiv_output - outputs_min) / (outputs_max - outputs_min) * 2 - 1)

            if cross_valid:
                valid_indiv_output = np.array([
                    valid_p_stat["WPM"],
                    valid_p_stat["ERROR_RATE"],
                    valid_p_stat["TS_BSP"],
                    valid_p_stat["TS_KSPC"],
                    valid_p_stat["TS_UILEN"],
                ]).T
                valid_outputs.append((valid_indiv_output - outputs_min) / (outputs_max - outputs_min) * 2 - 1)

            user_info.append(dict(
                id=target_id,
                age=self.p_df.query("PARTICIPANT_ID == @target_id")["AGE"].values.astype(int)[0],
                gender=self.p_df.query("PARTICIPANT_ID == @target_id")["GENDER"].values.astype(str)[0],
            ))

        if for_pop:
            outputs = np.concatenate(outputs, axis=0)
        if cross_valid:
            valid_outputs = np.concatenate(valid_outputs, axis=0)
            return outputs, valid_outputs, user_info
        else:
            return outputs, user_info


class TypingTrainDataset(object):
    """
    The class 'TypingTrainDataset' handles the creation and retrieval of a training dataset with touchscreen typing simulator.
    It samples data from the dataset and supports various configurations.
    """
    def __init__(self, total_sim=2**20, n_ep=16, sim_config=None):
        """
        Initialize with the total number of simulations, number of episodes, simulator configuration. 
        """
        if sim_config is None:
            self.sim_config = deepcopy(default_typing_config["simulator"])
        else:
            self.sim_config = sim_config
        self.sim_config["seed"] = 100
        self.total_sim = total_sim
        self.n_ep = n_ep
        self.n_param = total_sim // n_ep
        self.name = f"{self.total_sim//1000000}M_step_{self.n_ep}ep"
        if self.sim_config["use_uniform"]:
            self.name += "_uniform"

        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/typing/datasets/train_{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the PnCSimulator.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ train dataset ] {self.fpath}")
            self.simulator = TypingSimulator(self.sim_config)

            def get_simul_res(simulator, i):
                args = simulator.simulate(
                    1,
                    self.n_ep,
                    verbose=False
                )
                return args

            num_cpus = psutil.cpu_count(logical=False)
            eps = Parallel(n_jobs=num_cpus - 1)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.n_param))
            )
            params_arr = np.concatenate([eps[i][0] for i in range(self.n_param)], axis=0)
            stats_arr = np.concatenate([eps[i][1] for i in range(self.n_param)], axis=0)

            self.dataset = dict(
                params=params_arr,
                stats=stats_arr,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, batch_sz, sim_per_param=1):
        """
        Returns a random sample from the dataset with the specified number of parameter sets (batch size),
        and number of simulated trials per parameters (sim_per_param).
        """
        indices = np.random.choice(self.n_param, batch_sz)
        ep_indices = np.random.choice(self.n_ep, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        if sim_per_param == 1:
            return (
                np.array(self.dataset["params"][indices], dtype=np.float32),
                np.array(self.dataset["stats"][rows, cols].squeeze(1), dtype=np.float32),
            )
        else:
            return (
                np.array(self.dataset["params"][indices], dtype=np.float32),
                np.array(self.dataset["stats"][rows, cols], dtype=np.float32),
            )
            

class TypingValidDataset(object):
    """
    The class 'TypingValidDataset' handles the creation and retrieval of a validation dataset with touchscreen typing simulator.
    """
    def __init__(self, n_param=100, sim_per_param=500):
        """
        Initialize the dataset object with a specified number of total user (different parameter sets), episodes,
        and a simulation configuration.
        """
        self.n_param = n_param
        self.sim_per_param = sim_per_param
        self.sim_config = deepcopy(default_typing_config["simulator"])
        self.sim_config["seed"] = 121
        
        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/typing/datasets/valid_{self.n_param}_param_{self.sim_per_param}ep.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the PnCSimulator.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ valid dataset ] {self.fpath}")
            self.simulator = TypingSimulator(self.sim_config)

            def get_simul_res(simulator, i):
                args = simulator.simulate(
                    1,
                    self.sim_per_param,
                    verbose=False
                )
                return args

            num_cpus = psutil.cpu_count(logical=False)
            eps = Parallel(n_jobs=num_cpus - 1)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.n_param))
            )
            params_arr = np.concatenate([eps[i][0] for i in range(self.n_param)], axis=0)
            stats_arr = np.concatenate([eps[i][1] for i in range(self.n_param)], axis=0)

            self.dataset = dict(
                params = params_arr,
                stats = stats_arr,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, n_trial, n_user=None):
        """
        Return a sample from the dataset with the given number of trials and users.
        If the number of users is not specified, it defaults to the total number of parameters in the dataset.
        """
        if n_user is None:
            n_user = self.n_param
        params = list()
        stats = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats.append(self.dataset["stats"][user][:n_trial])
        return np.array(params, dtype=np.float32), np.array(stats, dtype=np.float32)