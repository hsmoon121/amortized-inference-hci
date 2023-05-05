"""
An implementation of "Menu search model"
from Kangasrääsiö et al., "Inferring cognitive models from data using approximate Bayesian computation." CHI 2017.

Original code: https://github.com/akangasr/cogsciabc
"""

import os
import numpy as np
import csv
import urllib3
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


class BaillyData():
    # Empirical dataset from "Model of Visual Search and Selection Time in Linear Menus" (CHI 2014) by Bailly et al.
    data_source = "http://gillesbailly.fr/exchange/eye-tracker/samples-cleaned.csv"

    def __init__(
            self,
            menu_type="Semantic",  # alt: "Unordered", "Alphabetic"
            allowed_users=[],
            excluded_users=[],
            trials_per_user_present=1,
            trials_per_user_absent=1
        ):
        self.menu_type = menu_type
        self.allowed_users = allowed_users
        self.excluded_users = excluded_users
        self.trials_per_user_present = trials_per_user_present
        self.trials_per_user_absent = trials_per_user_absent
        self.data = None

    def get(self):
        if self.data is not None:
            return self.data
        logger.info("Loading Bailly dataset..")
        loc_dir = os.path.dirname(os.path.realpath(__file__))
        data_target = "{}/materials/bailly.csv".format(loc_dir)
        if not os.path.isfile(data_target):
            logger.info("Dataset not present at {}".format(data_target))
            self._fetch_dataset(self.data_source, data_target)
        logger.info("Processing Bailly dataset..")
        self.data = self._process_csv(data_target)
        self._filter_data()
        logger.info("Processing done")
        return self.data

    @staticmethod
    def _fetch_dataset(data_source, data_target):
        """ Downloads a file over HTTP to local disc.
        """
        logger.info("Downloading {} to {}".format(data_source, data_target))
        http = urllib3.PoolManager()
        chunk_size = 1024
        r = http.request('GET', data_source, preload_content=False)
        with open(data_target, 'wb') as output:
            while True:
                data = r.read(chunk_size)
                if not data:
                    break
                output.write(data)
        r.release_conn()
        logger.info("Download done")

    def _add_end_step(self, gaze, target, list_len):
        """ Add final step to log
        """
        # assume user did not make mistakes
        if target == None:
            # missing item
            action = list_len # quit
            self.step_data["observation"].append(None)
            self.step_data["action"].append(action)
            self.step_data["reward"].append(0)
            self.step_data["gaze_location"].append(gaze)
            self.step_data["duration_focus_ms"].append(0)
            self.step_data["duration_saccade_ms"].append(0)
            self.step_data["action_duration"].append(0)

    def _add_look_step(self, gaze, saccade_ms, focus_ms):
        """ Add look step to log
        """
        self.step_data["observation"].append(None)
        self.step_data["action"].append(gaze)
        self.step_data["reward"].append(0)
        self.step_data["gaze_location"].append(gaze)
        self.step_data["duration_focus_ms"].append(focus_ms)
        self.step_data["duration_saccade_ms"].append(saccade_ms)
        self.step_data["action_duration"].append(focus_ms + saccade_ms)

    def _get_gaze(self, gazepoint, list_length, nearest=False):
        """ Return the current item user is looking at.
            If 'nearest' is True, return nearest item instead of None when
            not focused at any set item.
        """
        start = 3
        height = 19
        separator = 9
        current_gaze = None
        for gaze in range(list_length):
            if gaze + 1 > list_length / 2:
                sep = separator
            else:
                sep = 0
            low = start + height * gaze + sep
            high = low + height
            if gazepoint >= low and gazepoint < high:
                current_gaze = gaze
        if current_gaze == None and nearest == True:
            if gazepoint < start:
                return 0
            elif gazepoint >= start + height * list_length + sep:
                return list_length - 1
            elif gazepoint < start + height * (list_length / 2) + (sep / 2):
                return list_length / 2 - 1
            else:
                return list_length / 2
        return current_gaze

    def _parse_steps(self, fixationy, gazelength, target, list_length, user):
        """ Parse steps of one session and write them to self.log

        Parameters
        ----------
        fixationy : list of fixation y-coordinates
        gazelength : eye-tracker resolution in ms (difference in time between fixation points)
        target : target location (index)
        list_length : number of items in list
        user : user id
        """
        total_length_ms = len(fixationy) * gazelength
        gaze_items = [self._get_gaze(g, list_length, nearest=True) for g in fixationy]
        gaze_sections = list()
        gaze_sections.append(list())
        gaze_sections[0].append((0, gaze_items[0]))
        for i in range(1,len(gaze_items)):
            if gaze_items[i-1] != gaze_items[i]:
                gaze_sections.append(list())
            gaze_sections[-1].append((i, gaze_items[i]))
        if len(gaze_sections) == 0:
            raise Exception("No valid gaze sections")

        if "session" not in self.log:
            self.log["session"] = 0
            self.log["sessions"] = [dict()]
        else:
            self.log["session"] += 1
            self.log["sessions"].append(dict())
        self.step_data = self.log["sessions"][self.log["session"]]
        self.step_data["target_idx"]    = target
        self.step_data["user_id"]       = user
        self.step_data["items"]         = None
        self.step_data["observation"]   = list()
        self.step_data["action"]        = list()
        self.step_data["reward"]        = list()
        self.step_data["gaze_location"] = list()
        self.step_data["duration_focus_ms"]   = list()
        self.step_data["duration_saccade_ms"] = list()
        self.step_data["action_duration"]     = list()
        self.step_data["target_present"]      = (target != None)

        gaze = None
        logger.debug("Episode:")
        ms_spent = 0
        for i in range(len(gaze_sections)):
            saccade_length_ms = gazelength  # we have to assume instantaneous saccades
            focus_length_ms   = gazelength * (gaze_sections[i][-1][0] - gaze_sections[i][0][0])
            ms_spent += saccade_length_ms + focus_length_ms
            gaze = gaze_sections[i][0][1]
            self._add_look_step(gaze, saccade_length_ms, focus_length_ms)
            logger.debug("Saccade {} ms, focus {} ms, location {}".format(saccade_length_ms, focus_length_ms, gaze))
        assert ms_spent == total_length_ms
        self._add_end_step(gaze, target, list_length)

        logger.debug("Logged session")

    def _target_location(self, row, length):
        """ Returns the target location or None if target is not present """
        target_loc = int(row["item"]) - 1 # index starts at 0 whereas in log starts from 1
        if target_loc < 0 or target_loc >= length:
            target_loc = None
        return target_loc

    def _process_csv(self, filename):
        """ Processes bailly.csv to dict
        """
        logger.debug("Processing {}".format(filename))
        self.log = dict()
        length = 8
        users_target_present = defaultdict(list)
        users_target_absent = defaultdict(list)
        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            last_target = "init"
            gazes = list()
            gazelength = 20
            for row in reader:
                valid_line = row["organization"] == self.menu_type \
                        and int(row["length"]) == length
                if not valid_line:
                    continue

                user = row["user"]
                target_loc = self._target_location(row, length)
                if target_loc is not None:
                    user_trials = users_target_present[user]
                else:
                    user_trials = users_target_absent[user]
                current_trial_id = int(row["trial_id"])
                if current_trial_id not in user_trials:
                    if last_target != "init":
                        if len(gazes) == 0:
                            logger.warning("No gazes in this session? Skipping.")
                        else:
                            try:
                                logger.debug("Adding user '{}' trial id '{}' (target: {})..".format(last_user, last_trial_id, last_target))
                                self._parse_steps(gazes, gazelength, last_target, length, last_user)
                            except Exception as e:
                                if last_target != None:
                                    users_target_present[last_user].remove(last_trial_id)
                                else:
                                    users_target_absent[last_user].remove(last_trial_id)
                                logger.warning("Not able to parse steps! Skipping this session. (Error: {})".format(e))
                    user_trials.append(current_trial_id)
                    if target_loc != None:
                        users_target_present[user] = user_trials
                    else:
                        users_target_absent[user] = user_trials
                    last_user = user
                    last_trial_id = current_trial_id
                    last_target = target_loc
                    gazes = list()
                    step = 0

                # use pre-parsed fixations
                gazes.append(int(row["fixationy"]))

            # final step
            try:
                logger.debug("Adding user '{}' trial id '{}' (target: {})..".format(user, current_trial_id, target_loc))
                self._parse_steps(gazes, gazelength, target_loc, length, user)
            except Exception as e:
                if last_target != None:
                    users_target_present[last_user].remove(last_trial_id)
                else:
                    users_target_absent[last_user].remove(last_trial_id)
                logger.warning("Not able to parse steps! Skipping this session. (Error: {})".format(e))
        all_users = list(set(users_target_present.keys()).union(set(users_target_absent.keys())))
        all_users.sort()
        logger.debug("Sessions recorded per user:")
        n_present_total = 0
        n_absent_total = 0
        for user in all_users:
            n_target_present = len(users_target_present[user])
            n_target_absent  = len(users_target_absent[user])
            logger.debug("User {}: {} in 'target present' condition, {} in 'target absent' condition".format(user, n_target_present, n_target_absent))
            n_present_total += n_target_present
            n_absent_total += n_target_absent
        logger.debug("Total: {} in 'target present' condition, {} in 'target absent' condition".format(n_present_total, n_absent_total))

        return self.log

    def _filter_data(self):
        """ Filters dataset to only leave required sessions
        """
        filt_data = {"session": 0, "sessions": list()}
        current_trials_per_user_present = defaultdict(list)
        current_trials_per_user_absent = defaultdict(list)
        n_current_trials_per_user_present = defaultdict(int)
        n_current_trials_per_user_absent = defaultdict(int)
        all_users = list()
        for session in self.data["sessions"]:
            if session["user_id"] in self.excluded_users:
                continue
            if len(self.allowed_users) > 0 and session["user_id"] not in self.allowed_users:
                continue

            if session["target_present"] is True:
                current_trials_per_user_present[session["user_id"]].append(session)
            else:
                current_trials_per_user_absent[session["user_id"]].append(session)

        for user_id, sessions in current_trials_per_user_present.items():
            if len(sessions) > self.trials_per_user_present:
                sessions = sessions[:self.trials_per_user_present]
            n_current_trials_per_user_present[user_id] = len(sessions)
            filt_data["session"] += len(sessions)
            filt_data["sessions"].extend(sessions)
            if user_id not in all_users:
                all_users.append(user_id)

        for user_id, sessions in current_trials_per_user_absent.items():
            if len(sessions) > self.trials_per_user_absent:
                sessions = sessions[:self.trials_per_user_present]
            n_current_trials_per_user_absent[user_id] = len(sessions)
            filt_data["session"] += len(sessions)
            filt_data["sessions"].extend(sessions)
            if user_id not in all_users:
                all_users.append(user_id)

        logger.info("Observed sessions:")
        for user in all_users:
            logger.info("User {}: {} target present, {} target absent (from start)"
                .format(user,
                n_current_trials_per_user_present[user],
                n_current_trials_per_user_absent[user]))

        self.data = filt_data

