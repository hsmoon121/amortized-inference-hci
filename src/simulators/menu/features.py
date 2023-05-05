"""
An implementation of "Menu search model"
from Kangasrääsiö et al., "Inferring cognitive models from data using approximate Bayesian computation." CHI 2017.

Original code: https://github.com/akangasr/cogsciabc
"""

import numpy as np


def get_feature_set(data):
    return {
        "00_task_completion_time": get_value(data, get_task_completion_time, target_present=None),
        "01_task_completion_time_target_absent": get_value(data, get_task_completion_time, target_present=False),
        "02_task_completion_time_target_present": get_value(data, get_task_completion_time, target_present=True),
        "03_fixation_duration_target_absent": get_list(data, get_fixation_durations, target_present=False),
        "04_fixation_duration_target_present": get_list(data, get_fixation_durations, target_present=True),
        "05_saccade_duration_target_absent": get_list(data, get_saccade_durations, target_present=False),
        "06_saccade_duration_target_present": get_list(data, get_saccade_durations, target_present=True),
        "07_number_of_saccades_target_absent": get_value(data, get_number_of_saccades, target_present=False),
        "08_number_of_saccades_target_present": get_value(data, get_number_of_saccades, target_present=True),
        "09_fixation_locations_target_absent": get_list(data, get_fixation_locations, target_present=False),
        "10_fixation_locations_target_present": get_list(data, get_fixation_locations, target_present=True),
        "11_length_of_skips_target_absent": get_list(data, get_length_of_skips, target_present=False),
        "12_length_of_skips_target_present": get_list(data, get_length_of_skips, target_present=True),
        "13_location_of_gaze_to_target": get_list(data, get_location_of_gaze_to_target, target_present=True),
        "14_proportion_of_gaze_to_target": get_value(data, get_proportion_of_gaze_to_target, target_present=True),
    }

def get_list(data, extr_fun, target_present):
    ret = list()
    for session in data["sessions"]:
        if is_valid_condition(session, target_present) == True:
            ret.extend(extr_fun(session))
    return ret

def get_value(data, extr_fun, target_present):
    return [
        extr_fun(session)
        for session in data["sessions"]
        if is_valid_condition(session, target_present) == True
    ]

def get_task_completion_time(session):
    return sum(session["duration_saccade_ms"]) + sum(session["duration_focus_ms"])

def get_location_of_gaze_to_target(session):
    ret = list()
    for gaze_location in session["gaze_location"]:
        if gaze_location == session["target_idx"]:
            ret.append(gaze_location)
    return ret

def get_proportion_of_gaze_to_target(session):
    n_gazes_to_target = 0
    n_gazes = 0
    if len(session["gaze_location"]) < 1:
        return (session["target_idx"], 0)
    for gaze_location in session["gaze_location"]:
        n_gazes += 1
        if gaze_location == session["target_idx"]:
            n_gazes_to_target += 1
    if n_gazes == 0:
        return (session["target_idx"], 0)
    return (session["target_idx"], float(n_gazes_to_target) / n_gazes)

def get_fixation_durations(session):
    if session["action"][-1] == 8:
        return session["duration_focus_ms"][:-1]
    return session["duration_focus_ms"]

def get_length_of_skips(session):
    if session["action"][-1] == 8:
        locs = session["gaze_location"][:-1]
    else:
        locs = session["gaze_location"]
    if len(locs) < 2:
        return list()
    ret = list()
    prev_loc = locs[0]
    for i in range(1, len(locs)-1):
        cur_loc = locs[i]
        ret.append(abs(cur_loc - prev_loc))
        prev_loc = cur_loc
    return ret

def get_saccade_durations(session):
    if session["action"][-1] == 8:
        return session["duration_saccade_ms"][:-1]
    return session["duration_saccade_ms"]

def get_number_of_saccades(session):
    if session["action"][-1] == 8:
        return len(session["action"]) - 1
    return len(session["action"])

def get_fixation_locations(session):
    if session["action"][-1] == 8:
        return session["action"][:-1]
    return session["action"]

def is_valid_condition(session, target_present):
    if target_present is None:
        # None -> everything is ok
        return True
    if target_present == session["target_present"]:
        return True
    return False

def calculate(data):
    d = get_feature_set(data)
    vals = list()
    distr = dict()
    for k in d.keys():
        f = d.get(k)
        if "task_completion_time" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 3000.0
            nbins = 8
        elif "location_of_gaze_to_target" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "proportion_of_gaze_to_target" in k:
            feature_type = "graph"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "fixation_duration" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 1000.0
            nbins = 10
        elif "saccade_duration" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 150.0
            nbins = 10
        elif "number_of_saccades" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 14.0
            nbins = 15
        elif "fixation_locations" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        elif "length_of_skips" in k:
            feature_type = "histogram"
            minbin = 0.0
            maxbin = 7.0
            nbins = 8
        else:
            raise ValueError("Unknown feature: %s" % (k))

        bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
        if feature_type == "histogram":
            fout = [fi if fi < maxbin else maxbin+1e-10 for fi in f]
            h, e = np.histogram(fout, bins=bins)
            hnorm = h / sum(h)
        elif feature_type == "graph":
            hh, e = np.histogram(list(), bins=bins)
            hr = [0] * len(hh)
            n = [0] * len(hh)
            # assume minbin == 0, increment == 1
            for fi in f:
                hr[fi[0]] += fi[1]
                n[fi[0]] += 1
            h = list()
            for i in range(len(hr)):
                if n[i] == 0:
                    h.append(0)
                else:
                    h.append(hr[i] / float(n[i]))
            hnorm = h

        distr[k] = {
                "feature_type": feature_type,
                "f": f,
                "h": h,
                "e": e,
                "hnorm": hnorm
                }
    return distr