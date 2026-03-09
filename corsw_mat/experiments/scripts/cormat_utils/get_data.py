#!/usr/bin/env python3

# Taken from https://github.com/MultiScale-BCI/IV-2a

'''	Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''

import numpy as np
import scipy.io as sio
import re
import os
from typing import Optional, Sequence, Tuple

from .filters import load_filterbank, butter_fir_filter

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"


DATASET_PRESETS = {
    "bnci2014001": {
        "events": ["left_hand", "right_hand", "feet", "tongue"],
        "channels": None,
        "resample": 250,
        "tmin": 0.0,
        "tmax": 6.0,
        "cov_fs": 250,
        "cov_time_window": (2.5, 6.0),
        "subjects": [1, 3, 7, 8, 9],
    },
    "bnci2015001": {
        "events": ["right_hand", "feet"],
        "channels": None,
        "resample": 256,
        "tmin": 0.0,
        "tmax": 4.0,
        "cov_fs": 256,
        "cov_time_window": (1.0, 4.0),
        "subjects": list(range(1, 13)),
    },
    "lee2019": {
        "events": ["left_hand", "right_hand"],
        "channels": [
            "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        ],
        "resample": 250,
        "tmin": 0.0,
        "tmax": 3.5,
        "cov_fs": 250,
        "cov_time_window": (1.0, 3.5),
        "subjects": list(range(1, 55)),
    },
    "stieger2021": {
        "events": ["left_hand", "right_hand", "both_hand", "rest"],
        "channels": [
            "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
            "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
        ],
        "resample": 250,
        "tmin": 0.0,
        "tmax": 2.996,
        "cov_fs": 250,
        "cov_time_window": (1.0, 2.996),
        "subjects": list(range(1, 63)),
    },
}


def _resolve_time_windows(fs: int, time_window: Optional[Tuple[float, float]] = None):
    if time_window is None:
        time_windows_flt = np.array([[2.5, 6.0]]) * fs
    else:
        t0, t1 = time_window
        if t1 <= t0:
            raise ValueError(f"Invalid time window: ({t0}, {t1})")
        time_windows_flt = np.array([[t0, t1]]) * fs
    return time_windows_flt.astype(int)


def get_cov(data, ftype="butter", fs=250, time_window: Optional[Tuple[float, float]] = None):
    """
        One frequency
    """
    bw = [22] # [25] # bandwidth
    forder = 8
    max_freq = 30

    time_windows = _resolve_time_windows(fs=fs, time_window=time_window)

    filter_bank = load_filterbank(bandwidth = bw, fs = fs, order = forder, 
                                  max_freq = max_freq, ftype = ftype)

    n_tr_trial, n_channel, _ = data.shape
    n_riemann = int((n_channel+1)*n_channel/2)

    n_temp = time_windows.shape[0]
    n_freq = filter_bank.shape[0]
    rho = 0.1

    temp_windows = time_windows

    cov_mat = np.zeros((n_tr_trial, n_temp, n_freq, n_channel, n_channel))

    # calculate training covariance matrices  
    for trial_idx in range(n_tr_trial):	

        for temp_idx in range(n_temp): 
            t_start, t_end = temp_windows[temp_idx, 0], temp_windows[temp_idx, 1]
            n_samples = t_end-t_start

            for freq_idx in range(n_freq):
                # filter signal
                data_filter = butter_fir_filter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx])
                # regularized covariance matrix 
                cov_mat[trial_idx,temp_idx,freq_idx] = 1/(n_samples-1)*np.dot(data_filter,np.transpose(data_filter)) + rho/n_samples*np.eye(n_channel)

    return cov_mat


def get_cov2(data, fs=250, time_window: Optional[Tuple[float, float]] = None):
    """
        Multifrequency (hyperparameters from https://github.com/MultiScale-BCI/IV-2a)
    """
    # bw = [25] ## bandwidth [2, 4, 8, 16, 32]
    bw = [2, 4, 8, 16, 32]
#     max_freq = 40
    forder = 8
    max_freq = 30
    ftype = "butter"

    time_windows = _resolve_time_windows(fs=fs, time_window=time_window)
    
    filter_bank = load_filterbank(bandwidth=bw, fs=fs, order=forder,
                                  max_freq=max_freq, ftype=ftype, multifreq=False)

    n_tr_trial, n_channel, _ = data.shape

    n_temp = time_windows.shape[0]
    n_freq = filter_bank.shape[0]
    rho = 0.1

    temp_windows = time_windows

    cov_mat = np.zeros((n_tr_trial, n_temp, n_freq, n_channel, n_channel))

    # calculate training covariance matrices  
    for trial_idx in range(n_tr_trial):	

        for temp_idx in range(n_temp): 
            t_start, t_end  = temp_windows[temp_idx,0], temp_windows[temp_idx,1]
            n_samples = t_end-t_start

            for freq_idx in range(n_freq): 
                # filter signal 
                data_filter = butter_fir_filter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx])
                # regularized covariance matrix 
                cov_mat[trial_idx,temp_idx,freq_idx] = 1/(n_samples-1)*np.dot(data_filter,np.transpose(data_filter)) + rho/n_samples*np.eye(n_channel)

    return cov_mat


def _load_moabb_subject_data(
    dataset_name: str,
    subject: int,
    training: bool,
    sessions=None,
    channels: Optional[Sequence[str]] = None,
    resample: Optional[int] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
):
    from moabb.paradigms.motor_imagery import MotorImagery
    from moabb.datasets.bnci import BNCI2014001, BNCI2015001
    from moabb.datasets import Lee2019_MI

    name = dataset_name.lower()
    if name not in DATASET_PRESETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    preset = DATASET_PRESETS[name]
    if channels is None:
        channels = preset["channels"]
    if resample is None:
        resample = preset["resample"]
    if tmin is None:
        tmin = preset["tmin"]
    if tmax is None:
        tmax = preset["tmax"]
    events = preset["events"]

    if name == "bnci2014001":
        dataset = BNCI2014001()
    elif name == "bnci2015001":
        dataset = BNCI2015001()
    elif name == "lee2019":
        dataset = Lee2019_MI()
    elif name == "stieger2021":
        from moabb.datasets import Stieger2021 as MoabbStieger2021
        dataset = MoabbStieger2021()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    paradigm_kwargs = dict(events=events)
    if channels is not None:
        paradigm_kwargs["channels"] = list(channels)
    if resample is not None:
        paradigm_kwargs["resample"] = int(resample)
    if tmin is not None:
        paradigm_kwargs["tmin"] = float(tmin)
    if tmax is not None:
        paradigm_kwargs["tmax"] = float(tmax)

    paradigm = MotorImagery(n_classes=len(events), **paradigm_kwargs)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    def _session_to_int(value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        m = re.search(r"\d+", str(value))
        return int(m.group(0)) if m else None

    if isinstance(sessions, dict):
        unique_sessions = list(metadata["session"].unique())
        if sessions.get("order", "first") == "last":
            unique_sessions = unique_sessions[::-1]
        take_n = int(sessions.get("n", len(unique_sessions)))
        requested = {_session_to_int(s) for s in unique_sessions[:take_n]}
        requested = {s for s in requested if s is not None}
        session_norm = np.array([_session_to_int(s) for s in metadata["session"].to_numpy()], dtype=object)
        mask = np.array([(s in requested) for s in session_norm], dtype=bool)
        X = X[mask]
        y = y[mask]
        metadata = metadata.loc[mask]
    elif sessions is not None and len(sessions) > 0:
        requested = {_session_to_int(s) for s in sessions}
        requested = {s for s in requested if s is not None}
        session_norm = np.array([_session_to_int(s) for s in metadata["session"].to_numpy()], dtype=object)
        mask = np.array([(s in requested) for s in session_norm], dtype=bool)
        X = X[mask]
        y = y[mask]
        metadata = metadata.loc[mask]
    else:
        unique_sessions = list(metadata["session"].unique())
        unique_sessions = sorted(unique_sessions)
        if len(unique_sessions) >= 2:
            selected = unique_sessions[0] if training else unique_sessions[1]
            mask = (metadata["session"] == selected).to_numpy()
            X = X[mask]
            y = y[mask]

    if len(X) == 0 or len(y) == 0:
        raise RuntimeError(
            f"Empty trials after session filtering for dataset={dataset_name}, subject={subject}, "
            f"training={training}, sessions={sessions}."
        )

    uniq = np.unique(y)
    y_map = {lbl: i + 1 for i, lbl in enumerate(uniq)}
    y_int = np.array([y_map[v] for v in y], dtype=np.int64)
    return X, y_int


def get_data(
    subject,
    training,
    PATH,
    dataset: str = "bnci2014001",
    sessions=None,
    channels: Optional[Sequence[str]] = None,
    resample: Optional[int] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets

	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	name = dataset.lower()
	if name != "bnci2014001":
		return _load_moabb_subject_data(
			dataset_name=dataset,
			subject=subject,
			training=training,
			sessions=sessions,
			channels=channels,
			resample=resample,
			tmin=tmin,
			tmax=tmax,
		)

	local_split = f"A0{int(subject)}{'T' if training else 'E'}.mat"
	local_file = os.path.join(PATH, local_split)
	if not os.path.exists(local_file):
		return _load_moabb_subject_data(
			dataset_name=dataset,
			subject=subject,
			training=training,
			sessions=sessions,
			channels=channels,
			resample=resample,
			tmin=tmin,
			tmax=tmax,
		)

	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	a = sio.loadmat(local_file)
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2=[a_data1[0,0]]
		a_data3=a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]
		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]
