import warnings
import torch
import argparse
import time
import ot
import geoopt
import numpy as np
import pandas as pd
import itertools
import os
import sys
import json
import hashlib

from pathlib import Path
from joblib import Memory
from tqdm import trange

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from utils.get_data import get_data, get_cov, get_cov2
from utils.models import Transformations, FeaturesKernel, get_svc


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["PYTHONWARNINGS"] = "ignore::ConvergenceWarning:sklearn.svm.LinearSVC"

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
parser.add_argument(
    "--dataset",
    type=str,
    default="bnci2014001",
    choices=["bnci2014001", "bnci2015001", "lee2019", "stieger2021"],
    help="dataset name"
)
parser.add_argument(
    "--subjects",
    type=str,
    default="auto",
    help="'auto' (dataset defaults), 'all' (same as auto), or comma-separated ids like '1,2,3'"
)
parser.add_argument("--resample", type=int, default=None, help="resample frequency for MOABB datasets")
parser.add_argument("--epoch_tmin", type=float, default=None, help="epoch start time for MOABB datasets")
parser.add_argument("--epoch_tmax", type=float, default=None, help="epoch end time for MOABB datasets")
parser.add_argument("--cov_tmin", type=float, default=None, help="covariance window start (s)")
parser.add_argument("--cov_tmax", type=float, default=None, help="covariance window end (s)")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint directory")
parser.add_argument("--checkpoint_every", type=int, default=1, help="save checkpoint every N epochs")
parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume from checkpoint")
parser.add_argument("--no-resume", dest="resume", action="store_false", help="disable checkpoint resume")
parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0 ...")
args = parser.parse_args()

N_JOBS = 50
SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
RESULTS = os.path.join(EXPERIMENTS, "results/da.csv")
if args.device == "auto":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == "cuda":
    DEVICE = "cuda:0"
else:
    DEVICE = args.device

if str(DEVICE).startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested device '{DEVICE}' but CUDA is not available.")
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_da/"),
    verbose=0
)
CHECKPOINT_DIR = args.checkpoint_dir or os.path.join(EXPERIMENTS, "scripts/checkpoints_da500")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _checkpoint_path(params):
    payload = json.dumps(params, sort_keys=True, default=_json_default, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return os.path.join(CHECKPOINT_DIR, f"{digest}.pt")

# Set to True to download the data in experiments/data_bci
DOWNLOAD = False

if DOWNLOAD:
    from utils.download_bci import download_bci
    path_data = download_bci(EXPERIMENTS)




@mem.cache
def run_test(params):

    distance = params["distance"]
    n_proj = params["n_proj"]
    n_epochs = params["n_epochs"]
    seed = params["seed"]
    subject = params["subject"]
    multifreq = params["multifreq"]

    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]
    reg = params["reg"]
    dataset = params["dataset"]
    resample = params.get("resample", None)
    epoch_tmin = params.get("epoch_tmin", None)
    epoch_tmax = params.get("epoch_tmax", None)
    cov_fs = int(params.get("cov_fs", 250))
    cov_time_window = params.get("cov_time_window", None)

    if multifreq:
        get_cov_function = get_cov2
    else:
        get_cov_function = get_cov

    if cross_subject:
        if target_subject == subject:
            return 1., 1., 0

        Xs, ys = get_data(
            subject, True, PATH_DATA,
            dataset=dataset,
            resample=resample,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
        )
        cov_Xs = torch.tensor(
            get_cov_function(Xs, fs=cov_fs, time_window=cov_time_window),
            device=DEVICE,
            dtype=DTYPE,
        )
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(
            target_subject, True, PATH_DATA,
            dataset=dataset,
            resample=resample,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
        )
        cov_Xt = torch.tensor(
            get_cov_function(Xt, fs=cov_fs, time_window=cov_time_window),
            device=DEVICE,
            dtype=DTYPE,
        )
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:

        Xs, ys = get_data(
            subject, True, PATH_DATA,
            dataset=dataset,
            resample=resample,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
        )
        cov_Xs = torch.tensor(
            get_cov_function(Xs, fs=cov_fs, time_window=cov_time_window),
            device=DEVICE,
            dtype=DTYPE,
        )
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(
            subject, False, PATH_DATA,
            dataset=dataset,
            resample=resample,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
        )
        cov_Xt = torch.tensor(
            get_cov_function(Xt, fs=cov_fs, time_window=cov_time_window),
            device=DEVICE,
            dtype=DTYPE,
        )
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    d = cov_Xs.shape[-1]
    n_freq = cov_Xs.shape[2]

    n_samples_s = len(cov_Xs)
    n_samples_t = len(cov_Xt)

    model = Transformations(d, n_freq, DEVICE, DTYPE, seed=seed)
    optimizer = RiemannianSGD(model.parameters(), lr=1.0)

    checkpoint_path = _checkpoint_path(params)
    start_epoch = 0
    elapsed_before = 0.0

    if distance in ["lew", "les"]:
        lr = 1e-2
        a = torch.ones((n_samples_s,), device=DEVICE, dtype=DTYPE) / n_samples_s
        b = torch.ones((n_samples_t,), device=DEVICE, dtype=DTYPE) / n_samples_t
        manifold = geoopt.SymmetricPositiveDefinite("LEM")

    elif distance in ["spdsw", "logsw", "sw"]:
        if cross_subject:
            lr = 5e-1
        else:
            lr = 1e-1
            
        spdsw = SPDSW(
            d,
            n_proj,
            device=DEVICE,
            dtype=DTYPE,
            random_state=seed,
            sampling=distance
        )

    optimizer = RiemannianSGD(model.parameters(), lr=lr)

    if args.resume and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", -1)) + 1
        elapsed_before = float(state.get("elapsed", 0.0))
        if start_epoch > n_epochs:
            start_epoch = n_epochs

    start = time.time()
    pbar = trange(start_epoch, n_epochs)
    last_epoch = start_epoch - 1

    try:
        for e in pbar:
            zs = model(cov_Xs)

            loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
            for f in range(n_freq):
                if distance == "lew":
                    M = manifold.dist(zs[:, 0, f][:, None], cov_Xt[:, 0, f][None]) ** 2
                    loss += 0.1 * ot.emd2(a, b, M)

                elif distance == "les":
                    M = manifold.dist(zs[:, 0, f][:, None], cov_Xt[:, 0, f][None]) ** 2
                    loss += 0.1 * ot.sinkhorn2(a, b, M, reg)

                elif distance in ["spdsw", "logsw", "sw"]:
                    loss += spdsw.spdsw(zs[:, 0, f], cov_Xt[:, 0, f], p=2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_epoch = e

            if (e + 1) % args.checkpoint_every == 0 or (e + 1) == n_epochs:
                elapsed_total = elapsed_before + (time.time() - start)
                torch.save({
                    "epoch": e,
                    "elapsed": elapsed_total,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "params": params,
                }, checkpoint_path)

            pbar.set_postfix_str(f"loss = {loss.item():.3f}")
    except (KeyboardInterrupt, SystemExit):
        elapsed_total = elapsed_before + (time.time() - start)
        torch.save({
            "epoch": last_epoch,
            "elapsed": elapsed_total,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "params": params,
        }, checkpoint_path)
        raise

    stop = time.time()

    s_noalign = get_svc(cov_Xs[:, 0], cov_Xt[:, 0], ys, yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)
    s_align = get_svc(model(cov_Xs)[:, 0], cov_Xt[:, 0], ys, yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)

    return s_noalign, s_align, elapsed_before + (stop - start)


if __name__ == "__main__":
    dataset_subject_defaults = {
        # keep original SPDSW default subset for BNCI2014001
        "bnci2014001": [1, 3, 7, 8, 9],
        # full subject sets for additional datasets
        "bnci2015001": list(range(1, 13)),
        "lee2019": list(range(1, 55)),
        "stieger2021": list(range(1, 63)),
    }

    dataset_cov_defaults = {
        "bnci2014001": {"cov_fs": 250, "cov_time_window": (2.5, 6.0)},
        "bnci2015001": {"cov_fs": 256, "cov_time_window": (1.0, 4.0)},
        "lee2019": {"cov_fs": 250, "cov_time_window": (1.0, 3.5)},
        "stieger2021": {"cov_fs": 250, "cov_time_window": (1.0, 2.996)},
    }
    ds_defaults = dataset_cov_defaults[args.dataset]

    cov_time_window = ds_defaults["cov_time_window"]
    if args.cov_tmin is not None and args.cov_tmax is not None:
        cov_time_window = (args.cov_tmin, args.cov_tmax)

    subjects_arg = args.subjects.strip().lower()
    if subjects_arg in ["auto", "all"]:
        subject_list = dataset_subject_defaults[args.dataset]
    else:
        subject_list = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]

    hyperparams = {
        "distance": ["spdsw", "logsw"],
        "n_proj": [500],
        "n_epochs": [500],
        "seed": RNG.choice(10000, NTRY, replace=False),
        "subject": subject_list,
        "target_subject": subject_list,
#         "cross_subject": [False],
        "multifreq": [False],
        "reg": [10.],
        "dataset": [args.dataset],
        "resample": [args.resample],
        "epoch_tmin": [args.epoch_tmin],
        "epoch_tmax": [args.epoch_tmax],
        "cov_fs": [ds_defaults["cov_fs"]],
        "cov_time_window": [cov_time_window],
    }

    if args.task == "session":
        hyperparams["cross_subject"] = [False]
        RESULTS = os.path.join(EXPERIMENTS, f"results/da_{args.dataset}_cross_session_epho500.csv")
    elif args.task == "subject":
        hyperparams["cross_subject"] = [True]
        RESULTS = os.path.join(EXPERIMENTS, f"results/da_{args.dataset}_cross_subject_epho500.csv")

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {
        "align": [],
        "no_align": [],
        "time": []
    }

    for params in permuts_params:
        try:
            print(params)
            if not params["cross_subject"]:
                params["target_subject"] = 0
            if params["distance"] != "les":
                params["reg"] = 1.
            s_noalign, s_align, runtime = run_test(params)

            # Storing results
            for key in params.keys():
                if key not in dico_results:
                    dico_results[key] = [params[key]]
                else:
                    dico_results[key].append(params[key])

            dico_results["align"].append(s_align)
            dico_results["no_align"].append(s_noalign)
            dico_results["time"].append(runtime)

        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)
