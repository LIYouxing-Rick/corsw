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

from pathlib import Path
from joblib import Memory
from tqdm import trange

# 1. ========= 导入 SummaryWriter =========
from torch.utils.tensorboard import SummaryWriter

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from cormat_utils.download_bci import download_bci
from cormat_utils.get_data import get_data, get_cov, get_cov2
from cormat_utils.models_homo import CayleyLSMTransform, CayleyOLMTransform, get_svc
from corswmat.CorMatrix import cov2corr,Correlation
from corswmat.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
args = parser.parse_args()

N_JOBS = 50
SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
RESULTS = os.path.join(EXPERIMENTS, "results/da.csv")
DEVICE = "cuda:0"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_da/"),
    verbose=0
)

# Set to True to download the data in experiments/data_bci
DOWNLOAD = False

if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)

correlation=Correlation(22)

@mem.cache
def run_test(params):

    distance = params["distance"]
    n_proj = params["n_proj"]
    n_epochs = params["n_epochs"]
    seed = params["seed"].item()
    subject = params["subject"]
    multifreq = params["multifreq"]

    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]
    reg = params["reg"]
    
    # 2. ========= 创建 TensorBoard SummaryWriter =========
    # 创建一个唯一的日志目录名，以区分不同的实验
    base_log_dir = os.path.join(EXPERIMENTS, "scripts", "tensorboard_log")
    run_name = (
        f"dist={distance}_subj={subject}_target={target_subject}_"
        f"nproj={n_proj}_epochs={n_epochs}_seed={seed}"
    )
    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, run_name))


    if multifreq:
        get_cov_function = get_cov2
    else:
        get_cov_function = get_cov
   
    d = 22

    if distance =="olm":
        manifold=CorOffLogMetric(d)
    elif distance =="lsm":
        manifold=CorLogScaledMetric(d)

    if cross_subject:
        if target_subject == subject:
            writer.close() # 关闭 writer 以免创建空目录
            return 1., 1., 0

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        cor_Xs = cov2corr(cov_Xs)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.deformation(cor_Xs)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.deformation(cor_Xt)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        cor_Xs = cov2corr(cov_Xs)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.deformation(cor_Xs)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.deformation(cor_Xt)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    n_freq = cov_Xs.shape[2]
    m = d*(d-1)//2

    if distance== "olm":
        model = CayleyOLMTransform(d, n_freq, DEVICE, DTYPE)
    else:
        model = CayleyLSMTransform(d, n_freq, DEVICE, DTYPE)

    start = time.time()

    if cross_subject:
        lr = 5e-4
    else:
        lr = 1e-2
        
    spdsw = SPDSW(
        d,
        n_proj,
        device=DEVICE,
        dtype=DTYPE,
        random_state=seed,
        sampling="spdsw"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    pbar = trange(n_epochs)

    for e in pbar:
        zs =  model(L_Xs)
        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        for f in range(n_freq):
            loss += spdsw.spdsw(zs[:, 0, f], L_Xt[:, 0, f], p=2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.isnan(loss):
            print("!!! 损失值变为 NaN，训练已崩溃 !!!")
            break
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")

        # 3. ========= 每 20 个 Epoch 记录一次 Loss =========
        # 使用 e+1 是为了在第20、40...个 epoch 记录，而不是第19、39...个
        if (e + 1) % 20 == 0:
            writer.add_scalar('Loss/train', loss.item(), e + 1)

    stop = time.time()
    
    runtime = stop - start

    s_noalign = get_svc(cov_Xs[:, 0], cov_Xt[:, 0], ys, yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)
    s_align = get_svc(model(cov_Xs)[:, 0], cov_Xt[:, 0], ys, yt, d, multifreq, n_jobs=N_JOBS, random_state=seed)

    # 4. ========= 记录最终的 Accuracy 和 Runtime =========
    # 我们使用 n_epochs作为全局步骤，表示这是最终结果
    writer.add_scalar('Accuracy/no_align', s_noalign, n_epochs)
    writer.add_scalar('Accuracy/align', s_align, n_epochs)
    writer.add_scalar('Runtime/seconds', runtime, n_epochs)
    
    # 5. ========= 关闭 Writer =========
    writer.close()

    return s_noalign, s_align, runtime


if __name__ == "__main__":
    hyperparams = {
        "distance": [ "lsm"],
        "n_proj": [500],
        "n_epochs": [500],
        "seed": RNG.choice(10000, NTRY, replace=False),
        "subject": [1, 3, 7, 8, 9],
        "target_subject": [1, 3, 7, 8, 9],
        "multifreq": [False],
        "reg": [10.],
    }

    if args.task == "session":
        hyperparams["cross_subject"] = [False]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_session_olm_homo.csv")
    elif args.task == "subject":
        hyperparams["cross_subject"] = [True]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_subject.csv")

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