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
import math

from pathlib import Path
from joblib import Memory
from tqdm import trange

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from cormat_utils.download_bci import download_bci
from cormat_utils.get_data import get_data, get_cov, get_cov2
from utils.models_log import Transformations11,  FeaturesKernel, get_svc
from corswmat.CorMatrix import cov2corr,Correlation
from corswmat.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric

# Import SummaryWriter for TensorBoard
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["PYTHONWARNINGS"] = "ignore::ConvergenceWarning:sklearn.svm.LinearSVC"


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
args = parser.parse_args()


def vec_to_sym(v: torch.Tensor, n: int, sampling: str = "olm") -> torch.Tensor:
    """
    把向量 v (..., d) 映射到 Sym(n-1)（输出 (..., m, m)），
    其中 m = n-1, d = m(m+1)/2。
    向量顺序：前 m 个填对角，其余按严格上三角 (i<j) 顺序。
    sampling:
      - "olm","lsm": 非对角写入前除以 √2（与 Frobenius 等距）
      - "ecm","lecm": 非对角不缩放（对标准欧氏不等距）
    """
    m = n - 1
    D = m * (m + 1) // 2
    if v.shape[-1] != D:
        raise ValueError(f"期望 v 的最后一维为 {D}，实际 {v.shape[-1]}")
    sampling_l = sampling.lower()
    if sampling_l not in {"olm", "lsm", "ecm", "lecm"}:
        raise ValueError("sampling 仅支持 'olm','lsm','ecm','lecm'")

    diag = v[..., :m]        # (..., m)
    off  = v[..., m:]        # (..., m*(m-1)/2)
    if sampling_l in {"olm", "lsm"}:
        off = off / math.sqrt(2.0)

    S = torch.zeros(v.shape[:-1] + (m, m), dtype=v.dtype, device=v.device)
    S = S + torch.diag_embed(diag)
    rr, cc = torch.triu_indices(m, m, 1, device=v.device)
    S[..., rr, cc] = off
    S[..., cc, rr] = off
    return S


def sym_to_vec(S: torch.Tensor, n: int, sampling: str = "olm") -> torch.Tensor:
    """
    逆映射：Sym(n-1) → 向量（对角先、再严格上三角）。
    若 sampling in {"olm","lsm"}，严格上三角会乘回 √2，得到与 vec_to_sym 输入一致的向量。
    """
    m = n - 1
    if S.shape[-2:] != (m, m):
        raise ValueError(f"S 形状需为 (..., {m}, {m})，实际 {S.shape[-2:]}")
    sampling_l = sampling.lower()
    if sampling_l not in {"olm", "lsm", "ecm", "lecm"}:
        raise ValueError("sampling 仅支持 'olm','lsm','ecm','lecm'")

    diag = torch.diagonal(S, dim1=-2, dim2=-1)   # (..., m)
    rr, cc = torch.triu_indices(m, m, 1, device=S.device)
    off = S[..., rr, cc]                          # (..., m*(m-1)/2)
    if sampling_l in {"olm", "lsm"}:
        off = off * math.sqrt(2.0)

    return torch.cat([diag, off], dim=-1)



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

# Define and create TensorBoard log directory
TENSORBOARD_LOG_DIR = os.path.join(EXPERIMENTS, "scripts", "tensorboard_log")
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


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
    
    # Create a unique name for the TensorBoard run
    run_name = f"dist_{distance}_proj_{n_proj}_epochs_{n_epochs}_seed_{seed}_subj_{subject}_targ_{target_subject}"
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, run_name))


    if multifreq:
        get_cov_function = get_cov2
    else:
        get_cov_function = get_cov
   
    d = 22  # 移动到前面定义

    if distance == "ecm":
        manifold=CorEuclideanCholeskyMetric(d)
    elif distance =="lecm":
        manifold=CorLogEuclideanCholeskyMetric(d)
    elif distance =="olm":
        manifold=CorOffLogMetric(d)
    elif distance =="lsm":
        manifold=CorLogScaledMetric(d)
    
       
        

    if cross_subject:
        if target_subject == subject:
            return 1., 1., 0

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        cor_Xs = cov2corr(cov_Xs)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.vectorize(cor_Xs)
        sym_Xs = vec_to_sym(L_Xs,22,sampling=distance)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.vectorize(cor_Xt)
        sym_Xt = vec_to_sym(L_Xt,22,sampling=distance)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        
        
       
        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        
        cor_Xs = cov2corr(cov_Xs)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.vectorize(cor_Xs)
        sym_Xs = vec_to_sym(L_Xs,22,sampling=distance)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.vectorize(cor_Xt)  
        sym_Xt = vec_to_sym(L_Xt,22,sampling=distance)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

   
    n_freq = cov_Xs.shape[2]
    
    
    model = Transformations11(d-1, n_freq, DEVICE, DTYPE, seed=seed)

    start = time.time()


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
        sampling="logsw"
    )

   
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    pbar = trange(n_epochs)

    for e in pbar:
        zs =  model(sym_Xs)

        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        for f in range(n_freq):
            
            loss += spdsw.spdsw(zs[:, 0, f], sym_Xt[:, 0, f], p=2)
           

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        


    stop = time.time()

    

    # 关键修复：将张量移动到 CPU 并转换为 NumPy 数组，需要先 detach() 分离梯度
    s_noalign = get_svc(sym_Xs[:, 0].detach().cpu(), sym_Xt[:, 0].detach().cpu(), ys.detach().cpu(), yt.detach().cpu(), d-1, multifreq, n_jobs=N_JOBS, random_state=seed)
    s_align = get_svc(model(sym_Xs)[:, 0].detach().cpu(), sym_Xt[:, 0].detach().cpu(), ys.detach().cpu(), yt.detach().cpu(), d-1, multifreq, n_jobs=N_JOBS, random_state=seed)




    return s_noalign, s_align, stop - start


if __name__ == "__main__":
    hyperparams = {
        "distance": ["lsm","olm"],
        "n_proj": [500],
        "n_epochs": [500],
        "seed": RNG.choice(10000, NTRY, replace=False),
        "subject": [1, 3, 7, 8, 9],
        "target_subject": [1, 3, 7, 8, 9],
#         "cross_subject": [False],
        "multifreq": [False],
        "reg": [10.],
    }

    if args.task == "session":
        hyperparams["cross_subject"] = [False]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_session_matrix_olmlr1e-2.csv")
    elif args.task == "subject":
        hyperparams["cross_subject"] = [True]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_subject_matrix_olmls.csv")

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