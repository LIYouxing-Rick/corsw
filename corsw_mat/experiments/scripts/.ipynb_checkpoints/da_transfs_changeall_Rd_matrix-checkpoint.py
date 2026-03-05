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

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from corsw.corsw import CORSW
from corsw_utils.download_bci import download_bci
from corsw_utils.get_data import get_data, get_cov, get_cov2
from corsw_utils.models import Transformations,  FeaturesKernel,OrthLinear, get_svc
from corsw.CorMatrix import cov2corr,Correlation
from corsw.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["PYTHONWARNINGS"] = "ignore::ConvergenceWarning:sklearn.svm.LinearSVC"


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
args = parser.parse_args()

def vec_to_sym(v: torch.Tensor, sampling: str) -> torch.Tensor:
    """
    把 v(..., 231) 映射到 Sym(21)（即 (21,21) 对称矩阵）。
    向量展开顺序：前 21 个为对角；其后为严格上三角 (i<j)。

    sampling: 
      - "olm","lsm": 非对角在填充矩阵前除以 √2（与 Frobenius 范数等距）
      - "ecm","lecm": 非对角不缩放
    """
    # 局部常量（不污染全局命名空间）
    N = 22
    M = N - 1            # 21
    D = M * (M + 1) // 2 # 231

    if v.shape[-1] != D:
        raise ValueError(f"期望 v 的最后一维为 {D}，实际为 {v.shape[-1]}。")
    sampling_l = sampling.lower()
    if sampling_l not in {"olm", "lsm", "ecm", "lecm"}:
        raise ValueError("sampling 仅支持 'olm','lsm','ecm','lecm'。")

    diag = v[..., :M]          # (..., 21)
    off  = v[..., M:]          # (..., 210)

    if sampling_l in {"olm", "lsm"}:
        off = off / math.sqrt(2.0)

    # 构造对称矩阵
    S = torch.zeros(v.shape[:-1] + (M, M), dtype=v.dtype, device=v.device)
    S = S + torch.diag_embed(diag)
    rr, cc = torch.triu_indices(M, M, offset=1, device=v.device)
    S[..., rr, cc] = off
    S[..., cc, rr] = off
    return S


def sym_to_vec(S: torch.Tensor, sampling: str) -> torch.Tensor:
    """
    把 S(..., 21, 21) 展开为向量 (..., 231)，顺序与 vec22_to_sym 一致。
    为保证与原向量一致：
      - 若 sampling in {"olm","lsm"}，这里会把严格上三角 *√2 回去；
      - 若 sampling in {"ecm","lecm"}，不做缩放。
    """
    # 局部常量
    N = 22
    M = N - 1            # 21
    D = M * (M + 1) // 2 # 231

    if S.shape[-2:] != (M, M):
        raise ValueError(f"S 的形状需为 (..., {M}, {M})，实际为 {S.shape[-2:]}。")
    sampling_l = sampling.lower()
    if sampling_l not in {"olm", "lsm", "ecm", "lecm"}:
        raise ValueError("sampling 仅支持 'olm','lsm','ecm','lecm'。")

    diag = torch.diagonal(S, dim1=-2, dim2=-1)     # (..., 21)
    rr, cc = torch.triu_indices(M, M, offset=1, device=S.device)
    off = S[..., rr, cc]                           # (..., 210)

    if sampling_l in {"olm", "lsm"}:
        off = off * math.sqrt(2.0)

    return torch.cat([diag, off], dim=-1)          # (..., 231)

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
        sym_Xs = vec_to_sym(L_Xs,22)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.vectorize(cor_Xt)
        sym_Xt = vec_to_sym(L_Xt,22)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        cor_Xs = cov2corr(cov_Xs)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.vectorize(cor_Xs)
        sym_Xs = vec_to_sym(L_Xs,22)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cor_Xt = cov2corr(cov_Xt)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.vectorize(cor_Xt)  
        sym_Xt = vec_to_sym(L_Xt,22)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

   
    n_freq = cov_Xs.shape[2]
    m = d*(d-1)//2
    
    model = Transformations(m,device=DEVICE,dtype=DTYPE,seed=seed)

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
        zs =  model(L_Xs)

        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        for f in range(n_freq):
      
            loss += spdsw.sdpsw(zs[:, 0, f], L_Xt[:, 0, f], p=2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.isnan(loss):
            print("!!! 损失值变为 NaN，训练已崩溃 !!!")
            break
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")



    stop = time.time()

    

    # 关键修复：将张量移动到 CPU 并转换为 NumPy 数组，需要先 detach() 分离梯度
    s_noalign = get_svc(L_Xs[:, 0].detach().cpu(), L_Xt[:, 0].detach().cpu(), ys.detach().cpu(), yt.detach().cpu(), d, multifreq, n_jobs=N_JOBS, random_state=seed)
    s_align = get_svc(L_Xs_bar[:, 0].detach().cpu(), L_Xt[:, 0].detach().cpu(), ys.detach().cpu(), yt.detach().cpu(), d, multifreq, n_jobs=N_JOBS, random_state=seed)

    return s_noalign, s_align, stop - start


if __name__ == "__main__":
    hyperparams = {
        "distance": ["ecm", "lecm", "olm", "lsm"],
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
        RESULTS = os.path.join(EXPERIMENTS, "results/da_cross_session_changeback.csv")
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
            s_noalign, s_align, runtime = run_test(params)  # 修复：移除额外的 n_jobs 参数

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