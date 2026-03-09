import warnings
import torch
import torch.nn as nn
import argparse
import time
import ot
import geoopt
import numpy as np
import pandas as pd
import itertools
import os
import math
import shutil
from glob import glob

from pathlib import Path
from joblib import Memory
from tqdm import trange

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from cormat_utils.download_bci import download_bci
from cormat_utils.get_data import get_data, get_cov, get_cov2
from cormat_utils.models import Transformations, FeaturesKernel, get_svc
from corswmat.CorMatrix import cov2corr, Correlation
from corswmat.CorMatrix import CorEuclideanCholeskyMetric, CorLogEuclideanCholeskyMetric, CorOffLogMetric, CorLogScaledMetric

from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=5, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
parser.add_argument("--lr", type=float, default=None, help="(deprecated) base learning rate")
parser.add_argument("--lr_cov", type=float, default=None, help="learning rate for covariance network")
parser.add_argument("--lr_cor", type=float, default=None, help="learning rate for correlation network")
parser.add_argument("--lr_mix", type=float, default=None, help="learning rate for fusion lambdas")
parser.add_argument("--power", type=float, default=None, help="power value for covariance matrix")
parser.add_argument("--use_cov_net", type=int, default=1, help="enable covariance network (1=on, 0=off)")
parser.add_argument("--use_cor_net", type=int, default=1, help="enable correlation network (1=on, 0=off)")
parser.add_argument("--epho", type=int, default=500, help="number of epochs")
parser.add_argument("--distance", type=str, default="lsm", help="distance metric")
parser.add_argument("--dataset", type=str, default="bnci2014001",
                    choices=["bnci2014001", "bnci2015001", "lee2019", "stieger2021"],
                    help="dataset name")
parser.add_argument("--max_iter", type=int, default=100, help="max_iter for olm/lsm metrics")
args = parser.parse_args()

N_JOBS = 50
SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
DEVICE = "cuda:0"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(location=os.path.join(EXPERIMENTS, "scripts/tmp_da/"), verbose=0)

DOWNLOAD = False
if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)

def power_matrix(x, a=1):
    """
    对输入的矩阵批次进行a次幂次方运算
    """
    bs, _, _, d, _ = x.shape
    x_reshaped = x.view(bs, d, d)
    
    if isinstance(a, int) and a >= 0:
        result = torch.matrix_power(x_reshaped, a)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(x_reshaped)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        eigenvalues_power = eigenvalues ** a
        eigenvalues_diag = torch.diag_embed(eigenvalues_power)
        
        result = torch.bmm(
            torch.bmm(eigenvectors, eigenvalues_diag),
            torch.linalg.inv(eigenvectors)
        )
        result = result.real
    
    result = result.view(bs, 1, 1, d, d)
    return result

def _fmt_lr_tag(lr: float) -> str:
    """Format learning rate for file naming"""
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")

def _fmt_power_tag(p: float) -> str:
    """Format power value for file naming"""
    return f"{p:g}"

@mem.cache
def run_test(params):
    """Run a single experiment with given parameters"""
    distance = params["distance"]
    n_proj = params["n_proj"]
    n_epochs = params["n_epochs"]
    seed = params["seed"].item() if hasattr(params["seed"], 'item') else params["seed"]
    subject = params["subject"]
    multifreq = params["multifreq"]
    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]
    reg = params["reg"]
    # separate learning rates
    lr_cov = float(params["lr_cov"])
    lr_cor = float(params["lr_cor"])
    lr_mix = float(params["lr_mix"])
    use_cov_net = int(params.get("use_cov_net", 1))
    use_cor_net = int(params.get("use_cor_net", 1))
    power_exp = float(params["power"])
    dataset = params.get("dataset", "bnci2014001")
    cov_fs = int(params.get("cov_fs", 250))
    cov_time_window = params.get("cov_time_window", None)
    max_iter = int(params.get("max_iter", 100))

    if use_cov_net == 0 and use_cor_net == 0:
        raise ValueError("At least one network must be enabled: set --use_cov_net=1 and/or --use_cor_net=1")

    get_cov_function = get_cov2 if multifreq else get_cov

    if distance == "olm":
        d = 22
        manifold = CorOffLogMetric(d, max_iter=max_iter)
    elif distance == "lsm":
        d = 22
        manifold = CorLogScaledMetric(d, max_iter=max_iter)
    elif distance == "lecm":
        d = 22
        manifold =  CorLogEuclideanCholeskyMetric(d)
    elif distance == "ecm":
        d = 22
        manifold =  CorEuclideanCholeskyMetric(d)
    else:
        raise ValueError(f"Unknown distance: {distance}")
    
    correlation = Correlation(d)

    if cross_subject:
        if target_subject == subject:
            return 1.0, 1.0, 0.0

        Xs, ys = get_data(subject, True, PATH_DATA, dataset=dataset)
        cov_Xs = torch.tensor(get_cov_function(Xs, fs=cov_fs, time_window=cov_time_window), device=DEVICE, dtype=DTYPE)
        # Precompute mlog(cov) for source (exclude from training time)
        L_mlog_Xs = linalg.sym_logm(cov_Xs)
        d = cov_Xs.shape[-1]
        correlation = Correlation(d)
        covp_Xs = power_matrix(cov_Xs, power_exp)
        cor_Xs = correlation.symmetrize(cov2corr(covp_Xs))
        L_Xs = manifold.deformation(cor_Xs)
      
        

        Xt, yt = get_data(target_subject, True, PATH_DATA, dataset=dataset)
        cov_Xt = torch.tensor(get_cov_function(Xt, fs=cov_fs, time_window=cov_time_window), device=DEVICE, dtype=DTYPE)
        # Precompute mlog(cov) for target (exclude from training time)
        L_mlog_Xt = linalg.sym_logm(cov_Xt)
        covp_Xt = power_matrix(cov_Xt, power_exp)
        cor_Xt = correlation.symmetrize(cov2corr(covp_Xt))
        L_Xt = manifold.deformation(cor_Xt)
       

        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        Xs, ys = get_data(subject, True, PATH_DATA, dataset=dataset)
        cov_Xs = torch.tensor(get_cov_function(Xs, fs=cov_fs, time_window=cov_time_window), device=DEVICE, dtype=DTYPE)
        # Precompute mlog(cov) for source (exclude from training time)
        L_mlog_Xs = linalg.sym_logm(cov_Xs)
        d = cov_Xs.shape[-1]
        correlation = Correlation(d)
        cov_Xs_powered = power_matrix(cov_Xs, power_exp)
        cor_Xs = cov2corr(cov_Xs_powered)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.deformation(cor_Xs)
        
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA, dataset=dataset)
        cov_Xt = torch.tensor(get_cov_function(Xt, fs=cov_fs, time_window=cov_time_window), device=DEVICE, dtype=DTYPE)
        # Precompute mlog(cov) for target (exclude from training time)
        L_mlog_Xt = linalg.sym_logm(cov_Xt)
        cov_Xt_powered = power_matrix(cov_Xt, power_exp)
        cor_Xt = cov2corr(cov_Xt_powered)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.deformation(cor_Xt)
        
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    n_freq = cov_Xs.shape[2]

    # two branches: mlog(cov) path and L_P path (both Transformations)
    # reinterpret switches: use_cov_net -> use_mlog_branch, use_cor_net -> use_lp_branch
    model_mlog = Transformations(d, n_freq, DEVICE, DTYPE, seed=seed) if use_cov_net == 1 else None
    model_lp = Transformations(d, n_freq, DEVICE, DTYPE, seed=seed) if use_cor_net == 1 else None

    # learnable fusion weights on simplex via logits -> softmax: lambda1+lambda2=1, lambda>=0
    lambda_logits = nn.Parameter(torch.tensor([0.0, 0.0], device=DEVICE, dtype=DTYPE))

    # Start timing only for training loop
    start = time.time()
    
    spdsw = SPDSW(
        d,
        n_proj,
        device=DEVICE,
        dtype=DTYPE,
        random_state=seed,
        sampling="logsw"
    )
    
    # optimizer with three parameter groups
    param_groups = []
    if use_cov_net == 1:
        param_groups.append({"params": model_mlog.parameters(), "lr": lr_cov})
    if use_cor_net == 1:
        param_groups.append({"params": model_lp.parameters(), "lr": lr_cor})
    param_groups.append({"params": [lambda_logits], "lr": lr_mix})
    # geoopt.RiemannianSGD 需要提供基础 lr 参数；各参数组内 lr 会覆盖该基础值
    optimizer = RiemannianSGD(param_groups, lr=lr_cov)
    pbar = trange(
        n_epochs,
        desc=(
            f"Training (lr_cov={lr_cov:.1e}, lr_cor={lr_cor:.1e}, lr_mix={lr_mix:.1e}, "
            f"power={power_exp:.2f}, subject={subject}, seed={seed})"
        )
    )

    for e in pbar:
        # path outputs with optional networks (source only)
        zs_mlog = model_mlog(L_mlog_Xs) if use_cov_net == 1 else torch.zeros_like(L_Xs)
        zs_lp = model_lp(L_Xs) if use_cor_net == 1 else torch.zeros_like(L_Xs)
        # fusion weights (sum to 1) via softmax; handle disabled branches
        lam = torch.nn.functional.softmax(lambda_logits, dim=0)
        lambda1 = lam[0] if use_cov_net == 1 else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        lambda2 = lam[1] if use_cor_net == 1 else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        # fused source (transformed) and fused target (raw)
        zs_fused = lambda1 * zs_mlog + lambda2 * zs_lp
        zt_fused = lambda1 * L_mlog_Xt + lambda2 * L_Xt

        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        for f in range(n_freq):
            loss += spdsw.spdsw(zs_fused[:, 0, f], zt_fused[:, 0, f], p=2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_postfix_str(
            f"loss = {loss.item():.3f}, lambda1={float(lambda1.item()):.3f}, lambda2={float(lambda2.item()):.3f}"
        )

    stop = time.time()
    runtime = stop - start

    # evaluate with fused outputs (recompute to ensure variables exist even if epochs=0)
    lam_eval = torch.nn.functional.softmax(lambda_logits, dim=0)
    lambda1_eval = lam_eval[0] if use_cov_net == 1 else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    lambda2_eval = lam_eval[1] if use_cor_net == 1 else torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    zs_mlog_eval = model_mlog(L_mlog_Xs) if use_cov_net == 1 else torch.zeros_like(L_Xs)
    zs_lp_eval = model_lp(L_Xs) if use_cor_net == 1 else torch.zeros_like(L_Xs)
    zs_fused_eval = lambda1_eval * zs_mlog_eval + lambda2_eval * zs_lp_eval
    zt_fused_eval = lambda1_eval * L_mlog_Xt + lambda2_eval * L_Xt

    # no-align: raw fused (source & target)
    s_noalign = get_svc(
        (lambda1_eval * L_mlog_Xs[:, 0] + lambda2_eval * L_Xs[:, 0]).detach().cpu(),
        (lambda1_eval * L_mlog_Xt[:, 0] + lambda2_eval * L_Xt[:, 0]).detach().cpu(),
        ys.detach().cpu(),
        yt.detach().cpu(),
        d, multifreq, n_jobs=N_JOBS, random_state=seed
    )
    # align: source transformed fused vs target raw fused
    s_align = get_svc(
        zs_fused_eval[:, 0].detach().cpu(),
        zt_fused_eval[:, 0].detach().cpu(),
        ys.detach().cpu(),
        yt.detach().cpu(),
        d, multifreq, n_jobs=N_JOBS, random_state=seed
    )

    return s_noalign, s_align, runtime

def compute_aggregated_metrics(results_df, distance):
    """计算聚合后的指标"""
    subjects = sorted(results_df['subject'].unique())
    L_session = {distance: {}, "no_align": {}}
    
    for s in subjects:
        filt = (results_df["distance"] == distance) & (results_df["subject"] == s)
        scores_align = results_df[filt]["align"]
        scores_noalign = results_df[filt]["no_align"]
        times = results_df[filt]["time"]
        
        L_session[distance][s] = {
            "mean_score": scores_align.mean(),
            "std_score": scores_align.std() if len(scores_align) > 1 else 0,
            "mean_time": times.mean(),
            "std_time": times.std() if len(times) > 1 else 0
        }
        
        L_session["no_align"][s] = {
            "mean_score": scores_noalign.mean(),
            "std_score": scores_noalign.std() if len(scores_noalign) > 1 else 0
        }
    
    acc_align_per_subject = [L_session[distance][s]["mean_score"] for s in subjects]
    acc_noalign_per_subject = [L_session["no_align"][s]["mean_score"] for s in subjects]
    time_per_subject = [L_session[distance][s]["mean_time"] for s in subjects]
    
    final_acc_align = np.mean(acc_align_per_subject)
    final_acc_noalign = np.mean(acc_noalign_per_subject)
    final_acc_align_std = np.std(acc_align_per_subject)
    final_acc_noalign_std = np.std(acc_noalign_per_subject)
    final_time = np.mean(time_per_subject)
    
    return {
        'align_mean': final_acc_align,
        'align_std': final_acc_align_std,
        'noalign_mean': final_acc_noalign,
        'noalign_std': final_acc_noalign_std,
        'time_mean': final_time,
        'per_subject_align': acc_align_per_subject,
        'per_subject_noalign': acc_noalign_per_subject,
        'subjects': subjects
    }

if __name__ == "__main__":
    dataset_defaults = {
        "bnci2014001": {"subjects": [1, 3, 7, 8, 9], "cov_fs": 250, "cov_time_window": (2.5, 6.0)},
        "bnci2015001": {"subjects": list(range(1, 13)), "cov_fs": 256, "cov_time_window": (1.0, 4.0)},
        "lee2019": {"subjects": list(range(1, 55)), "cov_fs": 250, "cov_time_window": (1.0, 3.5)},
        "stieger2021": {"subjects": list(range(1, 63)), "cov_fs": 250, "cov_time_window": (1.0, 2.996)},
    }
    ds_conf = dataset_defaults[args.dataset]
    if args.distance == "all":
        distances = ["ecm", "lecm", "olm", "lsm"]
    else:
        distances = [args.distance]

    # Set default hyperparameters based on task
    if args.task == "session":
        default_lr = 1e-2 if args.lr is None else args.lr
        default_power = 2.25 if args.power is None else args.power
        task_tag = "cross_session"
    else:
        default_lr = 1e-1 if args.lr is None else args.lr
        default_power = 1.0 if args.power is None else args.power
        task_tag = "cross_subject"
    
    # CSV filename
    lr_tag = _fmt_lr_tag(default_lr)
    power_tag = _fmt_power_tag(default_power)
    csv_filename = f"下三角对称化_spdsw模型{task_tag}_{args.dataset}_{args.distance}_lr_{lr_tag}_power_{power_tag}_epho{args.epho}_ntry{NTRY}.csv"
    RESULTS = os.path.join(EXPERIMENTS, "results", csv_filename)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    
    # Store all NTRY results
    all_ntry_results = []
    
    # derive individual learning rates (with sensible defaults)
    default_lr_cov = args.lr_cov if args.lr_cov is not None else default_lr
    default_lr_cor = args.lr_cor if args.lr_cor is not None else default_lr
    default_lr_mix = args.lr_mix if args.lr_mix is not None else (default_lr * 0.1)

    # Generate all seeds and run per-seed (print NTRY Round like expected)
    all_seeds = RNG.choice(10000, NTRY, replace=False)
    for try_idx, seed in enumerate(all_seeds):
        print(f"\n{'='*60}")
        print(f"NTRY Round {try_idx + 1}/{NTRY}, Seed: {seed}")
        print(f"{'='*60}")

        hyperparams = {
            "distance": distances,
            "n_proj": [500],
            "n_epochs": [args.epho],
            "seed": [seed],
            "subject": ds_conf["subjects"],
            "multifreq": [False],
            "reg": [10.],
            "dataset": [args.dataset],
            "cov_fs": [ds_conf["cov_fs"]],
            "cov_time_window": [ds_conf["cov_time_window"]],
            "max_iter": [args.max_iter],
            "lr_cov": [default_lr_cov],
            "lr_cor": [default_lr_cor],
            "lr_mix": [default_lr_mix],
            "use_cov_net": [args.use_cov_net],
            "use_cor_net": [args.use_cor_net],
            "power": [default_power],
        }

        if args.task == "session":
            hyperparams["cross_subject"] = [False]
            hyperparams["target_subject"] = [0]
        else:
            hyperparams["cross_subject"] = [True]
            hyperparams["target_subject"] = ds_conf["subjects"]

        keys, values = zip(*hyperparams.items())
        permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i, params in enumerate(permuts_params):
            try:
                print(f"\n[Seed {try_idx+1}/{NTRY}, Exp {i+1}/{len(permuts_params)}] Running:")
                print(f"  lr_cov={params['lr_cov']:.1e}, lr_cor={params['lr_cor']:.1e}, lr_mix={params['lr_mix']:.1e}, power={params['power']:.2f}, seed={seed}")
                print(f"  subject={params['subject']}, target={params.get('target_subject', 'N/A')}")

                if not params["cross_subject"]:
                    params["target_subject"] = 0
                if params["distance"] != "les":
                    params["reg"] = 1.

                s_noalign, s_align, runtime = run_test(params)

                result_dict = params.copy()
                result_dict["align"] = s_align
                result_dict["no_align"] = s_noalign
                result_dict["time"] = runtime

                if hasattr(result_dict["seed"], 'item'):
                    result_dict["seed"] = result_dict["seed"].item()

                all_ntry_results.append(result_dict)

                print(f"  Results: align={s_align:.3f}, no_align={s_noalign:.3f}, time={runtime:.1f}s")

            except (KeyboardInterrupt, SystemExit):
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Save results after all NTRY rounds
    if all_ntry_results:
        results_df = pd.DataFrame(all_ntry_results)
        results_df.to_csv(RESULTS, index=False)
        
        print(f"\n{'='*60}")
        print(f"Saved all {NTRY} rounds to: {RESULTS}")
        print(f"Total experiments: {len(results_df)}")
        print(f"{'='*60}")
        
        # Compute aggregated metrics
        print("\n=== Aggregated Results (All NTRY Rounds) ===")
        print(f"Experiments per subject: {NTRY}")
        for distance in distances:
            current_metrics = compute_aggregated_metrics(results_df, distance)
            print(f"\nMethod: {distance}")
            print(f"Per-subject align accuracy (averaged over {NTRY} seeds):")
            for idx, s in enumerate(current_metrics['subjects']):
                print(f"  Subject {s}: {current_metrics['per_subject_align'][idx]:.3f}")
            print(f"Final Align: {current_metrics['align_mean']:.3f} ± {current_metrics['align_std']:.3f}")
            print(f"Final No-Align: {current_metrics['noalign_mean']:.3f} ± {current_metrics['noalign_std']:.3f}")
        
       # Rebuild TensorBoard curves
        print("\n=== Rebuilding complete TensorBoard curves ===")
        
        # 修正：使用正确的文件名模式
        pattern = f"{task_tag}_{args.distance}_lr_{lr_tag}_power_*_epho*_ntry*.csv"
        all_csvs = glob(os.path.join(os.path.dirname(RESULTS), pattern))
        
        print(f"Found {len(all_csvs)} CSV files for lr={lr_tag}")
        
        # 如果没找到匹配的历史文件，至少使用当前文件
        if not all_csvs and os.path.exists(RESULTS):
            all_csvs = [RESULTS]
            print(f"Using current file only: {os.path.basename(RESULTS)}")
                
        all_data = []
        for csv_file in all_csvs:
            df = pd.read_csv(csv_file)
            all_data.append(df)
            print(f"  - Loaded: {os.path.basename(csv_file)} ({len(df)} rows)")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            unique_powers = sorted(combined_df['power'].unique())
            print(f"Power values found: {unique_powers}")
            
            run_name = f"{task_tag}_{args.distance}_lr_{lr_tag}"
            log_dir = os.path.join("/root/tf-logs/", run_name)
            
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            
            writer = SummaryWriter(log_dir=log_dir)
            
            for power in unique_powers:
                df_power = combined_df[combined_df['power'] == power]
                power_metrics = compute_aggregated_metrics(df_power, args.distance)
                step = int(power * 1000)
                
                writer.add_scalar('accuracy/align', power_metrics['align_mean'], step)
                writer.add_scalar('accuracy/no_align', power_metrics['noalign_mean'], step)
                writer.add_scalar('accuracy/align_std', power_metrics['align_std'], step)
                writer.add_scalar('accuracy/no_align_std', power_metrics['noalign_std'], step)
                writer.add_scalar('time/mean', power_metrics['time_mean'], step)
                writer.add_scalar('meta/n_experiments', len(df_power), step)
                
                print(f"  Power={power:.2f}: align={power_metrics['align_mean']:.3f}±{power_metrics['align_std']:.3f}, "
                      f"no_align={power_metrics['noalign_mean']:.3f}±{power_metrics['noalign_std']:.3f} "
                      f"(n={len(df_power)})")
            
            writer.close()
            print(f"\n[TensorBoard] Complete curve written to: {log_dir}")
            print("To view: tensorboard --logdir=/root/tf-logs/")
        else:
            print("No historical data found to merge")
    else:
        print("\nNo experiments were run")
