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
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument("--power", type=float, default=None, help="power value for covariance matrix")
parser.add_argument("--epho", type=int, default=500, help="number of epochs")
parser.add_argument("--distance", type=str, default="lsm", help="distance metric")
args = parser.parse_args()

N_JOBS = 50
SEED = 2025
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

correlation = Correlation(22)


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
    lr = float(params["lr"])
    power_exp = float(params["power"])

    get_cov_function = get_cov2 if multifreq else get_cov
    d = 22

    if distance == "olm":
        manifold = CorOffLogMetric(d)
    elif distance == "lsm":
        manifold = CorLogScaledMetric(d)
    else:
        raise ValueError(f"Unknown distance: {distance}")
    
    manifold1 = CorLogScaledMetric(d,max_iter=1)
    manifold2 = CorOffLogMetric(d,max_iter=1)

    if cross_subject:
        if target_subject == subject:
            return 1.0, 1.0, 0.0, []

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        covp_Xs =  power_matrix(cov_Xs, power_exp)
        cor_Xs = correlation.symmetrize(cov2corr(covp_Xs))
        L_Xs = manifold.deformation(cor_Xs)
        L_Xs = linalg.sym_logm(cov_Xs) + L_Xs
      
        

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        covp_Xt =  power_matrix(cov_Xt, power_exp)
        cor_Xt = correlation.symmetrize(cov2corr(covp_Xt))
        L_Xt = manifold.deformation(cor_Xt)
        L_Xt = linalg.sym_logm(cov_Xt) + L_Xt
       

        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        cov_Xs_powered = power_matrix(cov_Xs, power_exp)
        cor_Xs = cov2corr(cov_Xs_powered)
        cor_Xs = correlation.symmetrize(cor_Xs)
        L_Xs = manifold.deformation(cor_Xs)
        L_Xs = linalg.sym_logm(cov_Xs) + L_Xs
        
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        cov_Xt_powered = power_matrix(cov_Xt, power_exp)
        cor_Xt = cov2corr(cov_Xt_powered)
        cor_Xt = correlation.symmetrize(cor_Xt)
        L_Xt = manifold.deformation(cor_Xt)
        L_Xt = linalg.sym_logm(cov_Xt) + L_Xt
        
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    n_freq = cov_Xs.shape[2]

 
    model = Transformations(d, n_freq, DEVICE, DTYPE, seed=seed)

    start = time.time()
    
    spdsw = SPDSW(
        d,
        n_proj,
        device=DEVICE,
        dtype=DTYPE,
        random_state=seed,
        sampling="logsw"
    )
    
    optimizer = RiemannianSGD(model.parameters(), lr=lr)
    pbar = trange(n_epochs, desc=f"Training (lr={lr:.1e}, power={power_exp:.2f}, subject={subject}, seed={seed})")

    for e in pbar:
        zs = model(L_Xs)
        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        
        for f in range(n_freq):
            loss += spdsw.spdsw(zs[:, 0, f], L_Xt[:, 0, f], p=2)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_postfix_str(f"loss = {loss.item():.3f}")

    stop = time.time()
    runtime = stop - start

    s_noalign = get_svc(
        L_Xs[:, 0].detach().cpu(),
        L_Xt[:, 0].detach().cpu(),
        ys.detach().cpu(),
        yt.detach().cpu(),
        d, multifreq, n_jobs=N_JOBS, random_state=seed
    )
    s_align = get_svc(
        model(L_Xs)[:, 0].detach().cpu(),
        L_Xt[:, 0].detach().cpu(),
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
    # Set default hyperparameters based on task
    if args.task == "session":
        default_lr = 1e-2 if args.lr is None else args.lr
        default_power = 2.25 if args.power is None else args.power
        task_tag = "cross_session"
    else:
        default_lr = 1e-1 if args.lr is None else args.lr
        default_power = 1.0 if args.power is None else args.power
        task_tag = "cross_subject"
    
    # Generate all seeds
    all_seeds = RNG.choice(10000, NTRY, replace=False)
    
    # CSV filename
    lr_tag = _fmt_lr_tag(default_lr)
    power_tag = _fmt_power_tag(default_power)
    csv_filename = f"下三角对称化_spdsw模型{task_tag}_{args.distance}_lr_{lr_tag}_power_{power_tag}_epho{args.epho}_ntry{NTRY}.csv"
    RESULTS = os.path.join(EXPERIMENTS, "results", csv_filename)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    
    # Store all NTRY results
    all_ntry_results = []
    
    # Run experiments for each seed
    for try_idx, seed in enumerate(all_seeds):
        print(f"\n{'='*60}")
        print(f"NTRY Round {try_idx + 1}/{NTRY}, Seed: {seed}")
        print(f"{'='*60}")
        
        hyperparams = {
            "distance": [args.distance],
            "n_proj": [500],
            "n_epochs": [args.epho],
            "seed": [seed],
            "subject": [1, 3, 7, 8, 9],
            "multifreq": [False],
            "reg": [10.],
            "lr": [default_lr],
            "power": [default_power],
        }

        if args.task == "session":
            hyperparams["cross_subject"] = [False]
            hyperparams["target_subject"] = [0]
        else:
            hyperparams["cross_subject"] = [True]
            hyperparams["target_subject"] = [1, 3, 7, 8, 9]

        keys, values = zip(*hyperparams.items())
        permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for i, params in enumerate(permuts_params):
            try:
                print(f"\n[Seed {try_idx+1}/{NTRY}, Exp {i+1}/{len(permuts_params)}] Running:")
                print(f"  lr={params['lr']:.1e}, power={params['power']:.2f}, seed={seed}")
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
                result_dict["ntry_round"] = try_idx + 1
                
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
        current_metrics = compute_aggregated_metrics(results_df, args.distance)
        
        print("\n=== Aggregated Results (All NTRY Rounds) ===")
        print(f"Experiments per subject: {NTRY}")
        print(f"Per-subject align accuracy (averaged over {NTRY} seeds):")
        for idx, s in enumerate(current_metrics['subjects']):
            print(f"  Subject {s}: {current_metrics['per_subject_align'][idx]:.3f}")
        print(f"\nFinal Align: {current_metrics['align_mean']:.3f} ± {current_metrics['align_std']:.3f}")
        print(f"Final No-Align: {current_metrics['noalign_mean']:.3f} ± {current_metrics['noalign_std']:.3f}")
        
       # Rebuild TensorBoard curves
        print("\n=== Rebuilding complete TensorBoard curves ===")
        
        # 修正：使用正确的文件名模式
        pattern = f"下三角对称化_spdsw模型{task_tag}_{args.distance}_lr_{lr_tag}_power_*_epho*_ntry*.csv"
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
            
            run_name = f"下三角对称化_spdsw模型{task_tag}_{args.distance}_lr_{lr_tag}"
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