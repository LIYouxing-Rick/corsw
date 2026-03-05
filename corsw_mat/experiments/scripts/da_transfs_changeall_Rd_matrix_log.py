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

from spdsw.spdsw_proj_copy import SPDSW
from cormat_utils.download_bci import download_bci
from cormat_utils.get_data import get_data, get_cov, get_cov2
from cormat_utils.models_homo_stifel_modrelu import get_svc, StiefelOLMTransform, StiefelLSMTransform
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
parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")  # 改为epochs
parser.add_argument("--distance", type=str, default="lsm", help="distance metric")
parser.add_argument("--lr_mode", type=str, default="equal", help="learning rate mode: auto10, auto5, equal, or custom:ratio")
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

def matrix_power(x, a=1):
    """
    对输入的矩阵批次进行a次幂次方运算
    Args:
        x: torch.Tensor, shape (bs, 1, 1, d, d)
        a: int or float
    Returns:
        torch.Tensor, shape (bs, 1, 1, d, d)
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

def setup_stifel_optimizer(model, base_lr=1e-3, lr_mode="equal"):
    """
    为Cayley模型设置优化器，支持不同参数类型的不同学习率
    """
    spd_params = model.get_spd_parameters()
    stifel_params = model.get_stiefel_parameters()
    euclidean_params = model.get_euclidean_parameters()
    
    if lr_mode == "auto10":
        spd_lr = base_lr
        stifel_lr = base_lr / 10
        euclidean_lr = base_lr
    elif lr_mode == "auto5":
        spd_lr = base_lr
        stifel_lr = base_lr / 5
        euclidean_lr = base_lr
    elif lr_mode == "equal":
        spd_lr = base_lr
        stifel_lr = base_lr
        euclidean_lr = base_lr
    elif lr_mode.startswith("custom:"):
        try:
            ratio = float(lr_mode.split(":")[1])
            spd_lr = base_lr
            stifel_lr = base_lr / ratio
            euclidean_lr = base_lr
        except (IndexError, ValueError):
            raise ValueError(f"Invalid custom ratio format: {lr_mode}. Use 'custom:N' format.")
    else:
        raise ValueError(f"Unknown lr_mode: {lr_mode}")
    
    print(f"优化器设置 (lr_mode={lr_mode}):")
    print(f"  SPD参数: lr={spd_lr:.1e} ({len(spd_params)}个参数)")
    print(f"  stiefel参数: lr={stifel_lr:.1e} ({len(stifel_params)}个参数)")
    print(f"  欧氏参数: lr={euclidean_lr:.1e} ({len(euclidean_params)}个参数)")
    
    param_groups = []
    if len(spd_params) > 0:
        param_groups.append({'params': spd_params, 'lr': spd_lr})
    if len(stifel_params) > 0:
        param_groups.append({'params': stifel_params, 'lr': stifel_lr})
    if len(euclidean_params) > 0:
        param_groups.append({'params': euclidean_params, 'lr': euclidean_lr})
    
    if len(param_groups) == 0:
        raise ValueError("模型中没有找到参数！")
    
    optimizer = RiemannianSGD(param_groups, lr=base_lr)
    return optimizer

# @mem.cache  # 注释掉缓存，因为我们要记录loss
def run_test_with_loss_tracking(params, target_epochs, writer=None, exp_tag=""):
    """
    训练函数 - 记录loss到TensorBoard
    """
    distance = params["distance"]
    n_proj = params["n_proj"]
    seed = params["seed"].item() if hasattr(params["seed"], 'item') else params["seed"]
    subject = params["subject"]
    multifreq = params["multifreq"]
    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]
    reg = params["reg"]
    lr = float(params["lr"])
    power_exp = float(params["power"])
    lr_mode = params.get("lr_mode", "equal")

    get_cov_function = get_cov2 if multifreq else get_cov
    d = 22

    # 设置距离度量
    if distance == "olm":
        manifold = CorOffLogMetric(d)
    elif distance == "lsm":
        manifold = CorLogScaledMetric(d)
    else:
        raise ValueError(f"Unknown distance: {distance}")

    manifold1 = CorLogScaledMetric(d)
    manifold2 = CorOffLogMetric(d)

    # 数据准备
    if cross_subject:
        if target_subject == subject:
            return 1.0, 1.0, 0.0, []

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        covp_Xs = matrix_power(cov_Xs, power_exp)
        cor_Xs = correlation.symmetrize(cov2corr(covp_Xs))
        L_Xs = manifold.deformation(cor_Xs)
      
        L_Xs1 = manifold1.deformation(cor_Xs)

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        covp_Xt = matrix_power(cov_Xt, power_exp)
        cor_Xt = correlation.symmetrize(cov2corr(covp_Xt))
        L_Xt = manifold.deformation(cor_Xt)
        L_Xt1 = manifold1.deformation(cor_Xt)

        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:
        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE, dtype=DTYPE)
        covp_Xs = matrix_power(cov_Xs, power_exp)
        cor_Xs = correlation.symmetrize(cov2corr(covp_Xs))
        L_Xs = manifold.deformation(cor_Xs)
        L_Xs =   L_Xs+cov_Xs
        L_Xs1 = manifold1.deformation(cor_Xs)

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE, dtype=DTYPE)
        covp_Xt = matrix_power(cov_Xt, power_exp)
        cor_Xt = correlation.symmetrize(cov2corr(covp_Xt))
        L_Xt = manifold.deformation(cor_Xt)
        L_Xt =   L_Xt+cov_Xt
        L_Xt1 = manifold1.deformation(cor_Xt)

        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    n_freq = cov_Xs.shape[2]

    # 创建模型
    if distance == "olm":
        model = StiefelOLMTransform(
            d=d, 
            device=DEVICE, 
            dtype=DTYPE, 
            use_translation=True,
            use_modrelu=False
        )
    elif distance == "lsm":
        model = StiefelLSMTransform(
            d=d, 
            device=DEVICE, 
            dtype=DTYPE, 
            use_translation=True,
            use_modrelu=False
        )
    
    model = model.to(device=DEVICE, dtype=DTYPE)
    
    spdsw = SPDSW(d, n_proj, device=DEVICE, dtype=DTYPE, random_state=seed)
    optimizer = setup_stifel_optimizer(model, base_lr=lr, lr_mode=lr_mode)

    # 训练并记录loss
    start = time.time()
    pbar = trange(target_epochs, desc=f"Training {target_epochs} epochs (subject={subject}, seed={seed})")
    
    loss_history = []
    
    for epoch in pbar:
        model.train()
        
        zs = model(L_Xs)
        loss = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        
        for f in range(n_freq):
            loss += spdsw.spdsw(zs[:, 0, f], L_Xt[:, 0, f], p=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # 记录loss到TensorBoard
        if writer is not None and exp_tag:
            writer.add_scalar(f'loss/{exp_tag}', loss_val, epoch)
        
        if torch.isnan(loss):
            print(f"!!! Loss became NaN at epoch {epoch}, training crashed !!!")
            break
        
        pbar.set_postfix_str(f"loss = {loss_val:.3f}")

    stop = time.time()
    runtime = stop - start

    # 评估
    model.eval()
    with torch.no_grad():
        L_Xs1_bar = model(L_Xs1)

    s_noalign = get_svc(
        L_Xs1[:, 0].detach().cpu(), L_Xt1[:, 0].detach().cpu(),
        ys.detach().cpu(), yt.detach().cpu(),
        d, multifreq, n_jobs=N_JOBS, random_state=seed
    )
    s_align = get_svc(
        L_Xs1_bar[:, 0].detach().cpu(), L_Xt1[:, 0].detach().cpu(),
        ys.detach().cpu(), yt.detach().cpu(),
        d, multifreq, n_jobs=N_JOBS, random_state=seed
    )

    return s_noalign, s_align, runtime, loss_history

def compute_aggregated_metrics(results_df, distance):
    """计算聚合后的指标（不包含std）"""
    subjects = sorted(results_df['subject'].unique())
    
    L_session = {distance: {}, "no_align": {}}
    
    for s in subjects:
        filt = (results_df["distance"] == distance) & (results_df["subject"] == s)
        scores_align = results_df[filt]["align"]
        scores_noalign = results_df[filt]["no_align"]
        times = results_df[filt]["time"]
        
        L_session[distance][s] = {
            "mean_score": scores_align.mean(),
            "mean_time": times.mean(),
        }
        
        L_session["no_align"][s] = {
            "mean_score": scores_noalign.mean(),
        }
    
    acc_align_per_subject = [L_session[distance][s]["mean_score"] for s in subjects]
    acc_noalign_per_subject = [L_session["no_align"][s]["mean_score"] for s in subjects]
    time_per_subject = [L_session[distance][s]["mean_time"] for s in subjects]
    
    final_acc_align = np.mean(acc_align_per_subject)
    final_acc_noalign = np.mean(acc_noalign_per_subject)
    final_time = np.mean(time_per_subject)
    
    return {
        'align_mean': final_acc_align,
        'noalign_mean': final_acc_noalign,
        'time_mean': final_time,
        'per_subject_align': acc_align_per_subject,
        'per_subject_noalign': acc_noalign_per_subject,
        'subjects': subjects
    }

if __name__ == "__main__":
    # Set default hyperparameters based on task
    if args.task == "session":
        default_lr = 1e-3 if args.lr is None else args.lr
        default_power = 0.25 if args.power is None else args.power
        task_tag = "cross_session"
    else:
        default_lr = 5e-4 if args.lr is None else args.lr
        default_power = 0.85 if args.power is None else args.power
        task_tag = "cross_subject"
    
    target_epochs = args.epochs  # 直接使用指定的epochs
    
    print(f"Training for {target_epochs} epochs")
    
    # 生成所有种子
    all_seeds = RNG.choice(10000, NTRY, replace=False)
    
    # 设置文件名和TensorBoard
    lr_tag = _fmt_lr_tag(default_lr)
    power_tag = _fmt_power_tag(default_power)
    
    os.makedirs(os.path.join(EXPERIMENTS, "results"), exist_ok=True)
    
    # TensorBoard设置
    run_name = f"stifel_notrans_{task_tag}_{args.lr_mode}_{args.distance}_lr_{lr_tag}_power_{power_tag}_epoch{target_epochs}"
    log_dir = os.path.join("/root/tf-logs/", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"[TensorBoard] Logging to: {log_dir}")
    print(f"To view: tensorboard --logdir=/root/tf-logs/")
    print(f"使用学习率模式: {args.lr_mode}")
    print(f"基础学习率: {default_lr}")
    print(f"Power值: {default_power}")
    print(f"训练轮数: {target_epochs}")
    
    # CSV文件名
    csv_filename = f"stifel_notrans_{task_tag}_{args.lr_mode}_{args.distance}_lr_{lr_tag}_power_{power_tag}_epoch{target_epochs}_ntry{NTRY}.csv"
    results_file = os.path.join(EXPERIMENTS, "results", csv_filename)
    
    # 检查是否已有结果文件（支持断点恢复）
    if os.path.exists(results_file):
        print(f"Found existing results, loading...")
        try:
            existing_df = pd.read_csv(results_file)
            completed_seeds = set(existing_df['seed'].unique())
            remaining_seeds = [s for s in all_seeds if s not in completed_seeds]
            
            if len(remaining_seeds) == 0:
                print(f"All seeds completed, showing results...")
                
                # 计算并显示结果
                epoch_metrics = compute_aggregated_metrics(existing_df, args.distance)
                
                print(f"\n=== Results for {target_epochs} epochs ===")
                print(f"  Align: {epoch_metrics['align_mean']:.3f}")
                print(f"  No-Align: {epoch_metrics['noalign_mean']:.3f}")
                print(f"  Time: {epoch_metrics['time_mean']:.1f}s")
                
                # 显示每个subject的结果
                print(f"\nPer-subject align accuracy (averaged over {NTRY} seeds):")
                for idx, s in enumerate(epoch_metrics['subjects']):
                    print(f"  Subject {s}: {epoch_metrics['per_subject_align'][idx]:.3f}")
                
                exit(0)
            else:
                print(f"Found {len(completed_seeds)} completed seeds, {len(remaining_seeds)} remaining")
                seeds_to_run = remaining_seeds
                all_results = existing_df.to_dict('records')
        except Exception as e:
            print(f"Error loading existing results: {e}")
            seeds_to_run = all_seeds
            all_results = []
    else:
        seeds_to_run = all_seeds
        all_results = []
    
    # 记录所有loss曲线用于平均
    all_loss_histories = []
    
    # 对每个种子进行实验
    for try_idx, seed in enumerate(seeds_to_run):
        print(f"\n[Seed {try_idx + 1}/{len(seeds_to_run)}] Seed: {seed}")
        
        # 定义当前种子的超参数
        hyperparams = {
            "distance": [args.distance],
            "n_proj": [500],
            "seed": [seed],
            "subject": [1, 3, 7, 8, 9],
            "multifreq": [False],
            "reg": [10.],
            "lr": [default_lr],
            "power": [default_power],
            "lr_mode": [args.lr_mode],
        }

        if args.task == "session":
            hyperparams["cross_subject"] = [False]
            hyperparams["target_subject"] = [0]
        else:
            hyperparams["cross_subject"] = [True]
            hyperparams["target_subject"] = [1, 3, 7, 8, 9]

        # 生成参数组合
        keys, values = zip(*hyperparams.items())
        permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # 对当前种子的所有参数组合进行实验
        for i, params in enumerate(permuts_params):
            try:
                print(f"  [Exp {i+1}/{len(permuts_params)}] subject={params['subject']}, target={params.get('target_subject', 'N/A')}")
                
                if not params["cross_subject"]:
                    params["target_subject"] = 0
                if params["distance"] != "les":
                    params["reg"] = 1.

                # 生成实验标签用于TensorBoard
                exp_tag = f"s{params['subject']}_t{params.get('target_subject', 0)}_seed{seed}"
                
                # 运行训练并记录loss
                s_noalign, s_align, runtime, loss_history = run_test_with_loss_tracking(
                    params, target_epochs, writer, exp_tag
                )
                
                all_loss_histories.append(loss_history)

                result_dict = params.copy()
                result_dict["align"] = s_align
                result_dict["no_align"] = s_noalign
                result_dict["time"] = runtime
                result_dict["ntry_round"] = try_idx + 1
                result_dict["n_epochs"] = target_epochs
                
                if hasattr(result_dict["seed"], 'item'):
                    result_dict["seed"] = result_dict["seed"].item()
                
                all_results.append(result_dict)
                
                print(f"    Results: align={s_align:.3f}, no_align={s_noalign:.3f}, time={runtime:.1f}s")

            except (KeyboardInterrupt, SystemExit):
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                print("完整错误堆栈:")
                traceback.print_exc()
                continue
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Saved results to: {results_file}")
        print(f"Total experiments: {len(results_df)}")
        
        # 计算聚合指标
        epoch_metrics = compute_aggregated_metrics(results_df, args.distance)
        
        # 写入最终指标到TensorBoard（不包含std）
        writer.add_scalar('final/accuracy_align', epoch_metrics['align_mean'], 0)
        writer.add_scalar('final/accuracy_no_align', epoch_metrics['noalign_mean'], 0)
        writer.add_scalar('final/time_mean', epoch_metrics['time_mean'], 0)
        
        # 计算并记录平均loss曲线
        if all_loss_histories:
            avg_loss = np.mean(all_loss_histories, axis=0)
            for epoch, loss_val in enumerate(avg_loss):
                writer.add_scalar('loss/average', loss_val, epoch)
        
        # 记录每个subject的结果
        for idx, s in enumerate(epoch_metrics['subjects']):
            writer.add_scalar(f'per_subject/align_s{s}', epoch_metrics['per_subject_align'][idx], 0)
            writer.add_scalar(f'per_subject/noalign_s{s}', epoch_metrics['per_subject_noalign'][idx], 0)
        
        writer.close()
        
        print(f"\n=== Final Results ({target_epochs} epochs) ===")
        print(f"  Align: {epoch_metrics['align_mean']:.3f}")
        print(f"  No-Align: {epoch_metrics['noalign_mean']:.3f}")
        print(f"  Time: {epoch_metrics['time_mean']:.1f}s")
        
        # 显示每个subject的结果
        print(f"\nPer-subject align accuracy (averaged over {NTRY} seeds):")
        for idx, s in enumerate(epoch_metrics['subjects']):
            print(f"  Subject {s}: {epoch_metrics['per_subject_align'][idx]:.3f}")
    
    print(f"\n[TensorBoard] Results logged to: {log_dir}")
    print(f"To view: tensorboard --logdir=/root/tf-logs/")
    print("Experiments completed!")
