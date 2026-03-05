import torch
import geoopt

import torch.nn as nn
import numpy as np

from geoopt import ManifoldParameter, Stiefel, linalg
from functools import lru_cache, partial
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from corsw.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric


class CayleyOffTransform(nn.Module):
    """
    仿照示例风格实现的 f(X, M) = Off(M^T * X * M)。
    M 通过 Cayley 变换生成，并且被所有 n_freq 通道共享。

    参数:
        d (int): 输入和输出矩阵的维度 (n)。
        device: PyTorch 设备 (e.g., torch.device('cuda:0'))。
        dtype: PyTorch 数据类型 (e.g., torch.float32)。
    """
    def __init__(self, d: int, device, dtype):
        super().__init__()
        self.d = d
        self.device = device
        self.dtype = dtype

        # 1. 定义可学习的参数 A。
        #    形状为 (d, d)，这个 A 将被用于所有 n_freq 通道。
        self.A = nn.Parameter(torch.empty(d, d, device=device, dtype=dtype))
        
        # --- 如何修改以实现每个 n_freq 通道拥有独立的 M ---
        # 如果需要像您的示例一样为每个 n_freq 学习独立参数，请取消下面的注释
        # 并传入 n_freq 参数到 __init__
        # self.A = nn.Parameter(torch.empty(n_freq, d, d, device=device, dtype=dtype))
        # --------------------------------------------------------

        # 2. 初始化参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用小的均匀分布初始化 A。"""
        nn.init.uniform_(self.A, -0.01, 0.01)

    def _get_orthogonal_matrix(self) -> torch.Tensor:
        """通过 Cayley 变换从可学习参数 A 生成正交矩阵 M。"""
        # 从 A 生成斜对称矩阵 S = A - A^T
        S = self.A - self.A.transpose(-2, -1)
        
        # 创建单位矩阵 I
        I = torch.eye(self.d, device=self.device, dtype=self.dtype)
        
        # 如果 A 是 (f, d, d)，需要扩展 I
        # if self.A.dim() > 2:
        #    I = I.unsqueeze(0).expand_as(S)
        
        # 计算 M = (I - S) @ inv(I + S)
        M = torch.linalg.solve(I + S, I - S)
        return M

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播。

        输入:
            X (torch.Tensor): 形状为 (batch, channels, n_freq, d, d)。
        返回:
            torch.Tensor: 形状为 (batch, channels, n_freq, d, d)。
        """
        # 1. 动态生成正交矩阵 M
        M = self._get_orthogonal_matrix() # 形状: (d, d) 或 (f, d, d)

        # 2. 计算三明治乘积 M^T * X * M
        # PyTorch 的 @ 会自动处理广播，无需调整 M 的维度
        # M.T 的形状是 (d, d)，X 的形状是 (b, c, f, d, d)
        # 运算会自动进行，相当于在最后两个维度上应用变换
        sandwich_product = M.T @ X @ M

        # 3. Off(...) 操作: 将对角线置零
        diagonal_part = torch.diagonal(sandwich_product, dim1=-2, dim2=-1)
        off_diagonal_result = sandwich_product - torch.diag_embed(diagonal_part)

        return off_diagonal_result

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d})"
    

class CayleyLSMTransform(nn.Module):
    """
    仿照示例风格实现的 φ(M^T * X * M)。
    M 通过 Cayley 变换生成，并且被所有 n_freq 通道共享。

    参数:
        d (int): 输入和输出矩阵的维度 (n)。
        device: PyTorch 设备。
        dtype: PyTorch 数据类型。
    """
    def __init__(self, d: int, device, dtype):
        super().__init__()
        self.d = d
        self.device = device
        self.dtype = dtype

        # 可学习参数 A，与第一个模型完全相同
        self.A = nn.Parameter(torch.empty(d, d, device=device, dtype=dtype))
        
        # --- 如何修改以实现每个 n_freq 通道拥有独立的 M ---
        # self.A = nn.Parameter(torch.empty(n_freq, d, d, device=device, dtype=dtype))
        # --------------------------------------------------------
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数 A。"""
        nn.init.uniform_(self.A, -0.01, 0.01)

    def _get_orthogonal_matrix(self) -> torch.Tensor:
        """通过 Cayley 变换生成正交矩阵 M。"""
        S = self.A - self.A.transpose(-2, -1)
        I = torch.eye(self.d, device=self.device, dtype=self.dtype)
        # if self.A.dim() > 2:
        #    I = I.unsqueeze(0).expand_as(S)
        M = torch.linalg.solve(I + S, I - S)
        return M

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播。

        输入:
            X (torch.Tensor): 形状为 (batch, channels, n_freq, d, d)。
        返回:
            torch.Tensor: 形状为 (batch, channels, n_freq, d, d)。
        """
        # 1. 生成正交矩阵 M
        M = self._get_orthogonal_matrix() # 形状: (d, d)
        
        # 2. 计算三明治乘积 Y = M^T * X * M
        Y = M.T @ X @ M
        
        # 3. 应用 phi(Y) = Y - diag(Y1)
        ones_vector = torch.ones(self.d, 1, device=self.device, dtype=self.dtype)
        row_sums = Y @ ones_vector
        diag_Y1 = torch.diag_embed(row_sums.squeeze(-1))
        phi_Y = Y - diag_Y1

        return phi_Y

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d})"


# Taken from geoopt
# https://github.com/geoopt/geoopt/blob/master/geoopt/linalg/batch_linalg.py

@lru_cache(None)
def _sym_funcm_impl(func, **kwargs):
    func = partial(func, **kwargs)

    def _impl(x):
        e, v = torch.linalg.eigh(x, "U")
        return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)

    return torch.jit.script(_impl)


def sym_funcm(
    x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply function to symmetric matrix.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    func : Callable[[torch.Tensor], torch.Tensor]
        function to apply
    Returns
    -------
    torch.Tensor
        symmetric matrix with function applied to
    """
    return _sym_funcm_impl(func)(x)


def sym_reeig(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix exponent.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`\exp(x)`
    Notes
    -----
    Naive implementation of `torch.matrix_exp` seems to be fast enough
    """
    return sym_funcm(x, nn.Threshold(1e-4, 1e-4))


class FeaturesKernel(BaseEstimator, TransformerMixin):
    
    def __init__(self, sigma=1.):
        self.sigma = sigma
    
    def fit(self, X, y=None):
        self.X = X.astype(np.float64)
        self.N =  np.sum(self.X ** 2, axis=(2, 3))
        return self
    
    def transform(self, X, y=None):
        C = 1.
        X_d = X.astype(np.float64)
                
        N = np.sum(X_d ** 2, axis=(2, 3))
        for i in range(X_d.shape[1]):
            C1 = self.N[None, :, i] + N[:, i, None]
            C2 = X_d[:, i].reshape(X_d.shape[0], -1) @ self.X[:, i].reshape(self.X.shape[0], -1).T
            C_current = np.exp(-(C1 - 2 * C2) / (self.sigma ** 2))
            C += C_current
        
        return C 
    
    def get_params(self, deep=True):
        return {"sigma": self.sigma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
def get_svc(Xs, Xt, ys, yt, d, multifreq=False, n_jobs=50, random_state=None, kernel=False):
    log_Xs =Xs.detach().cpu() #.reshape(-1, d * d)
    log_Xt = Xt.detach().cpu() #.reshape(-1, d * d)
    
    if multifreq:
        log_Xs = log_Xs.numpy()
        log_Xt = log_Xt.numpy()
    if not multifreq:
        log_Xs = log_Xs.reshape(-1, d*d)
        log_Xt = log_Xt.reshape(-1, d*d)
    

    if multifreq:
        clf = GridSearchCV(
            make_pipeline(
                FeaturesKernel(),
                GridSearchCV(
                    SVC(random_state=random_state),
                    {"C": np.logspace(-2, 2, 10), "kernel": ["precomputed"]},
                    n_jobs=n_jobs
                )
            ),
            {"featureskernel__sigma": np.logspace(-1,1,num=10)}
        )
#         clf = make_pipeline(
#             FeaturesKernel(),
#             GridSearchCV(
#                 SVC(random_state=random_state),
#                 {"C": np.logspace(-2, 2, 10), "kernel": ["precomputed"]},
#                 n_jobs=n_jobs
#             )
#         )
    
    elif kernel:
        clf = make_pipeline(
            GridSearchCV(
                SVC(random_state=random_state), 
                {"C": np.logspace(-2, 2, 10)}, 
                n_jobs=n_jobs
            )
        )
    else:
        clf = GridSearchCV(
            LinearSVC(random_state=random_state),
            {"C": np.logspace(-2, 2, 100)},
            n_jobs=n_jobs
        )
        
    clf.fit(log_Xs, ys.cpu())
    return clf.score(log_Xt, yt.cpu())