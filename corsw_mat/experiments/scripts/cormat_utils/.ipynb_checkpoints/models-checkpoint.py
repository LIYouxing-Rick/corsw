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



class Translation(nn.Module):
    def __init__(self, d, n_freq, device, dtype):
        super().__init__()

        manifold_spdai = geoopt.SymmetricPositiveDefinite("AIM")        
        self._W = ManifoldParameter(
            torch.eye(
                d,
                dtype=dtype,
                device=device
            )[None, :].repeat(n_freq, 1, 1),
            manifold=manifold_spdai
        )

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))


class Rotation(nn.Module):
    def __init__(self, d, n_freq, device, dtype):
        super().__init__()

        manifold = Stiefel()        
        self._W = ManifoldParameter(
            torch.eye(
                d,
                dtype=dtype,
                device=device)[None, :].repeat(n_freq, 1, 1),
            manifold=manifold
        )

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))


class Transformations(nn.Module):
    def __init__(self, d, n_freq, device, dtype, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.translation = Translation(d, n_freq, device, dtype)
        self.rotation = Rotation(d, n_freq, device, dtype)

    def forward(self, X):
        Y = self.translation(X)
        Y = self.rotation(Y)
        return Y
    


class CayleyOrthogonal(nn.Module):
    """
    生成 m×m 的（近）正交矩阵 O。
    - 训练参数：C（无约束欧式参数）
    - 前向：S = C - C^T（斜对称），B = (I + S) [+ eps*I]，O = (I - S) @ B^{-1}
    - 当 eps=0 且 (I+S) 可逆时，O 严格正交并且 det(O)=+1（Cayley 变换）。
      当 eps>0 时，O 为“近似正交”，换来更稳健的可逆性。
    """
    def __init__(self, m, eps: float = 1e-6, dtype=torch.float64, device="cuda:0"):
        super().__init__()
        self.m = m
        self.eps = float(eps)
        self.C = nn.Parameter(torch.zeros(m, m, dtype=dtype, device=device))

    def forward(self):
        C = self.C
        S = C - C.transpose(-1, -2)                          # 斜对称
        I = torch.eye(self.m, dtype=C.dtype, device=C.device)
        B = I + S if self.eps == 0.0 else (I + S + self.eps * I)
        O = (I - S) @ torch.linalg.solve(B, I)               # solve 比显式逆稳定
        return O

    @property
    def eps_value(self) -> float:
        return self.eps


class OrthLinear(nn.Module):
    """
    正交线性层（无偏置）：
        y = O x     （代码采用行批：y = x @ O^T）
    - O：由 CayleyOrthogonal 生成的 m×m 正交/近正交权重
    - 无偏置项
    输入 x: 形状 [..., m]；输出同形状。
    """
    def __init__(self, m, eps: float = 0.0, dtype=torch.float64, device="cuda:0", seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.O_layer = CayleyOrthogonal(m, eps=eps, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, m] 或 [m]
        我们想要 y = O x + b（列向量约定），
        对 batch 行向量实现成 y = x @ O^T + b。
        """
        O = self.O_layer()                       # [m, m]
        if x.dim() == 1:
            y = (O @ x)                          # [m]
        else:
            y = x @ O.transpose(-1, -2)          # [B, m]
        return y

    @property
    def weight(self) -> torch.Tensor:
        """便捷获取当前的 O。"""
        return self.O_layer()

    def orthogonality_error(self) -> torch.Tensor:
        """返回 ||O^T O - I||_F，用于监控正交性（eps>0 时可观察误差）。"""
        O = self.weight
        I = torch.eye(O.shape[-1], dtype=O.dtype, device=O.device)
        return torch.linalg.matrix_norm(O.transpose(-1, -2) @ O - I)

    @property
    def O(self):
        # 便于外部访问当前正交矩阵
        return self.O_layer()


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

    
def get_svc(L_Xs, L_Xt, ys, yt, d, multifreq=False, n_jobs=50, random_state=None, kernel=False):
    _,_,m = L_Xs.shape
    
    # 确保所有输入都在 CPU 上并转换为 numpy 数组，需要先 detach() 分离梯度
    if torch.is_tensor(L_Xs):
        L_Xs = L_Xs.detach().cpu().numpy()
    if torch.is_tensor(L_Xt):
        L_Xt = L_Xt.detach().cpu().numpy()
    if torch.is_tensor(ys):
        ys = ys.detach().cpu().numpy()
    if torch.is_tensor(yt):
        yt = yt.detach().cpu().numpy()
    
    if multifreq:
        pass  # 保持原来的形状
    if not multifreq:
        L_Xs = L_Xs.reshape(-1, m)
        L_Xt = L_Xt.reshape(-1, m)

    if multifreq:
        raise NotImplementedError("multifreq not implemented yet")
    
    elif kernel:
        raise NotImplementedError("kernel not implemented yet")
    else:
        clf = GridSearchCV(
            LinearSVC(random_state=random_state),
            {"C": np.logspace(-2, 2, 100)},
            n_jobs=n_jobs
        )
        
    clf.fit(L_Xs, ys)
    return clf.score(L_Xt, yt)