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
from corswmat.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric


class TranslationLayer(nn.Module):
    """
    实现 SPD 流形上的平移操作 Y = P * X * P。
    权重 P 是一个可学习的 SPD 矩阵。
    """
    def __init__(self, d: int, device, dtype):
        super().__init__()
        manifold_spd = geoopt.SymmetricPositiveDefinite("AIM")        
        self.P = geoopt.ManifoldParameter(
            torch.eye(d, dtype=dtype, device=device),
            manifold=manifold_spd
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d)
        bs = X.shape[0]
        
        # 方法1：reshape处理
        X_reshaped = X.view(bs, X.shape[-2], X.shape[-1])  # (bs, d, d)
        
        # 扩展P以匹配batch维度
        P_expanded = self.P.unsqueeze(0).expand(bs, -1, -1)  # (bs, d, d)
        
        # 执行批量矩阵乘法
        result = torch.bmm(torch.bmm(P_expanded, X_reshaped), P_expanded)  # (bs, d, d)
        
        # 恢复原始形状
        return result.view(bs, 1, 1, X.shape[-2], X.shape[-1])  # (bs, 1, 1, d, d)


class CayleyOrthogonal(nn.Module):
    """
    生成 m×m 的（近）正交矩阵 O。
    """
    def __init__(self, m, device="cuda:0", dtype=torch.float64, eps: float = 1e-6):
        super().__init__()
        self.m = m
        self.eps = float(eps)
        self.C = nn.Parameter(torch.zeros(m, m, device=device, dtype=dtype))
        self.register_buffer('I', torch.eye(m, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d)
        bs = X.shape[0]
        X_reshaped = X.view(bs, X.shape[-2], X.shape[-1])  # (bs, d, d)
        
        C = self.C
        S = C - C.transpose(-1, -2)  # 斜对称
        
        if self.eps == 0.0:
            B = self.I + S
        else:
            B = self.I + S + self.eps * self.I
            
        try:
            B_inv = torch.linalg.inv(B)
            O = (self.I - S) @ B_inv
        except:
            O = torch.linalg.solve(B.transpose(-1, -2), (self.I - S).transpose(-1, -2)).transpose(-1, -2)
        
        # 扩展O以匹配batch维度
        O_expanded = O.unsqueeze(0).expand(bs, -1, -1)  # (bs, d, d)
        O_T_expanded = O_expanded.transpose(-1, -2)  # (bs, d, d)
        
        # 批量矩阵乘法
        result = torch.bmm(torch.bmm(O_expanded, X_reshaped), O_T_expanded)  # (bs, d, d)
        
        # 恢复原始形状
        return result.view(bs, 1, 1, X.shape[-2], X.shape[-1])  # (bs, 1, 1, d, d)

    @property
    def eps_value(self) -> float:
        return self.eps


class CayleyOLMTransform(nn.Module):
    """
    复合模型，依次执行平移、旋转和 Off(...) 拉回操作。
    """
    def __init__(self, d: int, device, dtype):
        super().__init__()
        self.d = d
        self.device = device
        self.dtype = dtype
        self.rotation = CayleyOrthogonal(d, device, dtype)
        self.translation = TranslationLayer(d, device, dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d)
        
        # 1. 平移
        translated_x = self.translation(X)  # (bs, 1, 1, d, d)
        
        # 2. 旋转 (修复变量名错误)
        rotated_x = self.rotation(translated_x)  # (bs, 1, 1, d, d)
        
        # 3. 拉回 (Off 操作) - 移除对角元素
        bs = rotated_x.shape[0]
        mask = ~torch.eye(self.d, dtype=torch.bool, device=self.device)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d, d)
        mask = mask.expand(bs, 1, 1, -1, -1)  # (bs, 1, 1, d, d)
        
        final_result = rotated_x * mask.to(self.dtype)
        
        return final_result

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d})"


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