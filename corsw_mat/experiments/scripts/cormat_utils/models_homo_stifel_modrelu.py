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


import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter, Stiefel
from geoopt.optim import RiemannianSGD

class ModReLU(nn.Module):
    """
    ModReLU激活函数的实现
    σ_modReLU(z) = sign(z) * ReLU(|z| + b)
    """
    def __init__(self, features):
        super(ModReLU, self).__init__()
        self.features = features
        # 可训练的偏置参数
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, input):
        # input shape: (..., features)
        # 计算 |z| + b
        abs_input = torch.abs(input)
       # biased_abs = abs_input + self.bias
        
        # 应用ReLU到 |z| + b
        activated_output = F.relu(abs_input)
        
        # 计算 sign(z) * ReLU(|z| + b)
        output = torch.sign(input) * activated_output
        
        return output


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


class Rotation(nn.Module):
    """
    Stiefel 流形上的正交旋转
    """
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d) 
        bs = X.shape[0]
        X_reshaped = X.view(bs, X.shape[-2], X.shape[-1])  # (bs, d, d)
        
        # 获取旋转矩阵 (n_freq, d, d)，这里 n_freq=1
        W = self._W[0]  # (d, d)
        
        # 扩展W以匹配batch维度
        W_expanded = W.unsqueeze(0).expand(bs, -1, -1)  # (bs, d, d)
        W_T_expanded = W_expanded.transpose(-1, -2)  # (bs, d, d)
        
        # 批量矩阵乘法: W @ X @ W^T
        result = torch.bmm(torch.bmm(W_expanded, X_reshaped), W_T_expanded)  # (bs, d, d)
        
        # 恢复原始形状
        return result.view(bs, 1, 1, X.shape[-2], X.shape[-1])  # (bs, 1, 1, d, d)


class StiefelOLMTransform(nn.Module):
    """
    使用Stiefel流形旋转的OLM变换模型
    执行：平移 -> 旋转 -> bias -> ModReLU -> Off(...) 拉回操作
    """
    def __init__(self, d: int, device, dtype, use_translation=True, use_modrelu=True):
        super().__init__()
        self.d = d
        self.device = device
        self.dtype = dtype
        self.use_translation = use_translation
        self.use_modrelu = use_modrelu
        
        # 旋转层 (Stiefel流形参数)
        self.rotation = Rotation(d, n_freq=1, device=device, dtype=dtype)
        
        # 平移层 (SPD参数) - 可选
        if use_translation:
            self.translation = TranslationLayer(d, device, dtype)
        
        # Bias参数 - 针对矩阵的每个元素
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, d, d, device=device, dtype=dtype))
        
        # ModReLU激活函数 - 针对矩阵元素
        if use_modrelu:
            # 对于矩阵，我们将其展平后应用ModReLU
            self.activation = ModReLU(d * d)
        else:
            self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d)
        
        # 1. 平移 (如果启用)
        if self.use_translation:
            translated_x = self.translation(X)  # (bs, 1, 1, d, d)
        else:
            translated_x = X
        
        # 2. Stiefel旋转
        rotated_x = self.rotation(translated_x)  # (bs, 1, 1, d, d)
        
        # 3. 添加bias
        biased_x = rotated_x + self.bias  # (bs, 1, 1, d, d)
        
        # 4. 应用ModReLU激活函数
        bs = biased_x.shape[0]
        if self.use_modrelu:
            # 展平矩阵以应用ModReLU
            flattened = biased_x.view(bs, -1)  # (bs, d*d)
            activated = self.activation(flattened)  # (bs, d*d)
            activated_x = activated.view(bs, 1, 1, self.d, self.d)  # (bs, 1, 1, d, d)
        else:
            activated_x = self.activation(biased_x)
        
        # 5. 拉回 (Off 操作) - 移除对角元素
        mask = ~torch.eye(self.d, dtype=torch.bool, device=self.device)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, d, d)
        mask = mask.expand(bs, 1, 1, -1, -1)  # (bs, 1, 1, d, d)
        
        final_result = activated_x * mask.to(self.dtype)
        
        return final_result
    
    def get_spd_parameters(self):
        """获取SPD流形参数（Translation的P参数）"""
        spd_params = []
        if self.use_translation:
            for name, param in self.translation.named_parameters():
                if hasattr(param, 'manifold'):  # 检查是否是流形参数
                    spd_params.append(param)
        return spd_params
    
    def get_stiefel_parameters(self):
        """获取Stiefel流形参数（Rotation的_W参数）"""
        stiefel_params = []
        for name, param in self.rotation.named_parameters():
            if '_W' in name:  # Stiefel参数
                stiefel_params.append(param)
        return stiefel_params
    
    def get_euclidean_parameters(self):
        """获取欧氏参数（bias和ModReLU参数）"""
        euclidean_params = []
        
        # bias参数
        euclidean_params.append(self.bias)
        
        # ModReLU参数
        if self.use_modrelu:
            for param in self.activation.parameters():
                euclidean_params.append(param)
        
        return euclidean_params

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d}, use_translation={self.use_translation}, use_modrelu={self.use_modrelu})"


class StiefelLSMTransform(nn.Module):
    """
    使用Stiefel流形旋转的LSM变换模型
    执行：平移 -> 旋转 -> bias -> ModReLU -> phi(...) 拉回操作
    """
    def __init__(self, d: int, device, dtype, use_translation=True, use_modrelu=True):
        super().__init__()
        self.d = d
        self.device = device
        self.dtype = dtype
        self.use_translation = use_translation
        self.use_modrelu = use_modrelu
        
        # 旋转层 (Stiefel流形参数)
        self.rotation = Rotation(d, n_freq=1, device=device, dtype=dtype)
        
        # 平移层 (SPD参数) - 可选
        if use_translation:
            self.translation = TranslationLayer(d, device, dtype)
        
        # Bias参数
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, d, d, device=device, dtype=dtype))
        
        # ModReLU激活函数
        if use_modrelu:
            self.activation = ModReLU(d * d)
       
        
        # 预计算常用向量
        self.register_buffer('ones_vector', torch.ones(d, 1, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X shape: (bs, 1, 1, d, d)
        
        # 1. 平移 (如果启用)
        if self.use_translation:
            translated_x = self.translation(X)  # (bs, 1, 1, d, d)
        else:
            translated_x = X
        
        # 2. Stiefel旋转
        rotated_x = self.rotation(translated_x)  # (bs, 1, 1, d, d)
        
        # 3. 添加bias
        biased_x = rotated_x + self.bias  # (bs, 1, 1, d, d)
        
        # 4. 应用ModReLU激活函数
        bs = biased_x.shape[0]
        if self.use_modrelu:
            # 展平矩阵以应用ModReLU
            flattened = biased_x.view(bs, -1)  # (bs, d*d)
            activated = self.activation(flattened)  # (bs, d*d)
            activated_x = activated.view(bs, 1, 1, self.d, self.d)  # (bs, 1, 1, d, d)
            biased_x = activated_x
      
        
        # 5. 拉回 (phi 操作)
        biased_x_reshaped =  biased_x.view(bs, self.d, self.d)  # (bs, d, d)
        
        # 计算行和
        ones_expanded = self.ones_vector.unsqueeze(0).expand(bs, -1, -1)  # (bs, d, 1)
        row_sums = torch.bmm( biased_x_reshaped, ones_expanded)  # (bs, d, 1)
        
        # 创建对角矩阵
        diag_sums = torch.diag_embed(row_sums.squeeze(-1))  # (bs, d, d)
        
        # 减去对角矩阵
        final_result = biased_x_reshaped - diag_sums  # (bs, d, d)
        
        # 恢复原始形状
        return final_result.view(bs, 1, 1, self.d, self.d)  # (bs, 1, 1, d, d)
    
    def get_spd_parameters(self):
        """获取SPD流形参数（Translation的P参数）"""
        spd_params = []
        if self.use_translation:
            for name, param in self.translation.named_parameters():
                if hasattr(param, 'manifold'):  # 检查是否是流形参数
                    spd_params.append(param)
        return spd_params
    
    def get_stiefel_parameters(self):
        """获取Stiefel流形参数（Rotation的_W参数）"""
        stiefel_params = []
        for name, param in self.rotation.named_parameters():
            if '_W' in name:  # Stiefel参数
                stiefel_params.append(param)
        return stiefel_params
    
    def get_euclidean_parameters(self):
        """获取欧氏参数（bias和ModReLU参数）"""
        euclidean_params = []
        
        # bias参数
        euclidean_params.append(self.bias)
        
        # ModReLU参数
        if self.use_modrelu:
            for param in self.activation.parameters():
                euclidean_params.append(param)
        
        return euclidean_params

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d}, use_translation={self.use_translation}, use_modrelu={self.use_modrelu})"

    
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