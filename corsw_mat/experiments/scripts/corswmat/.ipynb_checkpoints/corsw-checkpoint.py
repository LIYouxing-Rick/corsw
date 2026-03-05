import torch

import numpy as np
import torch.nn.functional as F

from geoopt import linalg


from corsw.CorMatrix import CorEuclideanCholeskyMetric,CorLogEuclideanCholeskyMetric,CorOffLogMetric,CorLogScaledMetric

class CORSW:
    """
        Class for computing SPDSW distance and embedding

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        num_ts : int
            Number of timestamps for quantiles, default 20
        device : str
            Device for computations, default None
        dtype : type
            Data type, default torch.float
        random_state : int
            Seed, default 123456
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
            Default "spdsw"
        """

    def __init__(
        self,
        shape_X,
        num_projections,
        num_ts=20,
        device=None,
        dtype=torch.float,
        random_state=123456,
        sampling="ecm",
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sampling not in ["ecm", "lecm", "olm", "lsm"]:
            raise Exception(
                "'sampling' should be in ['ecm', 'lecm', 'olm', 'lsm']"
            )

        self.generate_projections(
            shape_X, num_projections, num_ts,
            device, dtype, random_state, sampling,vecterization=True
        )

        self.sampling = sampling

   # 将 sampling 和 vecterization 移动到 random_state 前面
    def generate_projections(self, shape_X, num_projections, num_ts, device, dtype, random_state, sampling, vecterization=True):
    # ...
        """
        Generate projections for sampling
        ... (文档字符串不变) ...
        """
        # NumPy的rng不再需要
        # rng = np.random.default_rng(random_state)
        
        self.ts = torch.linspace(0, 1, num_ts, dtype=dtype, device=device)

        # --- 主要修改在这里 ---
        # 1. 创建一个本地的、可复现的随机数生成器
        g = torch.Generator(device=device).manual_seed(random_state)
        
        # 2. 使用 torch.randn 并传入该生成器，替代 empty().normal_()
        shape = (num_projections, shape_X * (shape_X - 1) // 2)
        self.A = torch.randn(shape, generator=g, device=device, dtype=dtype)
        
        # 后续的归一化代码不变
        scale = self.A.mul(self.A).sum(dim=1, keepdim=True).add_(1e-12).rsqrt_()
        self.A.mul_(scale)
        
        if vecterization:
            return self.A
        else:
            raise NotImplementedError("matrix-space projection path is not implemented yet")
        

    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None, p=1):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        # Sort
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

        # Compute CDF
        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

        u_index = torch.searchsorted(u_cdf, cdf_axis)
        v_index = torch.searchsorted(v_cdf, cdf_axis)

        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

        cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)

        return torch.sum(
            delta * torch.pow(torch.abs(u_icdf - v_icdf), p),
            axis=-1
        )

    def corsw(self, L_Xs, L_Xt, u_weights=None, v_weights=None, p=2):
        """
            Parameters:
            Xs: ndarray, shape (n_batch, d, d)
                Samples in the source domain
            Xt: ndarray, shape (m_batch, d, d)
                Samples in the target domain
            device: str
            p: float
                Power of SW. Need to be >= 1.
        """
        d=22
        n, _ = L_Xs.shape
        m, _ = L_Xt.shape

        n_proj = self.A.shape[0]

        

        prod_Xs = (self.A[None] * L_Xs[:, None]).reshape(n, n_proj, -1)
        prod_Xt = (self.A[None] * L_Xt[:, None]).reshape(m, n_proj, -1)

        Xps = prod_Xs.sum(-1)
        Xpt = prod_Xt.sum(-1)

        return torch.mean(
            self.emd1D(Xps.T, Xpt.T, u_weights, v_weights, p)
        )

    def get_quantiles(self, x, ts, weights=None):
        """
            Inputs:
            - x: 1D values, size: n_projs * n_batch
            - ts: points at which to evaluate the quantile
        """
        n_projs, n_batch = x.shape

        if weights is None:
            X_weights = torch.full(
                (n_batch,), 1/n_batch, dtype=x.dtype, device=x.device
            )
            X_values, X_sorter = torch.sort(x, -1)
            X_weights = X_weights[..., X_sorter]

        X_cdf = torch.cumsum(X_weights, -1)

        X_index = torch.searchsorted(X_cdf, ts.repeat(n_projs, 1))
        X_icdf = torch.gather(X_values, -1, X_index.clip(0, n_batch-1))

        return X_icdf

    def get_features(self, x, weights=None, p=2):
        """
            Inputs:
            - x: ndarray, shape (n_batch, d, d)
                Samples of SPD
            - weights: weight of each sample, if None, uniform weights
            - p
        """
        num_unifs = len(self.ts)
        n_proj, d, _ = self.A.shape
        n, _, _ = x.shape

        if self.sampling in ["spdsw", "logsw"]:
            log_x = linalg.sym_logm(x)
            Xp = (self.A[None] * log_x[:, None]).reshape(n, n_proj, -1).sum(-1)
        elif self.sampling == "sw":
            Xp = (self.A[None] * x[:, None]).reshape(n, n_proj, -1).sum(-1)
        elif self.sampling == "aispdsw":
            x2 = torch.matmul(
                torch.matmul(
                    torch.transpose(self.P, -2, -1)[:, None],
                    x[None]
                ),
                self.P[:, None]
            )
            LD, pivots = torch.linalg.ldl_factor(x2)
            P, L, D_x = torch.lu_unpack(LD, pivots)
            log_x = torch.transpose(linalg.sym_logm(D_x), 0, 1)
            Xp = (self.A[None]*log_x).reshape(n, n_proj, -1).sum(-1)

        q_Xp = self.get_quantiles(Xp.T, self.ts, weights)

        return q_Xp / ((n_proj * num_unifs) ** (1 / p))
