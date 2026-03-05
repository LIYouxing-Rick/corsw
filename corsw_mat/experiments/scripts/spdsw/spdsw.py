import torch

import numpy as np
import torch.nn.functional as F

from geoopt import linalg


class SPDSW:
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
        sampling="spdsw",
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sampling not in ["spdsw", "logsw", "sw", "aispdsw","oula"]:
            raise Exception(
                "'sampling' should be in ['spdsw', 'logsw', 'sw', 'aispdsw']"
            )

        self.generate_projections(
            shape_X, num_projections, num_ts,
            device, dtype, random_state, sampling
        )

        self.sampling = sampling

    def generate_projections(self, shape_X, num_projections, num_ts,
                             device, dtype, random_state, sampling):
        """
        Generate projections for sampling

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        device : str
            Device for computations
        dtype : type
            Data type
        random_state : int
            Seed
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
        """

        rng = np.random.default_rng(random_state)

        if sampling == "spdsw":

            # Random projection directions, shape (d-1, num_projections)
            theta = rng.normal(size=(num_projections, shape_X))
            theta = F.normalize(
                torch.from_numpy(theta), p=2, dim=-1
            ).type(dtype).to(device)

            D = theta[:, None] * torch.eye(
                theta.shape[-1],
                device=device,
                dtype=dtype
            )

            # Random orthogonal matrices
            Z = rng.normal(size=(num_projections, shape_X, shape_X))
            Z = torch.tensor(
                Z,
                dtype=dtype,
                device=device
            )
            Q, R = torch.linalg.qr(Z)
            lambd = torch.diagonal(R, dim1=-2, dim2=-1)
            lambd = lambd / torch.abs(lambd)
            P = lambd[:, None] * Q

            self.A = torch.matmul(
                P,
                torch.matmul(D, torch.transpose(P, -2, -1))
            )

        elif sampling in ["logsw", "sw"]:

            self.A = torch.tensor(
                rng.normal(size=(num_projections, shape_X, shape_X)),
                dtype=dtype,
                device=device
            )

            self.A /= torch.norm(self.A, dim=(1, 2), keepdim=True)
            

        elif sampling == "aispdsw":
            # Random projection directions, shape (d-1, num_projections)
            theta = rng.normal(size=(num_projections, shape_X))
            theta = F.normalize(
                torch.from_numpy(theta), p=2, dim=-1
            ).type(dtype).to(device)
            theta_sorted, sorter = torch.sort(theta, descending=False)

            perm_matrix = F.one_hot(sorter).type(dtype)
            D_sorted = theta_sorted[:, None] * torch.eye(
                theta.shape[-1],
                dtype=dtype,
                device=device
            )

            # Random orthogonal matrices
            Z = rng.normal(size=(num_projections, shape_X, shape_X))
            Z = torch.tensor(
                Z,
                dtype=dtype,
                device=device
            )
            Q, R = torch.linalg.qr(Z)
            lambd = torch.diagonal(R, dim1=-2, dim2=-1)
            lambd = lambd / torch.abs(lambd)
            P = lambd[:, None] * Q

            self.P = torch.matmul(P, torch.transpose(perm_matrix, -2, -1))
            self.A = D_sorted
            
        elif sampling == "oula":
            
            sphere_dim = 254  # S^253 sphere embedded in R^254

            # Step 1: Sample unit vectors from S^253 sphere
            unit_vectors = rng.normal(size=(num_projections, sphere_dim))
            unit_vectors = torch.tensor(
                unit_vectors,
                dtype=dtype,
                device=device
            )
            unit_vectors = F.normalize(unit_vectors, p=2, dim=-1)
            
            # Step 2: Extract 253 components for symmetric matrix
            sym_matrix_dim = shape_X * (shape_X + 1) // 2  # 253 for 22×22
            vector_components = unit_vectors[:, :sym_matrix_dim]
            
            # Step 3: Construct symmetric matrices with proper scaling for isometry
            indices = torch.triu_indices(shape_X, shape_X)
            S = torch.zeros(
                (num_projections, shape_X, shape_X),
                dtype=dtype,
                device=device
            )
            
            # Create mask for diagonal elements
            diag_mask = indices[0] == indices[1]
            off_diag_mask = ~diag_mask
            
            # Fill the matrix with proper scaling
            # Diagonal elements: use as is
            S[:, indices[0][diag_mask], indices[1][diag_mask]] = vector_components[:, diag_mask]
            
            # Off-diagonal elements: divide by sqrt(2) because they appear twice in the symmetric matrix
            # This ensures ||vec||^2 = ||mat||_F^2
            sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            S[:, indices[0][off_diag_mask], indices[1][off_diag_mask]] = vector_components[:, off_diag_mask] / sqrt_2
            S[:, indices[1][off_diag_mask], indices[0][off_diag_mask]] = vector_components[:, off_diag_mask] / sqrt_2
            
            # Step 4: Scale for numerical stability (optional)
            S = S * 2.0  # Adjust as needed
            
            # Step 5: Map to SPD manifold via matrix exponential
            eigenvalues, eigenvectors = torch.linalg.eigh(S)
            exp_eigenvalues = torch.exp(eigenvalues)
            self.A = torch.matmul(
                eigenvectors,
                torch.matmul(
                    torch.diag_embed(exp_eigenvalues),
                    eigenvectors.transpose(-2, -1)
                )
            )



        self.ts = torch.linspace(0, 1, num_ts, dtype=dtype, device=device)

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

    def spdsw(self, Xs, Xt, u_weights=None, v_weights=None, p=2):
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
        n, _, _ = Xs.shape  #number
        m, _, _ = Xt.shape  #number

        n_proj, d, _ = self.A.shape  #d=21

        if self.sampling in ["spdsw", "logsw", "oula"]:
            # Busemann Coordinates
              #Xs=Sym_Xs
               #Xt=Sym_Xt

            prod_Xs = (self.A[None] * Xs[:, None]).reshape(n, n_proj, -1)
            prod_Xt = (self.A[None] * Xt[:, None]).reshape(m, n_proj, -1)

        elif self.sampling in ["sw"]:
            # Euclidean Coordinates
            prod_Xs = (self.A[None] * Xs[:, None]).reshape(n, n_proj, -1)
            prod_Xt = (self.A[None] * Xt[:, None]).reshape(m, n_proj, -1)

        elif self.sampling in ["aispdsw"]:
            Xs2 = torch.matmul(
                torch.matmul(
                    torch.transpose(self.P, -2, -1)[:, None],
                    Xs[None]
                ),
                self.P[:, None]
            )
            Xt2 = torch.matmul(
                torch.matmul(
                    torch.transpose(self.P, -2, -1)[:, None],
                    Xt[None]
                ),
                self.P[:, None]
            )

            LD, pivots = torch.linalg.ldl_factor(Xs2)
            P, L, D_Xs = torch.lu_unpack(LD, pivots)

            LD, pivots = torch.linalg.ldl_factor(Xt2)
            P, L, D_Xt = torch.lu_unpack(LD, pivots)

            log_Xs = torch.transpose(linalg.sym_logm(D_Xs), 0, 1)
            log_Xt = torch.transpose(linalg.sym_logm(D_Xt), 0, 1)

            prod_Xs = (self.A[None]*log_Xs).reshape(n, n_proj, -1)
            prod_Xt = (self.A[None]*log_Xt).reshape(m, n_proj, -1)

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
