"""
SPD computations under LieBN:
    @inproceedings{chen2024liebn,
        title={A Lie Group Approach to Riemannian Batch Normalization},
        author={Ziheng Chen and Yue Song and Yunmei Liu and Nicu Sebe},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=okYdj8Ysru}
    }
"""
import torch as th

from .sym_functional import sym_logm, sym_expm
from .cor_functions import SPDScalingFinder, HolDplusFinder, FDplus, FDstar
from .BaseNN import SPDGroup, PullbackEuclideanMetric


# ==============================
# Utilities: vectorization helpers (isometric)
# ==============================

def _svec_sym(X: th.Tensor) -> th.Tensor:
    """
    Isometric vectorization for Sym(k): diag kept, off-diagonals * sqrt(2).
    Args:
        X: [..., k, k], symmetric
    Returns:
        z: [..., k(k+1)/2]
    """
    k = X.shape[-1]
    diag = th.diagonal(X, dim1=-2, dim2=-1)
    iu = th.triu_indices(k, k, offset=1, device=X.device)
    off_up = X[..., iu[0], iu[1]] * (2.0 ** 0.5)
    return th.cat([diag, off_up], dim=-1)


def _svec_hol(S: th.Tensor) -> th.Tensor:
    """
    Isometric vectorization for Hol(n) (symmetric with zero diag):
    take strictly upper-triangular entries and multiply by sqrt(2).
    Args:
        S: [..., n, n], symmetric, diag=0
    Returns:
        z: [..., n(n-1)/2]
    """
    n = S.shape[-1]
    iu = th.triu_indices(n, n, offset=1, device=S.device)
    off_up = S[..., iu[0], iu[1]] * (2.0 ** 0.5)
    return off_up


def _vech_strict_lower(L: th.Tensor) -> th.Tensor:
    """
    Isometric vectorization for strict-lower-triangular matrices (no sqrt(2)):
    just stack the strict lower triangle row-wise.
    Args:
        L: [..., n, n], strictly lower-triangular part meaningful
    Returns:
        z: [..., n(n-1)/2]
    """
    n = L.shape[-1]
    il = th.tril_indices(n, n, offset=-1, device=L.device)
    return L[..., il[0], il[1]]


def _helmert(n: int, device=None, dtype=None) -> th.Tensor:
    """
    n x (n-1) Helmert matrix Q: Q^T Q = I, Q^T 1 = 0.
    This gives an orthogonal isomorphism Row_0(n) -> Sym(n-1) via X -> Q^T X Q.
    """
    Q = th.zeros(n, n - 1, device=device, dtype=dtype)
    # Standard Helmert construction
    for j in range(n - 1):
        Q[:j + 1, j] = 1.0 / ((j + 1) * (j + 2)) ** 0.5
        Q[j + 1, j] = -(j + 1) / ((j + 1) * (j + 2)) ** 0.5
    return Q  # columns orthonormal, orthogonal to 1-vector


def cov2corr(cov_matrices):
        # Extract the diagonal elements (variances) from the covariance matrix
        diag_elements = th.diagonal(cov_matrices, dim1=-2, dim2=-1)
        # Compute the standard deviations (sqrt of the diagonal elements)
        std_devs = th.sqrt(diag_elements)
        # Outer product of standard deviations to form the normalization matrix
        normalization_matrix = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
        # Avoid division by zero in case of any zero variances (though variances are typically positive)
        normalization_matrix = th.where(normalization_matrix == 0, th.ones_like(normalization_matrix),
                                        normalization_matrix)
        # Compute the correlation matrix by dividing element-wise
        correlation_matrices = cov_matrices / normalization_matrix
        return correlation_matrices


cor_metrics = {'ECM', 'LECM', 'OLM', 'LSM'}


class Correlation(SPDGroup):
    """Computation for Correlation data with [...,n,n]
        Our exp indicates that is_detach=false is better
    """

    def __init__(self, n, is_detach=False):
        super().__init__(is_detach=is_detach)
        self.n = n
        self.dim = int(n * (n - 1) / 2)
        self.register_buffer('I', th.eye(n))

    def _check_point_on_manifold(self, matrix, tol=1e-6):
        """
        Check if a batch of matrices are valid correlation matrices and provide detailed feedback.

        Parameters:
        - matrix: Input tensor of shape [..., n, n]
        - tol: Tolerance for floating point comparison

        Returns:
        - True if all matrices in the batch are valid correlation matrices, False otherwise
        """

        # Ensure matrix is at least 2D and square
        if matrix.shape[-1] != matrix.shape[-2]:
            print("Failed: Matrices must be square.")
            return False

        # Get matrix size n
        n = matrix.shape[-1]

        # 1. Check symmetry in batch (matrix should be equal to its transpose)
        if not th.allclose(matrix, matrix.transpose(-2, -1), atol=tol):
            print("Failed: Batch contains non-symmetric matrices.")
            return False

        # 2. Check positive semi-definiteness by checking if eigenvalues are non-negative
        eigenvalues = th.linalg.eigvalsh(matrix)  # Calculate eigenvalues for symmetric matrices
        if not th.all(eigenvalues >= -tol):
            print("Failed: Batch contains non-SPD (non-positive semi-definite) matrices.")
            return False

        # 3. Perform Cholesky decomposition to ensure positive definiteness
        try:
            L = th.linalg.cholesky(matrix)
        except RuntimeError:
            print("Failed: Batch contains matrices that are not positive definite (Cholesky decomposition failed).")
            return False

        # 4. Check that the diagonal elements of L are positive
        if not th.all(th.diagonal(L, dim1=-2, dim2=-1) > 0):
            print("Failed: Batch contains matrices with non-positive diagonal elements in Cholesky factor.")
            return False

        # 5. Check that each row of L has unit norm
        row_norms = th.sum(L ** 2, dim=-1)
        if not th.allclose(row_norms, th.ones_like(row_norms), atol=tol):
            print("Failed: Batch contains matrices whose Cholesky factor rows do not have unit norm.")
            return False

        print("Passed: All matrices are valid correlation matrices.")
        return True

    def symmetrize(self, X):
        return (X + X.transpose(-1, -2)) / 2

    def random(self, *shape, eps=1e-6):
        """ Generate random SPD matrices based on the given shape [..., n, n]."""
        assert len(shape) >= 2 and shape[-2] == shape[-1], "Shape must be [..., n, n] for square matrices"
        n = shape[-1]
        A = th.randn(shape) * 2 - 1
        spd_matrices = th.matmul(A, A.transpose(-2, -1)) + eps * th.eye(n, device=A.device)

        return self.covariance_to_correlation(spd_matrices)

    def covariance_to_correlation(self,cov_matrices):
        # Extract the diagonal elements (variances) from the covariance matrix
        diag_elements = th.diagonal(cov_matrices, dim1=-2, dim2=-1)
        # Compute the standard deviations (sqrt of the diagonal elements)
        std_devs = th.sqrt(diag_elements)
        # Outer product of standard deviations to form the normalization matrix
        normalization_matrix = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
        # Avoid division by zero in case of any zero variances (though variances are typically positive)
        normalization_matrix = th.where(normalization_matrix == 0, th.ones_like(normalization_matrix),
                                        normalization_matrix)
        # Compute the correlation matrix by dividing element-wise
        correlation_matrices = cov_matrices / normalization_matrix
        return correlation_matrices

    def inner_product(slef, A, B):
        # Ensure A and B are of the same shape [..., n, n]
        return th.einsum('...ij,...ij->...', A, B)


class CorFlatMetric(PullbackEuclideanMetric, Correlation):
    def __init__(self, n):
        super().__init__(n)

    def dist2Isquare(self, X):
        return th.linalg.matrix_norm(X, keepdim=True).square()

    def diff_phi_inv_I(self, V):
        """(\phi_{*,I})^{-1}: the inverse map of the differential of phi at I"""
        raise NotImplementedError

    # ----- NEW: common interface for isometric vectorization -----
    def vectorize(self, C: th.Tensor) -> th.Tensor:
        """
        Map C in Cor^+(n) to an isometric vector z in R^{d} with d=n(n-1)/2.
        Each subclass implements its own isometric coordinates.
        """
        raise NotImplementedError


class CorEuclideanCholeskyMetric(CorFlatMetric):
    def __init__(self, n):
        super().__init__(n)

    def deformation(self, C):
        """ECM: \tril \circ \Theta(C) = \lfloor D(L)^{-1} L \rfloor, with L=Chol(C), Cor^+(n) \rightarrow LT^{0}"""
    
        L = th.linalg.cholesky(C)
        diag_elements = th.diagonal(L, dim1=-2, dim2=-1)  # 提取对角元素
        # 将每行除以对应的对角元素
        L_normalized = L.div(diag_elements.unsqueeze(-1))
        # 提取严格下三角部分，不包括对角线
        lower_triangular = L_normalized.tril(-1)
        return lower_triangular
      

    def inv_deformation(self, V):
        """ECM: \Theta^{-1} \circ \tril^{-1}(L) = Cor((L + I)(L + I)^\top), LT^{0} \rightarrow Cor^+(n)"""
        L = V + self.I
        Sigma = L.matmul(L.transpose(-1, -2))
        return self.covariance_to_correlation(Sigma)

    def diff_phi_inv_I(self, V):
        """V \in LT^0(n) and identical for ECM and LECM at I"""
        return V + V.transpose(-1, -2)

    # ----- NEW: isometric vectorization for ECM -----
    def vectorize(self, C: th.Tensor) -> th.Tensor:
        """
        ECM isometric vectorization:
            z = vech_strict_lower( tril_{-1}( D(L)^{-1} L ) ),  L=chol(C)
        No sqrt(2) needed because strict-lower entries are counted once in Frobenius norm.
        """
        V = self.deformation(C)  # [..., n, n], strict lower part meaningful
        return _vech_strict_lower(V)


class CorLogEuclideanCholeskyMetric(CorEuclideanCholeskyMetric):
    def __init__(self, n):
        super().__init__(n)

    def matrix_logarithm_lt1(self, L):
        """
        Computes the matrix logarithm for matrices in LT^1 (lower triangular with unit diagonal).
        Parameters:
            L (torch.Tensor): The input matrix of shape [..., n, n] assumed to be in LT^1.
        """
        n = L.shape[-1]
        L_minus_I = L.tril(-1)  # Directly get the strictly lower part, assuming L has unit diagonal
        log_L = th.zeros_like(L, dtype=L.dtype)
        # Initial power (L - I)^1
        current_power = L_minus_I
        # Series expansion up to (n-1) terms for LT^1 matrices
        for k in range(1, n):  # k goes from 1 to n-1
            term = ((-1) ** (k - 1)) / k * current_power
            log_L = log_L + term
            current_power = current_power @ L_minus_I  # Update to the next power
        return log_L

    def matrix_exponential_lt0(self, xi):
        """
        Computes the matrix exponential for matrices in LT^0 (lower triangular with zero diagonal).
        Parameters:
            xi (torch.Tensor): The input matrix in LT^0 of shape [..., n, n].
        """
        n = xi.shape[-1]
        exp_xi = th.eye(n, device=xi.device, dtype=xi.dtype)  # Initialize with identity matrix
        term = th.eye(n, device=xi.device, dtype=xi.dtype)  # First term in the series (k=0)
        # Sum terms up to (n-1), as higher powers will be zero for LT^0 matrices
        for k in range(1, n):
            term = term @ xi / k  # Compute the next term in the series
            exp_xi = exp_xi + term  # Add the term to the series sum
        return exp_xi

    def deformation(self, C):
        """LECM: \log \circ \Theta(C), with L=Chol(C), Cor^+(n) \rightarrow LT^{0}"""
        L = th.linalg.cholesky(C)
        diag_elements = th.diagonal(L, dim1=-2, dim2=-1)  # Extract diagonal elements
        # Divide each row of L by its corresponding diagonal element
        L_normalized = L.div(diag_elements.unsqueeze(-1))  # Normalize the rows by diagonal
        L_normalized_log = self.matrix_logarithm_lt1(L_normalized)  # matrix logarithm

        return L_normalized_log.tril(-1)  # [..., n, n], lower triangular part

    def inv_deformation(self, V):
        """LECM: \Theta^{-1} \circ \exp(V), LT^{0} \rightarrow Cor^+(n)"""
        L = self.matrix_exponential_lt0(V)
        Sigma = L.matmul(L.transpose(-1, -2))
        return self.covariance_to_correlation(Sigma)

    # ----- NEW: isometric vectorization for LECM (same as ECM in Cor case) -----
    def vectorize(self, C: th.Tensor) -> th.Tensor:
        """
        LECM (for correlation matrices) shares the same free parameters as ECM:
            z = vech_strict_lower( tril_{-1}( log( D(L)^{-1} L ) ) )
        """
        V = self.deformation(C)
        return _vech_strict_lower(V)


class CorOffLogMetric(CorFlatMetric):
    def __init__(self, n, alpha=1.0, beta=0.0, gamma=0.0, max_iter=100):
        super().__init__(n)
        self.max_iter = max_iter
        self.HolDplusFinder = HolDplusFinder(max_iter=self.max_iter)

        cond1 = n >= 4 and alpha > 0 and 2 * alpha + (n - 2) * beta > 0 and alpha + (n - 1) * (beta + n * gamma) > 0
        cond2 = n == 3 and alpha == 0 and beta > 0 and beta + 3 * gamma > 0
        cond3 = n == 2 and alpha == 0 and beta == 0 and gamma > 0
        assert cond1 or cond2 or cond3, \
            f"Invalid parameters for n>=4, α>0,  2α + (n-2)β > 0, and α + (n-1)(β + nγ) > 0 must hold. " \
            f"\n n=3, α=0, β>0, and β+3γ>0." \
            f"\n n=2, α=β=0, and γ>0"

        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('beta', th.tensor(beta))
        self.register_buffer('gamma', th.tensor(gamma))

    def deformation(self, C):
        """OLM: \off \circ \log(C), Cor^+(n) \rightarrow Hol(n)"""
        sym = sym_logm.apply(C)
        return self.symmetrize(sym.tril(-1) + sym.triu(1))

    def inv_deformation(self, V):
        """OLM: \Exp^{o}: Hol(n) \rightarrow Cor^+(n)"""
        # return sym_expm.apply(FDplus.apply(V,self.HolDplusFinder))
        return sym_expm.apply(V + self.HolDplusFinder(V))

    def dist2Isquare(self, X):
        """ Permutation-invariant distance for Hol(n) matrices."""
        # Ensure input is in Hol(n) (symmetric, zero diagonal)
        hol = X.tril(-1) + X.triu(1)
        X2 = hol @ hol
        tr_X2 = X2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdims=True).unsqueeze(-1)  # Trace term
        sum_X2 = X2.sum(dim=(-2, -1), keepdim=True)  # Sum of all squared elements
        sum_X_sq = hol.sum(dim=(-2, -1), keepdim=True) ** 2  # Square of sum of elements

        # Compute the quadratic form
        return self.alpha * tr_X2 + self.beta * sum_X2 + self.gamma * sum_X_sq

    def diff_phi_inv_I(self, V):
        """V \in Hol(n)"""
        return V

    def project_busemann(self, X: th.Tensor, A: th.Tensor) -> th.Tensor:
        """
        使用自定义的Busemann函数高效地将一批矩阵X投影到一批方向A上。

        投影函数 proj(x, v) = tr(v @ x) + Sum(v @ x + x @ v) + sum(x) * sum(v)
        其中 Sum() 表示对矩阵所有元素求和, sum() 也表示相同意思。

        参数:
        X (torch.Tensor): 输入的矩阵批次，形状为 (n, d, d)。
        A (torch.Tensor): 投影方向矩阵批次，形状为 (num_projections, d, d)。

        返回:
        torch.Tensor: 投影结果坐标矩阵，形状为 (n, num_projections)。
        """
        # 验证输入维度
        assert X.ndim == 3 and A.ndim == 3, "Inputs X and A must be 3D tensors."
        assert X.shape[-1] == A.shape[-1], "Dimension d of X and A must match."

        # -------- Term 1: tr(v @ x) --------
        term1 = th.einsum('jlk,ikl->ij', A, X)

        # -------- Term 2 (Optimized): Sum(v @ x + x @ v) --------
        col_sums_A = A.sum(dim=-2)  # v的列和, 形状: (num_projections, d)
        row_sums_X = X.sum(dim=-1)  # x的行和, 形状: (n, d)
        sum_vx = th.einsum('jd,id->ij', col_sums_A, row_sums_X)

        col_sums_X = X.sum(dim=-2)  # x的列和, 形状: (n, d)
        row_sums_A = A.sum(dim=-1)  # v的行和, 形状: (num_projections, d)
        sum_xv = th.einsum('id,jd->ij', col_sums_X, row_sums_A)

        term2 = sum_vx + sum_xv

        # -------- Term 3: sum(x) * sum(v) --------
        sum_of_elements_X = th.sum(X, dim=(-2, -1))  # 形状: (n,)
        sum_of_elements_A = th.sum(A, dim=(-2, -1))  # 形状: (num_projections,)
        term3 = th.einsum('i,j->ij', sum_of_elements_X, sum_of_elements_A)

        projection_coordinates = self.alpha * term1 + self.beta * term2 + self.gamma * term3

        return projection_coordinates

    # ----- NEW: isometric vectorization for OLM -----
    def vectorize(self, C: th.Tensor) -> th.Tensor:
        """
        OLM isometric vectorization:
            S = Off(log C) ∈ Hol(n)
            z = svec_hol(S)  (take upper off-diagonal * sqrt(2))
        """
        S = self.deformation(C)  # [..., n, n], symmetric with zero diagonal
        return _svec_hol(S)


class CorLogScaledMetric(CorFlatMetric):
    def __init__(self, n, alpha=1.0, delta=0.0, zeta=0.0, max_iter=100):
        super().__init__(n)
        self.max_iter = max_iter
        self.SPDScalingFinder = SPDScalingFinder(max_iter=self.max_iter)

        cond1 = n >= 4 and alpha > 0 and n * alpha + (n - 2) * delta > 0 and n * alpha + (n - 1) * (delta + n * zeta) > 0
        cond2 = n == 3 and alpha == 0 and delta > 0 and delta + 3 * zeta > 0
        cond3 = n == 2 and alpha == 0 and delta == 0 and zeta > 0
        assert cond1 or cond2 or cond3, \
            f"Invalid parameters: n>=4, α > 0, nα + (n-2)δ > 0, and nα + (n-1)(δ + nζ) > 0 must hold." \
            f"\n n=3, α=0, δ>0, and δ+3ζ>0." \
            f"\n n=3 and α=δ=0, and ζ>0."

        # ✅ Register α, δ, ζ as buffers for numerical stability
        self.register_buffer('alpha', th.tensor(alpha))
        self.register_buffer('delta', th.tensor(delta))
        self.register_buffer('zeta', th.tensor(zeta))

        # ----- NEW: cache Helmert matrix as buffer -----
        Q = _helmert(n, device=self.I.device, dtype=self.I.dtype)
        self.register_buffer('Q_helmert', Q)

    def is_rzero(self, matrix, atol=1e-6):
        """
        Checks if the input matrix is in Row_0(n), which requires:
          1. The matrix is symmetric.
          2. Each row sums to zero.

        Parameters
        ----------
        matrix : th.Tensor
            The input matrix or batch of matrices of shape [..., n, n].
        atol : float, optional
            Absolute tolerance for checking row sums and symmetry.

        Returns
        -------
        bool
            True if all matrices meet the Row_0(n) conditions, False otherwise.
        """
        # Check symmetry
        is_symmetric = th.allclose(matrix, matrix.transpose(-1, -2), atol=atol)
        # Sum along the last dimension (columns) to get row sums
        row_sums = matrix.sum(dim=-1)
        # Check if all row sums are close to zero within the specified tolerance
        is_row_sum_zero = th.allclose(row_sums, th.zeros_like(row_sums), atol=atol)

        return is_symmetric and is_row_sum_zero

    def deformation(self, C):
        """LSM: \Log^\star, Cor^+(n) \rightarrow Row_0 (n).
        """
        D = th.diag_embed(self.SPDScalingFinder(C))
        R1_spd = D @ C @ D
        # #
        # R1_spd=FDstar.apply(C, self.SPDScalingFinder)
        return sym_logm.apply(R1_spd)

    def inv_deformation(self, V):
        """LSM: \Exp^{\star} = \cor \circ \exp"""
        return self.covariance_to_correlation(sym_expm.apply(V))

    def dist2Isquare(self, Y):
        """Permutation-invariant squared distance for Row₀(n) matrices."""

        # Compute tr(Y²) (sum of squared diagonal elements)
        Y2 = Y @ Y
        tr_Y2 = Y2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdims=True).unsqueeze(-1)  # Trace term

        # Compute tr(Diag(Y)²) (sum of squared diagonal elements only)
        Y_diag = Y.diagonal(dim1=-2, dim2=-1)
        tr_diag_Y2 = (Y_diag * Y_diag).sum(dim=-1, keepdims=True).unsqueeze(-1)

        # Compute tr(Y)² = (sum of all elements in Y)²
        sum_trY_sq = Y_diag.sum(dim=-1, keepdims=True).pow(2).unsqueeze(-1)

        # Compute the quadratic form
        return self.alpha * tr_Y2 + self.delta * tr_diag_Y2 + self.zeta * sum_trY_sq

    def diff_phi_inv_I(self, V):
        """V \in Row^0(n)"""
        return V.tril(-1) + V.triu(1)

    def project_busemann(self, X: th.Tensor, A: th.Tensor, ) -> th.Tensor:
        """
        使用自定义的Busemann函数将一批矩阵X投影到一批方向A上。

        投影函数 proj(x, v) = α * tr(x @ v) + δ * tr(Diag(x) @ Diag(v)) + ζ * tr(x) * tr(v)

        参数:
        X (th.Tensor): 输入的矩阵批次，形状为 (n, d, d)
        A (th.Tensor): 投影方向矩阵批次，形状为 (num_projections, d, d)
        alpha (float): 第1项的权重系数
        delta (float): 第2项的权重系数
        zeta (float): 第3项的权重系数

        返回:
        th.Tensor: 投影结果坐标矩阵，形状为 (n, num_projections)。
                  矩阵的 (i, j) 元素是 X[i] 在方向 A[j] 上的投影值。
        """
        # 1. 验证输入维度
        assert X.ndim == 3 and A.ndim == 3, "输入 X 和 A 必须是 3D 张量"
        assert X.shape[1] == X.shape[2], "X 中的矩阵必须是方阵"
        assert A.shape[1] == A.shape[2], "A 中的矩阵必须是方阵"
        assert X.shape[1] == A.shape[1], "X 和 A 中的矩阵维度必须匹配"

        # -------- Term 1: tr(x @ v) --------
        term1 = th.einsum('ikl,jlk->ij', X, A)

        # -------- Term 2: tr(Diag(x) @ Diag(v)) --------
        diag_X = th.diagonal(X, dim1=-2, dim2=-1)  # 形状: (n, d)
        diag_A = th.diagonal(A, dim1=-2, dim2=-1)  # 形状: (num_projections, d)
        term2 = diag_X @ diag_A.T

        # -------- Term 3: tr(x) * tr(v) --------
        trace_X = th.einsum('nii->n', X)
        trace_A = th.einsum('mii->m', A)
        term3 = th.outer(trace_X, trace_A)

        # -------- 将三项加权相加得到最终的投影坐标矩阵 --------
        projection_coordinates = self.alpha * term1 + self.delta * term2 + self.zeta * term3

        return projection_coordinates

    # ----- NEW: isometric vectorization for LSM -----

    ...
    def vectorize(self, C: th.Tensor) -> th.Tensor:
        """
        LSM isometric vectorization:
            Y = Log^*(C) ∈ Row_0(n)  (here implemented via D(C) scaling + log)
            X = Q^T Y Q ∈ Sym(n-1)   (Q = Helmert)
            z = svec_sym(X)          (diag + sqrt(2)*upper-off)
        """
        Y = self.deformation(C)  # [..., n, n], symmetric & row-sum zero
        Q = self.Q_helmert.to(Y)  # 直接对齐到Y的设备和数据类型
        X = Q.transpose(-1, -2) @ Y @ Q
        return _svec_sym(X)


# ==============================
# Optional: a convenience router
# ==============================

def vectorize_cor_matrix(C: th.Tensor, metric: str) -> th.Tensor:
    """
    Convenience wrapper to vectorize a correlation matrix C under a chosen metric.
    metric in {'ECM','LECM','OLM','LSM'}
    """
    n = C.shape[-1]
    metric = metric.upper()
    if metric == 'ECM':
        m = CorEuclideanCholeskyMetric(n)
        return m.vectorize(C)
    elif metric == 'LECM':
        m = CorLogEuclideanCholeskyMetric(n)
        return m.vectorize(C)
    elif metric == 'OLM':
        m = CorOffLogMetric(n)
        return m.vectorize(C)
    elif metric == 'LSM':
        m = CorLogScaledMetric(n)
        return m.vectorize(C)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {cor_metrics}.")