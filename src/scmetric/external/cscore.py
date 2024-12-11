"""
==========

Iteratively reweighted least squares (IRLS) procedure in CS-CORE
Adopted from: https://github.com/ChangSuBiostats/CS-CORE_python
==========
"""

from typing import TypedDict

import anndata as ad
import numpy as np
import scipy.stats as stats
from scipy.sparse import issparse

from scmetric.tl.stats import cov2corr, nearest_spd


class ExpressionStats(TypedDict):
    """A TypedDict for storing expression statistics."""

    mu: np.ndarray
    sigma2: np.ndarray
    corr_mat: np.ndarray
    pval_mat: np.ndarray
    test_stat_mat: np.ndarray


def IRLS(
    X,
    seq_depth: np.ndarray,
    max_iter: int = 10,
    delta_threshold: float = 0.01,
    compute_pvals: bool = False,
) -> ExpressionStats:
    """
    Implement the iteratively reweighted least squares algorithm of CS-CORE.

    Parameters
    ----------
    X: scipy.sparse._csr.csr_matrix or numpy.ndarray
        Raw UMI count matrix. n cells by p genes.
    seq_depth: 1-dimensional Numpy array, length n
        Sum of UMI counts across all genes for each cell.
    max_iter: integer
        Maximum number of iterations in IRLS.
    delta_threshold: float
        Convergence threshold for IRLS.
    post_process: logical
        Whether to post-process co-expression estimates to be within [-1,1].

    Returns
    -------
    est: p by p Numpy array
        Estimates of co-expression networks among p genes. Each entry saves the correlation between two genes.
    p_value: p by p Numpy array
        p values against H_0: two gene expressions are independent. Please refer to the paper for more details.
    test_stat:
        Test statistics against H_0: two gene expressions are independent. Please refer to the paper for more details.

    References
    ----------
    Su, C., Xu, Z., Shan, X. et al. Cell-type-specific co-expression inference from single cell RNA-sequencing data.
    Nat Commun 14, 4846 (2023). https://doi.org/10.1038/s41467-023-40503-7
    """
    seq_depth_sq = np.power(seq_depth, 2)
    seq_2 = np.sum(seq_depth_sq)
    seq_4 = np.sum(np.power(seq_depth_sq, 2))
    if issparse(X):
        mu = X.transpose().dot(seq_depth) / seq_2
        # The rest of the computation will use the centered X, which is no longer sparse
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        mu = np.dot(seq_depth, X) / seq_2
    else:
        raise ValueError(
            "Unsupported type for X: \n\
        Matrix X is neither a scipy csr_matrix nor a numpy ndarray. Please reformat the input X."
        )
    M = np.outer(seq_depth, mu)
    X_centered = X - M
    sigma2 = np.dot(seq_depth_sq, (np.power(X_centered, 2) - M)) / seq_4
    theta = np.power(mu, 2) / sigma2
    j = 0
    delta = np.inf

    # IRLS for estimating mu and sigma_jj
    while delta > delta_threshold and j <= max_iter:
        print(f"Iteration {j}")

        theta_previous = theta
        theta_median = np.median(theta[theta > 0])
        theta[theta < 0] = np.inf
        w = M + np.outer(seq_depth_sq, np.power(mu, 2) / theta_median)
        w[w <= 0] = 1
        mu = np.dot(seq_depth, X / w) / np.dot(seq_depth_sq, 1 / w)
        M = np.outer(seq_depth, mu)
        X_centered = X - M
        h = np.power(np.power(M, 2) / theta_median + M, 2)
        h[h <= 0] = 1
        sigma2 = np.dot(seq_depth_sq, (np.power(X_centered, 2) - M) / h) / np.dot(np.power(seq_depth_sq, 2), 1 / h)
        theta = np.power(mu, 2) / sigma2
        j = j + 1
        theta_subset = np.logical_and(theta_previous > 0, theta > 0)
        delta = np.max(np.abs(np.log(theta[theta_subset]) - np.log(theta_previous[theta_subset])))

    if j == max_iter and delta > delta_threshold:
        print("IRLS failed to converge after 10 iterations. Please check your data.")
    else:
        print(f"IRLS converged after {j} iterations (delta={delta:.2e}).")

    # Weighted least squares for estimating sigma_jj'
    theta_median = np.median(theta[theta > 0])
    theta[theta < 0] = np.inf
    w = M + np.outer(seq_depth_sq, np.power(mu, 2) / theta_median)
    w[w <= 0] = 1

    X_weighted = X_centered / w
    num = np.einsum("i,ij->ij", seq_depth_sq, X_weighted).T @ X_weighted
    seq_depth_sq_weighted = np.einsum("i,ij->ij", seq_depth_sq, 1 / w)
    deno = seq_depth_sq_weighted.T @ seq_depth_sq_weighted
    cov_mat = num / deno

    # Evaluate test statistics and p values
    if compute_pvals:
        Sigma = M + np.outer(seq_depth_sq, sigma2)
        X_weighted = X_centered / Sigma
        num = np.einsum("i,ij->ij", seq_depth_sq, X_weighted).T @ X_weighted
        seq_depth_sq_weighted = np.einsum("i,ij->ij", seq_depth_sq, 1 / Sigma)
        deno = seq_depth_sq_weighted.T @ seq_depth_sq_weighted
        test_stat_mat = num / np.sqrt(deno)
        pval_mat = 2 * (1 - stats.norm.cdf(np.abs(test_stat_mat)))
    else:
        test_stat_mat = np.zeros_like(cov_mat)
        pval_mat = np.ones_like(cov_mat)

    # Evaluate co-expression estimates
    neg_inds = sigma2 < 0
    sigma2[neg_inds] = np.nan
    np.fill_diagonal(cov_mat, sigma2)
    corr_mat = cov2corr(cov_mat)[0]

    # Clean up mean rate estimates
    mu[neg_inds] = np.nan
    mu = np.clip(mu, 0, 1)
    mu = mu / np.nansum(mu)

    return {
        "mu": mu,
        "sigma2": sigma2,
        "corr_mat": corr_mat,
        "pval_mat": pval_mat,
        "test_stat_mat": test_stat_mat,
    }


def CSCORE(
    adata: ad.AnnData,
    seq_depth=None,
    compute_pvals=True,
    enforce_positive=False,
    seq_depth_key="seq_depth",
    mean_key="mu",
    sigma2_key="sigma2",
    low_var_key="has_low_variance",
    corr_mat_key="corr_mat",
    corr_mat_pval_key="corr_mat_pval",
    corr_mat_z_key="corr_mat_z",
    layer=None,
    return_raw=False,
    copy=False,
):
    """
    Implement CS-CORE for inferring cell-type-specific co-expression networks with scanpy object.

    Parameters
    ----------
    adata: AnnData
        Single cell data object.
    seq_depth: 1-dimensional Numpy array, optional
        Sum of UMI counts across all genes for each cell. If None, it will be computed from the data.
    compute_pvals: bool, optional
        Whether to compute p-values and test statistics, by default False.
    seq_depth_key: str, optional
        Key for the sequencing depth values in adata.obs, by default "seq_depth".
    mean_key: str, optional
        Key for the mean values in adata.var, by default "mu".
    sigma2_key: str, optional
        Key for the variance values in adata.var, by default "sigma2".
    low_var_key: str, optional
        Key for the low variance indicator in adata.var, by default "has_low_variance".
    corr_mat_key: str, optional
        Key for the correlation matrix in adata.varp, by default "corr_mat".
    corr_mat_pval_key: str, optional
        Key for the correlation matrix p-values in adata.varp, by default "corr_mat_pval".
    corr_mat_z_key: str, optional
        Key for the correlation matrix test statistics in adata.varp, by default "corr_mat_z".
    layer: str, optional
        Layer of the AnnData object to use, by default None.
    return_raw: bool, optional
        Whether to return the raw results, by default False.
    copy: bool, optional
        Whether to return a copy of the AnnData object, by default False.

    Returns
    -------
    AnnData | dict | None
        AnnData object with co-expression estimates and optionally p-values and test statistics, or raw results if return_raw is True.
    """
    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if seq_depth is None:
        seq_depth = np.array(X.sum(axis=1)).squeeze()

    res = IRLS(X, seq_depth, compute_pvals=compute_pvals)

    if enforce_positive:
        res["corr_mat"] = nearest_spd(res["corr_mat"])

    if return_raw:
        return res
    else:
        adata.var[mean_key] = res["mu"]
        adata.var[sigma2_key] = res["sigma2"]
        adata.var[low_var_key] = np.isnan(res["sigma2"])
        adata.obs[seq_depth_key] = seq_depth

        adata.varp[corr_mat_key] = res["corr_mat"]
        if compute_pvals:
            adata.varp[corr_mat_z_key] = res["test_stat_mat"]
            adata.varp[corr_mat_pval_key] = res["pval_mat"]

        return adata if copy else None


def compute_pearson_residuals(
    adata: ad.AnnData,
    seq_depth: np.ndarray | None = None,
    min_variance: float = 1e-12,
    trim_frac: float = 0.25,
    z_threshold: float = 10,
    layer: str | None = None,
    mu_key: str = "mu",
    sigma2_key: str = "sigma2",
    output_layer: str = "pearson_residuals",
    return_raw: bool = False,
    copy: bool = True,
    **kwargs,
) -> ad.AnnData | np.ndarray | None:
    """
    Compute Pearson residuals for single-cell RNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Single cell data object.
    min_variance : float, optional
        Minimum variance threshold, by default 1e-12.
    trim_frac : float, optional
        Fraction of data to trim when computing the trimmed mean, by default 0.25.
    z_threshold : float, optional
        Threshold for clipping Z-scores, by default 10.
    layer : str | None, optional
        Layer of the AnnData object to use, by default None.
    mu_key : str, optional
        Key for the mean values in adata.var, by default "mu".
    sigma2_key : str, optional
        Key for the variance values in adata.var, by default "sigma2".
    seq_depth_key : str | None, optional
        Key for the sequencing depth values in adata.obs, by default None.
    output_layer : str, optional
        Layer to store the Pearson residuals, by default "pearson_residuals".
    return_raw : bool, optional
        Whether to return the raw Pearson residuals, by default False.
    copy : bool, optional
        Whether to return a copy of the AnnData object, by default True.

    Returns
    -------
    AnnData | np.ndarray | None
        AnnData object with Pearson residuals stored in the specified layer, or raw Pearson residuals if return_raw is True.
    """
    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if seq_depth is None:
        seq_depth = np.array(X.sum(axis=1)).squeeze()

    mu = np.array(adata.var[mu_key].values)
    sigma2 = np.array(adata.var[sigma2_key].values)
    dispersion = np.power(mu, 2) / sigma2
    dispersion[sigma2 <= 0] = np.nan

    print("Computing Pearson's residuals")
    seq_depth_sq = np.power(seq_depth, 2)
    d = dispersion[min_variance < sigma2]
    dispersion_trimmed_mean = stats.trimboth(d, trim_frac).mean()

    # compute expected mean of counts
    M = np.outer(seq_depth, mu)
    # compute variance of counts
    V = M + np.outer(seq_depth_sq, np.power(mu, 2) / dispersion_trimmed_mean)
    V[V <= 0] = 1

    # normalize counts
    Z = (X - M) / np.sqrt(V)
    Z = Z.clip(-z_threshold, z_threshold)

    Z = np.array(Z)

    if return_raw:
        return Z
    else:
        adata.layers[output_layer] = Z
        return adata if copy else None
