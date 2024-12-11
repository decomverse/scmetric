import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import t as tdist


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_spd(A, iterate=True):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A = np.array(A)
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if iterate:
        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

    return A3


def cov2corr(cov_mat: np.ndarray, postprocess: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov_mat : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    sigma2 = np.diag(cov_mat).copy()

    filter_mask = (sigma2 <= 0) | np.isnan(sigma2)
    sigma2[filter_mask] = 1
    sigma = np.sqrt(sigma2)
    corr_mat = cov_mat / np.outer(sigma, sigma)
    corr_mat[filter_mask, :] = 0
    corr_mat[:, filter_mask] = 0

    # Symmetrize and clip values to [-1, 1]
    if postprocess:
        corr_mat = np.clip((corr_mat + corr_mat.T) / 2, -1, 1)
        np.fill_diagonal(corr_mat, 1)

    return corr_mat, sigma2


def cor2cov(corr_mat: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to a covariance matrix.

    Parameters
    ----------
    corr_mat : np.ndarray
        Correlation matrix.
    sigma2 : np.ndarray
        Vector of variances.

    Returns
    -------
    np.ndarray
        Covariance matrix.
    """
    filter_mask = (sigma2 <= 0) | np.isnan(sigma2)
    sigma2[filter_mask] = 1
    sigma = np.sqrt(sigma2)
    cov_mat = corr_mat * np.outer(sigma, sigma)
    cov_mat[filter_mask, :] = 0
    cov_mat[:, filter_mask] = 0

    return cov_mat


def cov2prec_chol(cov_mat, return_chol=False):
    """
    convert covariance matrix to precision matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    prec : array_like, 2d
        precision matrix
    L : array_like, 2d (lower triangular matrix)
        cholesky factor of the covariance
    L_inv : array_like, 2d
        inverse of the cholesky factor matrix of the covariance, L (can be used for whitening)
    """
    cov_mat = nearest_spd(np.asanyarray(cov_mat))

    L = np.linalg.cholesky(cov_mat)
    L_inv = np.linalg.inv(L)

    prec = L_inv.T @ L_inv

    if return_chol:
        return prec, L, L_inv
    else:
        return prec


def corr_eig2prec(evals, evecs, sigma, evals_threshold=1e-8):
    """
    Convert eigenvalues and eigenvectors of a correlation matrix to a precision matrix.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues of the correlation matrix.
    evecs : np.ndarray
        Eigenvectors of the correlation matrix.
    sigma : np.ndarray
        Standard deviations.
    evals_threshold : float, optional
        Threshold for eigenvalues to be considered significant (default is 1e-8).

    Returns
    -------
    np.ndarray
        Precision matrix.
    """
    mask = evals_threshold < evals

    whitening_mat = evecs[:, mask] / np.sqrt(evals[mask])
    corr_mat_inv = whitening_mat @ whitening_mat.T

    std_inv = 1 / sigma
    std_inv[sigma == 0] = 1
    prec = corr_mat_inv * np.outer(std_inv, std_inv)

    return prec


def corr2prec_eig(corr_mat, sigma, evals_threshold=1e-8, return_eigs=False):
    """
    Convert a correlation matrix to a precision matrix using eigenvalue decomposition.

    Parameters
    ----------
    corr_mat : np.ndarray
        Correlation matrix.
    sigma : np.ndarray
        Standard deviations.
    evals_threshold : float, optional
        Threshold for eigenvalues to be considered significant (default is 1e-8).
    return_eigs : bool, optional
        If True, also return eigenvalues and eigenvectors (default is False).

    Returns
    -------
    np.ndarray
        Precision matrix.
    tuple of np.ndarray, optional
        Eigenvalues and eigenvectors, if return_eigs is True.
    """
    evals, evecs = np.linalg.eigh(corr_mat)
    prec = corr_eig2prec(evals, evecs, sigma, evals_threshold)

    if return_eigs:
        return prec, evals, evecs
    else:
        return prec


def cov_eig2prec(evals, evecs, evals_threshold=1e-8):
    """
    Convert eigenvalues and eigenvectors of a covariance matrix to a precision matrix.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues of the covariance matrix.
    evecs : np.ndarray
        Eigenvectors of the covariance matrix.
    evals_threshold : float, optional
        Threshold for eigenvalues to be considered significant (default is 1e-8).

    Returns
    -------
    np.ndarray
        Precision matrix.
    """
    mask = evals_threshold < evals

    whitening_mat = evecs[:, mask] / np.sqrt(evals[mask])
    prec = whitening_mat @ whitening_mat.T

    return prec


def cov2prec_eig(cov_mat, evals_threshold=1e-8, return_eigs=False):
    """
    Convert a covariance matrix to a precision matrix using eigenvalue decomposition.

    Parameters
    ----------
    cov_mat : np.ndarray
        Covariance matrix.
    evals_threshold : float, optional
        Threshold for eigenvalues to be considered significant (default is 1e-8).
    return_eigs : bool, optional
        If True, also return eigenvalues and eigenvectors (default is False).

    Returns
    -------
    np.ndarray
        Precision matrix.
    tuple of np.ndarray, optional
        Eigenvalues and eigenvectors, if return_eigs is True.
    """
    evals, evecs = np.linalg.eigh(cov_mat)
    prec = cov_eig2prec(evals, evecs, evals_threshold)

    if return_eigs:
        return prec, evals, evecs
    else:
        return prec


def prec2pcorr(prec):
    """
    convert precision matrix to partial correlation matrix

    Parameters
    ----------
    prec : array_like, 2d
        precision matrix

    Returns
    -------
    pcorr : ndarray (subclass)
        partial correlation matrix
    """
    prec = np.asanyarray(prec)
    inv_std_ = np.sqrt(np.diag(prec))
    pcorr = prec / np.outer(inv_std_, inv_std_)

    return pcorr


def welch_tstat(X, Y):
    """
    Compute Welch's t-statistic for two sets of samples.

    Parameters
    ----------
    X : np.ndarray
        First set of samples with shape (features, samples).
    Y : np.ndarray
        Second set of samples with shape (features, samples).

    Returns
    -------
    pd.DataFrame
        DataFrame containing t-statistics, p-values, log fold ratios, means, variances, and mean differences.
    """
    nX = X.shape[1]
    nY = Y.shape[1]

    X_mean = np.mean(X, axis=1)
    Y_mean = np.mean(Y, axis=1)
    delta = X_mean - Y_mean

    X_var = np.var(X, axis=1)
    Y_var = np.var(Y, axis=1)

    X_ste_sq = X_var / nX
    Y_ste_sq = Y_var / nY

    denom = np.sqrt(X_ste_sq + Y_ste_sq)
    tstat = delta / denom

    df = (X_ste_sq + Y_ste_sq) ** 2 / ((X_ste_sq**2 / (nX - 1)) + (Y_ste_sq**2 / (nY - 1)))
    pvals = tdist.sf(abs(tstat), df)
    LFR = np.log2(X_mean / Y_mean)

    res_tbl = pd.DataFrame(
        {
            "t": tstat,
            "pval": pvals,
            "LFR": LFR,
            "X_mean": X_mean,
            "Y_mean": Y_mean,
            "delta_mean": delta,
            "X_var": X_var,
            "Y_var": Y_var,
        }
    )

    return res_tbl
