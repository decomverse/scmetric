import numpy as np
import numpy.linalg as la


def cov2corr(cov_mat: np.ndarray, posrprocess: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
    if posrprocess:
        corr_mat = np.clip((corr_mat + corr_mat.T) / 2, -1, 1)
        np.fill_diagonal(corr_mat, 1)

    return cov_mat, sigma2


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


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_spd(A, iterate=False):
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
