"""
==========

Contrastive PCA (cPCA) is a linear dimensionality reduction technique that uses eigenvalue decomposition to identify directions that have increased variance in the primary (foreground) dataset relative to a secondary (background) dataset. Then, those directions are used to project the data to a lower dimensional space.
==========
"""

import numpy.linalg as la
from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class CPCA(BaseEstimator, TransformerMixin):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    cPCA: https://www.nature.com/articles/s41467-018-04608-8
    """

    def __init__(self, n_components=10, **kwargs):
        self.svd_ = TruncatedSVD(n_components=n_components, **kwargs)
        self.fitted = False

    def _trace_ratio(self, eps=1e-3):
        utils.validation.check_is_fitted(self)

        # Contrastive axes
        V = self.svd_.components_

        target_var = la.multi_dot([V, self.target_cov, V.T]).trace()

        # this is the way to add eps in cNRL by Fujiwara et al., 2020.
        # https://arxiv.org/abs/2005.12419
        # tr_bg = (self.components.T @ self.B_bg @ self.components +
        #          np.identity(self.components.shape[1]) * eps).trace()

        # here is the new way to add eps to make sure eps is the ratio of tr_fg

        background_var = la.multi_dot([V, self.background_cov, V.T]).trace() + (target_var * eps)

        alpha = target_var / background_var

        return alpha, target_var, background_var

    def fit(
        self,
        target_cov,
        background_cov,
        alpha=None,
        eps=1e-3,
        convergence_ratio=1e-2,
        max_iter=10,
    ):
        """
        Fit the model with the given target and background covariance matrices.

        Parameters
        ----------
        target_cov : array-like, shape (n_features, n_features)
            Covariance matrix of the target (foreground) dataset.
        background_cov : array-like, shape (n_features, n_features)
            Covariance matrix of the background dataset.
        alpha : float, optional
            Regularization parameter. If None, the best alpha is found automatically.
        eps : float, optional
            Small value to ensure numerical stability.
        convergence_ratio : float, optional
            Convergence threshold for the iterative process to find the best alpha.
        max_iter : int, optional
            Maximum number of iterations for the iterative process to find the best alpha.

        # Automatic contrast parameter estimation method is adopted from:
        #   https://github.com/takanori-fujiwara/cmca/blob/c459393f517b44f616c000f8790df71161869837/cmca.py#L242
        """
        if background_cov.shape[1] != target_cov.shape[1]:
            raise ValueError("Covariance matrices should have the same dim.")

        self.target_cov = target_cov
        self.background_cov = background_cov

        if alpha is None:
            self._fit_with_best_alpha(eps, convergence_ratio, max_iter)
        else:
            self._fit_with_manual_alpha(alpha)

    def _fit_with_manual_alpha(self, alpha):
        # Recompute contrastive covariance matrix
        self.alpha = alpha
        self.contrastive_cov = self.target_cov - alpha * self.background_cov

        self.svd_.fit(self.contrastive_cov)

        return self

    def _fit_with_best_alpha(self, eps=1e-3, convergence_ratio=1e-2, max_iter=10):
        self.fit_trace = {}

        alpha = 0
        self.fit_trace = []
        for iter in range(max_iter):
            self._fit_with_manual_alpha(alpha)

            new_alpha, target_var, background_var = self._trace_ratio(eps)

            log = {
                "alpha": self.alpha,
                "target_var": target_var,
                "background_var": background_var,
                "contrastive_var": target_var - self.alpha * background_var,
                "components": self.svd_.components_.copy(),
                "explained_variance_ratio": self.svd_.explained_variance_ratio_.copy(),
                "explained_variance": self.svd_.explained_variance_.copy(),
                "singular_values": self.svd_.singular_values_.copy(),
            }
            self.fit_trace.append(log)

            rel_delta_alpha = (new_alpha - alpha) / (alpha + 1e-15)
            print(
                f"{iter}: alpha={alpha:.2e}, target_var={target_var:.2e}, background_var={background_var:.2e}, rel_delta_alpha={rel_delta_alpha:.2e}"
            )
            if rel_delta_alpha <= convergence_ratio:
                break

            alpha = new_alpha

        self.alpha = alpha
        return self

    def transform(self, X):
        """
        Project the data onto the contrastive principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        utils.validation.check_is_fitted(self)

        return self.svd_.transform(X)
