"""
==========

Contrastive PCA (cPCA) is a linear dimensionality reduction technique that uses eigenvalue decomposition to identify directions that have increased variance in the primary (foreground) dataset relative to a secondary (background) dataset. Then, those directions are used to project the data to a lower dimensional space.
==========
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance, empirical_covariance

from scmetric.external.cscore import IRLS
from scmetric.tl.stats import cor2cov, nearest_spd


class CPCA_cov(BaseEstimator, TransformerMixin):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    cPCA: https://www.nature.com/articles/s41467-018-04608-8
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        self.fitted = False

    def _trace_ratio(self, eps=1e-3, min_eig=1e-6):
        """
        Compute the trace ratio for the contrastive PCA.

        Parameters
        ----------
        eps : float, optional
            Small value to ensure numerical stability.
        min_eig : float, optional
            Minimum eigenvalue threshold.

        Returns
        -------
        alpha : float
            Regularization parameter.
        target_var : float
            Variance of the target dataset.
        background_var : float
            Variance of the background dataset.
        """
        utils.validation.check_is_fitted(self)

        # Contrastive axes
        pos_eigs = max(self.n_components, (self.w_ >= min_eig).sum())
        V = self.v_[:, 0:pos_eigs]

        target_var = np.linalg.multi_dot([V.T, self.target_cov, V]).trace()

        # this is the way to add eps in cNRL by Fujiwara et al., 2020.
        # https://arxiv.org/abs/2005.12419
        # tr_bg = (self.components.T @ self.B_bg @ self.components +
        #          np.identity(self.components.shape[1]) * eps).trace()

        # here is the new way to add eps to make sure eps is the ratio of tr_fg

        background_var = np.linalg.multi_dot([V.T, self.background_cov, V]).trace()

        delta = target_var * eps
        alpha = target_var / (background_var + delta)

        return alpha, target_var, background_var

    def fit(
        self,
        target_cov,
        background_cov,
        alpha=None,
        eps=1e-3,
        convergence_ratio=1e-2,
        max_iter=20,
        cov_correction_method=nearest_spd,
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
        correct_cov : bool, optional
            Whether to correct the covariance matrices to be positive definite.

        # Automatic contrast parameter estimation method is adopted from:
        #   https://github.com/takanori-fujiwara/cmca/blob/c459393f517b44f616c000f8790df71161869837/cmca.py#L242
        """
        if background_cov.shape[1] != target_cov.shape[1]:
            raise ValueError("Covariance matrices should have the same dim.")

        self.cov_correction_method = cov_correction_method

        self.target_cov = self.cov_correction_method(target_cov)
        self.background_cov = self.cov_correction_method(background_cov)

        if alpha is None:
            self._fit_with_best_alpha(eps=eps, convergence_ratio=convergence_ratio, max_iter=max_iter)
        else:
            self._fit_with_manual_alpha(alpha=alpha)

    def _fit_with_manual_alpha(self, alpha):
        """
        Fit the model with a manually specified alpha.

        Parameters
        ----------
        alpha : float
            Regularization parameter.
        """
        # Recompute contrastive covariance matrix
        self.alpha = alpha
        self.contrastive_cov = self.target_cov - alpha * self.background_cov

        self.w_, self.v_ = np.linalg.eigh(self.contrastive_cov)

        # schur_form, self.v_ = la.schur(self.contrastive_cov)
        # self.w_ = la.eigvals(schur_form)

        perm = np.argsort(-self.w_)
        self.w_ = self.w_[perm]
        self.v_ = self.v_[:, perm]

        self.components = self.v_[:, : self.n_components]
        top_w = self.w_[: self.n_components]

        self.loadings = self.components @ np.diag(np.sqrt(np.abs(top_w)))

        return self

    def _fit_with_best_alpha(self, eps=1e-3, convergence_ratio=1e-2, max_iter=10):
        """
        Fit the model by finding the best alpha automatically.

        Parameters
        ----------
        eps : float, optional
            Small value to ensure numerical stability.
        convergence_ratio : float, optional
            Convergence threshold for the iterative process to find the best alpha.
        max_iter : int, optional
            Maximum number of iterations for the iterative process to find the best alpha.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_trace = {}

        self.fit_trace = []
        self._fit_with_manual_alpha(0)
        alpha, target_var, background_var = self._trace_ratio(eps)

        for iter in range(max_iter):
            print(
                f"iter {iter+1}: alpha={alpha:0.2e}, target_var={target_var:0.2e}, background_var={background_var:0.2e}, contrastive_var={target_var - alpha * background_var:0.2e}"
            )

            self._fit_with_manual_alpha(alpha)
            alpha, target_var, background_var = self._trace_ratio(eps)

            log = {
                "alpha": self.alpha,
                "target_var": target_var,
                "background_var": background_var,
                "contrastive_var": target_var - self.alpha * background_var,
                "components": self.components.copy(),
                "loadings": self.loadings.copy(),
            }
            self.fit_trace.append(log)

            rel_delta_alpha = (alpha - self.alpha) / (self.alpha + 1e-15)
            if rel_delta_alpha <= convergence_ratio:
                break

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

        return X @ self.components

    def get_projection_trace(self, X):
        """
        Projects the input data `X` onto the components stored in `fit_trace` and returns an AnnData object containing the projections.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be projected.

        Returns
        -------
        adata : AnnData
            An AnnData object containing the projections in its `.obsm` attribute. The keys in `.obsm` correspond to the `alpha` values from `fit_trace`, and the values are the projected data matrices.

        Notes
        -----
        This method assumes that the object has been fitted and `fit_trace` is available.
        """
        utils.validation.check_is_fitted(self)

        adata = ad.AnnData(X)

        obsm = {}
        for i in range(len(self.fit_trace)):
            key = f'alpha={self.fit_trace[i]["alpha"]:.2e}'
            components_ = self.fit_trace[i]["components"]
            projections = adata.X @ components_
            obsm[key] = projections

        adata.obsm = obsm

        return adata

    def plot_projection_trace(self, X, label, palette="tab10", size=None):
        """
        Plots the projection trace of the given data matrix.

        Parameters
        ----------
        X : array-like
            The input data to project.
        label : array-like
            Labels corresponding to the data points in X.
        palette : str, optional (default: "tab10")
            The color palette to use for plotting.
        size : float, optional
            The size of the points in the plot.

        Returns
        -------
        None
            This function does not return any value. It generates and displays plots.
        """
        adata_proj = self.get_projection_trace(X)

        adata_proj.obs["label"] = pd.Categorical(label)
        adata_proj.obs["label"] = adata_proj.obs["label"].astype("category")

        for key in adata_proj.obsm.keys():
            sc.pl.embedding(
                adata_proj,
                basis=key,
                color="label",
                title=key,
                palette=palette,
                size=size,
                frameon=False,
                show=True,
                ncols=1,
            )

    def plot_projection(self, X, label, palette="tab10", size=None):
        """
        Plots the projection of the given data matrix.

        Parameters
        ----------
        X : array-like
            The input data to project.
        label : array-like
            Labels corresponding to the data points in X.
        palette : str, optional (default: "tab10")
            The color palette to use for plotting.
        size : float, optional
            The size of the points in the plot.

        Returns
        -------
        None
            This function does not return any value. It generates and displays plots.
        """
        adata = ad.AnnData(X)
        adata.obs["label"] = label
        adata.obs["label"] = adata.obs["label"].astype("category")

        key = f"alpha={self.alpha}"
        adata.obsm[key] = X @ self.components
        sc.pl.embedding(
            adata, basis=key, color="label", title=key, palette=palette, size=size, frameon=False, show=True, ncols=1
        )


class CPCA(CPCA_cov):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    cPCA: https://www.nature.com/articles/s41467-018-04608-8
    """

    def __init__(self, n_components=2, scale=True, **kwargs):
        self.scale = scale
        self.n_components = n_components
        super().__init__(n_components=n_components, **kwargs)

    def fit(
        self,
        target,
        background,
        algorithm="empirical",
        alpha=None,
        eps=1e-3,
        convergence_ratio=1e-2,
        max_iter=20,
    ):
        """
        Fit the model with the given target and background covariance matrices.

        Parameters
        ----------
        target : array-like, shape (n_target_samples, n_features)
            Data matrix of the target (foreground) dataset.
        background : array-like, shape (n_background_samples, n_features)
            Data matrix of the background dataset.
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
        if background.shape[1] != target.shape[1]:
            raise ValueError("Data matrices should have the same dim.")

        self.target = target.copy()
        self.background = background.copy()

        if self.scale:
            self.target = (self.target - np.mean(self.target, axis=0)) / np.std(self.target, axis=0)
            self.background = (self.background - np.mean(self.background, axis=0)) / np.std(self.background, axis=0)

        # Add wrapper for covariance matrix computation using sklearn that takes the algorithm argument and runs the appropriate method, including but not limiter to empirical_covariance, shrunk, OAS, LedoitWolf
        if algorithm == "empirical":
            target_cov = empirical_covariance(self.target, assume_centered=False)
            background_cov = empirical_covariance(self.background, assume_centered=False)
        elif algorithm == "shrunk":
            target_cov = ShrunkCovariance().fit(self.target).covariance_
            background_cov = ShrunkCovariance().fit(self.background).covariance_
        elif algorithm == "oas":
            target_cov = OAS().fit(self.target).covariance_
            background_cov = OAS().fit(self.background).covariance_
        elif algorithm == "lw":
            target_cov = LedoitWolf().fit(self.target).covariance_
            background_cov = LedoitWolf().fit(self.background).covariance_
        else:
            raise ValueError("Invalid algorithm. Choose from 'empirical' or 'shrunk'.")

        super().fit(
            target_cov, background_cov, alpha=alpha, eps=eps, convergence_ratio=convergence_ratio, max_iter=max_iter
        )

    def fit_transform(
        self,
        target,
        background,
        algorithm="empirical",
        alpha=None,
        eps=1e-3,
        convergence_ratio=1e-2,
        max_iter=20,
    ):
        """
        Fit the model with the given target and background covariance matrices,
        then apply the model to target.

        Parameters
        ----------
        target : array-like, shape (n_target_samples, n_features)
            Data matrix of the target (foreground) dataset.
        background : array-like, shape (n_background_samples, n_features)
            Data matrix of the background dataset.
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
        self.fit(
            target=target,
            background=background,
            algorithm=algorithm,
            alpha=alpha,
            eps=eps,
            convergence_ratio=convergence_ratio,
            max_iter=max_iter,
        )

        return self.transform(target)


# Try with dosing data from sciplex
class scCPCA(CPCA_cov):
    """
    Contrastive PCA (cPCA)

    Linear dimensionality reduction that uses eigenvalue decomposition
    to identify directions that have increased variance in the primary (foreground)
    dataset relative to a secondary (background) dataset. Then, those directions
    are used to project the data to a lower dimensional space.
    cPCA: https://www.nature.com/articles/s41467-018-04608-8
    """

    def __init__(self, n_components=10, scale=True, **kwargs):
        self.scale = scale
        super().__init__(n_components=10, **kwargs)

    def fit(
        self,
        target,
        background,
        layer=None,
        alpha=None,
        eps=1e-3,
        convergence_ratio=1e-2,
        max_iter=20,
        cscore_max_iter: int = 10,
        cscore_delta_threshold: float = 0.01,
    ):
        """
        Fit the model with the given target and background covariance matrices.

        Parameters
        ----------
        target : AnnData, shape (n_target_samples, n_features)
            Gene expression of the target (foreground) dataset.
        background : AnnData, shape (n_background_samples, n_features)
            Gene expression of the background dataset.
        alpha : float, optional
            Regularization parameter. If None, the best alpha is found automatically.
        eps : float, optional
            Small value to ensure numerical stability.
        convergence_ratio : float, optional
            Convergence threshold for the iterative process to find the best alpha.
        max_iter : int, optional
            Maximum number of iterations for the iterative process to find the best alpha.
        cscore_max_iter : int, optional
            Maximum number of iterations for the CSCORE algorithm.
        cscore_delta_threshold : float, optional
            Convergence threshold for the CSCORE algorithm.

        # Automatic contrast parameter estimation method is adopted from:
        #   https://github.com/takanori-fujiwara/cmca/blob/c459393f517b44f616c000f8790df71161869837/cmca.py#L242
        """
        common_vars = target.var_names.intersection(background.var_names)

        self.var = target.var.loc[common_vars].copy()

        if layer is not None:
            self.target = target[:, common_vars].layers[layer].copy()
            self.background = background[:, common_vars].layers[layer].copy()
        else:
            self.target = target[:, common_vars].X.copy()
            self.background = background[:, common_vars].X.copy()

        background_cscore = IRLS(
            self.background,
            seq_depth=np.array(self.background.sum(axis=1)).squeeze(),
            max_iter=cscore_max_iter,
            delta_threshold=cscore_delta_threshold,
            compute_pvals=False,
        )
        if self.scale:
            background_cov = background_cscore["corr_mat"]
        else:
            background_cov = cor2cov(corr_mat=background_cscore["corr_mat"], sigma2=background_cscore["sigma2"])

        target_cscore = IRLS(
            self.target,
            seq_depth=np.array(self.target.sum(axis=1)).squeeze(),
            max_iter=cscore_max_iter,
            delta_threshold=cscore_delta_threshold,
            compute_pvals=False,
        )
        if self.scale:
            target_cov = target_cscore["corr_mat"]
        else:
            target_cov = cor2cov(corr_mat=target_cscore["corr_mat"], sigma2=target_cscore["sigma2"])

        super().fit(
            target_cov,
            background_cov,
            alpha=alpha,
            eps=eps,
            convergence_ratio=convergence_ratio,
            max_iter=max_iter,
        )

    def _transform(self, adata, layer=None, raw=False, copy=False, components=None):
        """
        Transform the data using the fitted model.

        Parameters
        ----------
        adata : AnnData
            The input data to be transformed.
        layer : str, optional
            The layer of the AnnData object to use for transformation.
        raw : bool, optional
            Whether to return a raw AnnData object.
        copy : bool, optional
            Whether to return a copy of the AnnData object.
        components : array-like, optional
            The components to use for transformation.

        Returns
        -------
        adata_proj : AnnData
            The transformed AnnData object.
        """
        if components is None:
            utils.validation.check_is_fitted(self)
            components = self.components

        if layer is not None:
            projections = np.array(adata.layers[layer] @ components)
        else:
            projections = np.array(adata.X @ components)

        if raw:
            adata_proj = ad.AnnData(X=None, obs=adata.obs)
        else:
            adata_proj = adata.copy() if copy else adata

        adata_proj.obsm["X_cPCA"] = projections
        return adata_proj

    def transform(self, adata, layer=None, raw=False, copy=False):
        """
        Transform the data using the fitted model.

        Parameters
        ----------
        adata : AnnData
            The input data to be transformed.
        layer : str, optional
            The layer of the AnnData object to use for transformation.
        raw : bool, optional
            Whether to return a raw AnnData object.
        copy : bool, optional
            Whether to return a copy of the AnnData object.

        Returns
        -------
        adata_proj : AnnData
            The transformed AnnData object.
        """
        return self._transform(adata, layer=layer, raw=raw, copy=copy)

    def get_projection_trace(self, adata, layer=None, raw=False, copy=False):
        """
        Get the projection trace of the data.

        Parameters
        ----------
        adata : AnnData
            The input data to be projected.
        layer : str, optional
            The layer of the AnnData object to use for projection.
        raw : bool, optional
            Whether to return a raw AnnData object.
        copy : bool, optional
            Whether to return a copy of the AnnData object.

        Returns
        -------
        adata_proj : AnnData
            The AnnData object containing the projection trace.
        """
        utils.validation.check_is_fitted(self)

        obsm = {}
        for i in range(len(self.fit_trace)):
            key = f'cpca_alpha={self.fit_trace[i]["alpha"]:.2e}'
            components_ = self.fit_trace[i]["components"]
            projection = np.array(self._transform(adata, layer=layer, raw=True, copy=False, components=components_).obsm["X_cPCA"])
            obsm[key] = projection

        if raw:
            return ad.AnnData(X=projection, obs=adata.obs, obsm=obsm)
        else:
            adata_proj = adata.copy() if copy else adata
            adata_proj.obsm = dict(adata_proj.obsm) | obsm
            return adata_proj

    def plot_projection_trace(self, adata, labels, cmap="RdYlBu_r", palette="tab10", size=None, layer=None):
        """
        Plot the projection trace of the data.

        Parameters
        ----------
        adata : AnnData
            The input data to be projected.
        labels : array-like
            Labels corresponding to the data points in adata.
        cmap : str, optional
            The color map to use for plotting.
        palette : str, optional
            The color palette to use for plotting.
        size : float, optional
            The size of the points in the plot.
        layer : str, optional
            The layer of the AnnData object to use for projection.
        """
        adata_proj = self.get_projection_trace(adata=adata, layer=layer, raw=True, copy=False)

        for key in adata_proj.obsm.keys():
            sc.pl.embedding(
                adata_proj,
                basis=key,
                color=labels,
                title=key,
                cmap=cmap,
                palette=palette,
                size=size,
                frameon=False,
                show=True,
                ncols=1,
            )

    def plot_projection(self, adata, labels, cmap="RdYlBu_r", palette="tab10", size=None, layer=None):
        """
        Plot the projection of the data.

        Parameters
        ----------
        adata : AnnData
            The input data to be projected.
        labels : array-like
            Labels corresponding to the data points in adata.
        cmap : str, optional
            The color map to use for plotting.
        palette : str, optional
            The color palette to use for plotting.
        size : float, optional
            The size of the points in the plot.
        layer : str, optional
            The layer of the AnnData object to use for projection.
        """
        adata_proj = self.transform(adata, layer=layer, raw=True, copy=False)

        sc.pl.embedding(
            adata_proj,
            basis="X_cPCA",
            color=labels,
            cmap=cmap,
            palette=palette,
            size=size,
            frameon=False,
            show=True,
            ncols=1,
        )
