from collections.abc import Hashable, Sequence
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix, issparse


def match(a: Sequence[Hashable], b: Sequence[Hashable]) -> pd.Series:
    """
    Matches elements of sequence `a` to their corresponding indices in sequence `b`.

    Parameters
    ----------
    a (Sequence[Hashable]): Sequence of hashable elements to match.
    b (Sequence[Hashable]): Sequence of hashable elements to match against.

    Returns
    -------
    pd.Series: Series containing the indices of elements in `a` as they appear in `b`.
    """
    b_dict = {x: i for i, x in enumerate(b)}

    a_matched = pd.Series([b_dict.get(x, np.nan) for x in a], index=pd.Index(a), name="matched_idx")

    return a_matched


def compute_pseudobulk(
    adata: ad.AnnData,
    layer: str = "logcounts",
    grouping_var_key: str = "samples",
    normalization_method: Literal["cell_counts_norm"] | Literal["lib_size_norm"] = "cell_counts_norm",
    target_lib_size: float = 1e4,
) -> ad.AnnData:
    """
    Compute pseudobulk samples from single-cell RNA-seq data.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    layer : str, optional
        The layer of `adata` to use for aggregation. If None, the `.X` attribute is used. Default is "logcounts".
    grouping_var_key : str, optional
        The key in `adata.obs` to group cells by. Default is "samples".
    normalization_method : {"cell_counts_norm", "lib_size_norm"}, optional
        The method to normalize pseudobulk samples. Default is "cell_counts_norm".
    target_lib_size : float, optional
        The target library size for normalization when `normalization_method` is "lib_size_norm". Default is 1e4.

    Returns
    -------
    ad.AnnData
        Annotated data matrix containing pseudobulk samples.
    """
    obs_df = adata.obs.copy()
    obs_df.loc[:, grouping_var_key] = obs_df.loc[:, grouping_var_key].astype(str)

    grouped = obs_df.groupby(grouping_var_key)
    ID_pb = list(grouped.groups.keys())
    ID_cells = list(obs_df[grouping_var_key])

    # Extract pseudbulk sample metadata
    ID_pb_matched = match(ID_pb, ID_cells)
    row_idx = ID_pb_matched["matched_idx"].dropna().astype(int)
    PB_obs = obs_df.iloc[row_idx, :].set_index(ID_pb_matched.index[row_idx])

    # Remove covariates that are either constant or have the same number of unique values as the number of pseudobulk samples
    uv_count = np.array(obs_df.nunique())
    col_idx = (uv_count <= len(grouped)) & (len(grouped) > 2)
    PB_obs = PB_obs.loc[:, col_idx]

    # Add cell counts to pseudobulk samples
    cell_counts = grouped.size().to_numpy()
    cell_counts = cell_counts[cell_counts != 0]
    PB_obs["cell_counts"] = cell_counts

    # Compute pseudobulk samples
    if layer is not None:
        print(f"aggregating on layer {layer}")
        X = adata.layers[layer]
    else:
        print("aggregating on .X attribute since no layer specified")
        X = adata.X

    # Create pseudobulk samples
    class_map = np.zeros(adata.shape[0], dtype=int)
    for i, idx in enumerate(grouped.indices.values()):
        class_map[idx] = i
    design_mat = coo_matrix(
        (np.ones(class_map.shape[0]), (class_map, range(class_map.shape[0]))),
        shape=(len(grouped.indices), class_map.shape[0]),
    )

    if not issparse(X):
        X_PB = np.array(design_mat @ X)
    else:
        X_coo = coo_matrix(X)

        ii = class_map[X_coo.row]
        jj = X_coo.col
        vv = X_coo.data
        X_PB_coo = coo_matrix((vv, (ii, jj)), shape=(len(grouped), adata.shape[1]))
        X_PB_coo.sum_duplicates()

        X_PB = np.array(X_PB_coo.todense())

    # Combine into an ad.AnnData object
    PB_adata = ad.AnnData(X=X_PB, var=adata.var)
    PB_adata.obs = PB_obs

    # Scale each pseudobulk sample
    if (normalization_method == "lib_size_norm") & (
        len(set(np.unique(X_PB)).difference(set(np.arange(X_PB.max() + 1)))) > 0
    ):
        print(
            "`lib_size_norm` was passed as `normalization_method` parameter, but input data is not counts. Reverting back to `cell_counts_norm`"
        )
        normalization_method = "cell_counts_norm"
    PB_adata.uns["normalization_method"] = normalization_method

    PB_adata.layers[layer + "sum"] = np.array(PB_adata.X)
    if normalization_method == "cell_counts_norm":
        X_PB = X_PB / PB_obs["cell_counts"].to_numpy().reshape(-1, 1)
        PB_adata.X = X_PB
    elif normalization_method == "lib_size_norm":
        sc.pp.normalize_total(PB_adata, target_sum=target_lib_size)
        sc.pp.log1p(PB_adata)
    else:
        print(f"Unknown normalization_method={normalization_method}. Samples are returned unscaled.")

    return PB_adata
