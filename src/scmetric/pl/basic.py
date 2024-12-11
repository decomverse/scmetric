from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# method = ["centroid", "median", "ward", "weighted", "average", "complete", "single"]
def plot_heatmap(
    df,
    cmap="RdBu_r",
    linkage_method: Literal["centroid", "median", "ward", "weighted", "average", "complete", "single"] = "ward",
    linkage_metric: Literal[
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulczynski1",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ] = "euclidean",
    width=8,
    height=6,
    bidirectional=False,
    quants=(0.1, 0.5, 0.9),
    title=None,
    annotate=False,
    **kwargs,
):
    """
    Plots a heatmap with hierarchical clustering.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame to be visualized as a heatmap.
    cmap : str, optional
        The colormap to use for the heatmap. Default is "RdBu_r".
    linkage_method : Literal, optional
        The linkage method to use for hierarchical clustering. Default is "ward".
    linkage_metric : Literal, optional
        The distance metric to use for hierarchical clustering. Default is "euclidean".
    width : int, optional
        The width of the heatmap figure. Default is 8.
    height : int, optional
        The height of the heatmap figure. Default is 6.
    bidirectional : bool, optional
        If True, perform bidirectional clustering. Default is False.
    quants : tuple, optional
        Quantiles to use for setting vmin, center, and vmax of the heatmap. Default is (0.1, 0.5, 0.9).
    title : str, optional
        Title of the heatmap. Default is None.
    annotate : bool, optional
        If True, annotate the heatmap cells. Default is False.
    **kwargs : dict
        Additional keyword arguments to pass to seaborn.heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the heatmap.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the heatmap.
    """
    from scipy.cluster import hierarchy

    X = df.to_numpy().copy()
    m, n = df.shape
    if bidirectional or (m != n):
        row_perm = hierarchy.leaves_list(
            hierarchy.optimal_leaf_ordering(
                hierarchy.linkage(X, method=linkage_method, metric=linkage_metric, optimal_ordering=True), X
            )
        )
        col_perm = hierarchy.leaves_list(
            hierarchy.optimal_leaf_ordering(
                hierarchy.linkage(X.T, method=linkage_method, metric=linkage_metric, optimal_ordering=True), X.T
            )
        )
        df = df.iloc[row_perm, col_perm]
    else:
        perm = hierarchy.leaves_list(
            hierarchy.optimal_leaf_ordering(hierarchy.linkage(X.T, method=linkage_method, metric=linkage_metric), X.T)
        )
        df = df.iloc[perm, perm]

    vmin, center, vmax = np.quantile(X, quants)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
    ax.grid(False)

    ax = sns.heatmap(
        df,
        center=center,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
        annot=annotate,
        fmt=".1f",
        linewidth=0.5,
        linecolor="black",
        clip_on=False,
        ax=ax,
        **kwargs,
    )

    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_corr_scatter(
    df,
    x_attr,
    y_attr,
    x_label="X",
    y_label="Y",
    title="Correlation plot",
    hue_attr=None,
    annot_attr=None,
    hue_pal="tab10",
    width=8,
    height=6,
):
    """
    Plots a scatter plot with a regression line and optional annotations.

    Parameters
    ----------
    df (DataFrame): The data frame containing the data to plot.
    x_attr (str): The column name for the x-axis values.
    y_attr (str): The column name for the y-axis values.
    x_label (str, optional): The label for the x-axis. Default is "X".
    y_label (str, optional): The label for the y-axis. Default is "Y".
    title (str, optional): The title of the plot. Default is "Correlation plot".
    hue_attr (str, optional): The column name for the hue values. Default is None.
    annot_attr (str, optional): The column name for the annotation values. Default is None.
    hue_pal (str, optional): The color palette for the hue values. Default is "tab10".
    width (int, optional): The width of the plot. Default is 8.
    height (int, optional): The height of the plot. Default is 6.

    Returns
    -------
    tuple: A tuple containing the figure and axis objects of the plot.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
    ax.grid(False)

    if hue_attr is None:
        ax = sns.scatterplot(df, x=x_attr, y=y_attr)
    else:
        ax = sns.scatterplot(df, x=x_attr, y=y_attr, hue=hue_attr, palette=hue_pal)

    sns.regplot(df, x=x_attr, y=y_attr, scatter=False, ax=ax)
    plt.text(0.75, 0.05, f"Pearson's r ={np.corrcoef(df[x_attr],df[y_attr])[0,1]:.2f}", transform=ax.transAxes)

    if annot_attr is not None:
        from adjustText import adjust_text

        annots = []
        for i in range(df.shape[0]):
            l = df.iloc[i, :]
            if l[annot_attr] != "":
                annots.append(
                    ax.text(l[x_attr], l[y_attr], l[annot_attr], color="black", fontsize=8, weight="semibold")
                )
        adjust_text(annots, arrowprops={"arrowstyle": "->", "color": "#7F7F7F", "lw": 1}, ax=ax)

    ax.axvline(df[x_attr].mean(), color="#7F7F7F", ls=(0, (5, 5)), alpha=0.8, zorder=0, linewidth=1)
    ax.axhline(df[y_attr].mean(), color="#7F7F7F", ls=(0, (5, 5)), alpha=0.8, zorder=0, linewidth=1)

    x_min = df[x_attr].min()
    x_max = df[x_attr].max()
    x_range = x_max - x_min
    x_min = x_min - 0.1 * x_range
    x_max = x_max + 0.1 * x_range
    ax.set_xlim(x_min, x_max)

    y_min = df[y_attr].min()
    y_max = df[y_attr].max()
    y_range = y_max - y_min
    y_min = y_min - 0.1 * y_range
    y_max = y_max + 0.1 * y_range
    ax.set_ylim(y_min, y_max)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig, ax
