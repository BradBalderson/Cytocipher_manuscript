""" For short, simple plots used for generating diagnostics. """

import numbers
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import upsetplot
import matplotlib
import matplotlib.pyplot as plt

import beautifulcells.visualisation.helpers as vhs

import beautifulcells.visualisation.quick_helpers as qhs

fp = {'weight': 'bold', 'size': 12}

def setUp(font_size=10, axis_font_size=12, fig_title_size=15):
    """Creates bold headings & other matplotlib formatting to create nice plots.
    """
    matplotlib.rcParams.update({'font.size':font_size, 'font.weight': 'bold',
                                'figure.titlesize': fig_title_size,
                                'figure.titleweight': 'bold'})
    global fp
    fp = {'weight': 'bold', 'size': axis_font_size}

def density_scatter(x, y, fig_title='', x_label='', y_label='', figsize=(6.4,4.8),
                    file_name=None, return_ax=False):
    """ Scatter plot with relationship of X & Y, with points coloured by density.
    """
    stack = np.vstack([x, y])
    densities = gaussian_kde(stack)(stack)

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(fig_title)
    ax.scatter(x, y, c=densities)
    ax.set_xlabel(x_label, fp)
    ax.set_ylabel(y_label, fp)
    if not return_ax:
        vhs.dealWithPlot(type(file_name)!=type(None), True, True,
                         '',file_name, 300)
    else:
        return ax

def distrib(x, bins=100, x_label='', fig_title='', log=False, density=False,
            figsize=(6.4,4.8), file_name=None, add_mean=False, logbase=np.e,
            color='blue', alpha=1, ax=None, fig=None, show=True, cutoff=None,
            cutoff_color='r',
            label='', total=None, return_total=False, ylims=None, xlims=None):
    """Plots a histogram of values."""
    if type(ax)==type(None) or type(fig)==type(None):
        fig, ax = plt.subplots(figsize=figsize)

    # Getting the counts in desired format #
    counts, bins = np.histogram(x, bins=bins)
    logcounts = np.log(counts+1)/np.log(logbase) if log else counts
    if density and type(total)==type(None):
        total = sum(logcounts)
        logcounts = logcounts/total
    elif density:
        logcounts = logcounts/total

    ax.hist(bins[:-1], bins, weights=logcounts, color=color, alpha=alpha,
            label=label)
    ax.set_xlabel(x_label, fp)
    if not density:
        ax.set_ylabel(f'log{round(logbase, 2)}-counts' if log else 'counts', fp)
    else:
        ax.set_ylabel('density-'+f'log{round(logbase, 2)}-counts'
                                                      if log else 'density', fp)
    fig.suptitle(fig_title)

    if add_mean:
        mean = np.mean(x)
        y = ax.get_ylim()[1]*.5
        ax.vlines(mean, 0, y, colors=cutoff_color)
        ax.text(mean, y, f'mean:{round(mean, 4)}', c=cutoff_color)
    if cutoff:
        y = ax.get_ylim()[1] * .5
        ax.vlines(cutoff, 0, y, colors=cutoff_color)
        ax.text(cutoff, y, f'cutoff: {round(cutoff, 4)}', c=cutoff_color)

    # Add axes these limits #
    if type(xlims)!=type(None):
        ax.set_xlim(*xlims)
    if type(ylims)!=type(None):
        ax.set_ylim(*ylims)

    # Removing boxes outside #
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show:
        #vhs.dealWithPlot(type(file_name)!=type(None), True, True, '',
        #                                                        file_name, 300)
        plt.show()
    elif not return_total:
        return fig, ax
    else:
        return fig, ax, total

def lineplot(xs, ys, y_names=None, y_colors=None,
             fig_title='', x_label='', y_label='', figsize=(6.4,4.8),
                       file_name=None, return_ax=False, legend_loc='best',
             fig=None, ax=None, show=True):
    """Plots line of multiple data points."""

    if type(y_names)==type(None):
        y_names = list(range(len(ys)))
    if type(y_colors)==type(None):
        y_colors = vhs.getColors(y_names)

    if type(xs[0])!=list and type(xs[0])!=np.array:
        xs = [xs]*len(ys)

    if type(fig)==type(None) or type(ax)==type(None):
        fig, ax = plt.subplots(figsize=figsize)
    for i, y in enumerate(ys):
        ax.plot(xs[i], y, '-', c=y_colors[y_names[i]], linewidth=4,
                label=y_names[i] if type(y_names[i])!=int else None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(x_label, fp)
    ax.set_ylabel(y_label, fp)
    fig.suptitle(fig_title)
    plt.legend(loc=legend_loc)

    if show:
        vhs.dealWithPlot(type(file_name) != type(None), True, True, '',
                         file_name, 300)
    else:
        return ax

def multi_density(values, labels, label_colors=None, bins=100, alpha=.5,
                  fig_title='', x_label='', y_label='', figsize=(6.4,4.8),
                  file_name=None, return_ax=False, legend_loc='best',
                  max_val=None, log=False, add_mean=False, mean_text=False):
    """Plots density of multiple input value across range for comparison."""
    if type(label_colors)==type(None):
        label_colors = vhs.getColors(labels)
    if type(max_val) != type(None):
        values = [np.array(vals)[np.array(vals<max_val)] for vals in values]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(values)):
        ax.hist(values[i], bins, alpha=alpha, label=labels[i], density=True,
                color=label_colors[labels[i]], log=log)

    if add_mean:
        for i in range(len(values)):
            mean = np.mean(values[i])
            y = ax.get_ylim()[1] * .5
            ax.vlines(mean, 0, y, colors=label_colors[labels[i]])
            if mean_text:
                ax.text(mean, y, f'mean:{round(mean, 4)}',
                        c=label_colors[labels[i]])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(x_label, fp)
    ax.set_ylabel(y_label, fp)
    fig.suptitle(fig_title)
    plt.legend(loc=legend_loc)

    if not return_ax:
        vhs.dealWithPlot(type(file_name) != type(None), True, True, '',
                         file_name, 300)
    else:
        return ax

def upset_plot(obj_lists, group_names=None, fig_title='', min_subset_size=1,
               sort_by="cardinality", sort_groups_by=None, show=True):
    """ Creates an upset plot, a visualisation which is useful for comparing \
        overlaps between multiple groups when have more than one group.

    Args:
        obj_lists (list<list<object>>): List of items in different groups want \
                                          to generate upset-plot for to compare.
        group_names (list<str>): List of strings indicating the names for the \
                                                               different groups.
    """
    obj_lists = obj_lists[::-1]
    if type(group_names)==type(None):
        group_names = [f'group_{i}' for i in range(len(obj_lists))]
    else:
        group_names = group_names[::-1]

    upset_df = qhs.get_upset_df(obj_lists, group_names)

    upsetplot.plot(upset_df['c'], sort_by=sort_by,
                   sort_categories_by=sort_groups_by,
                   min_subset_size=min_subset_size)
    plt.title(fig_title, loc='left')
    if show:
        plt.show()


def bar_plot(labels, label_set=None,
             cell_counts: np.array=None, colors=None, cmap=None,
             fig=None, ax=None, figsize=(6,4.8), label_size=10,
             label_name='', axis_text_size=12, tick_size=8,
             fig_title='', n_top=None, horizontal: bool=False,
             rank: bool=True, show=True):

    if type(label_set)==type(None):
        label_set = np.unique(labels)
    if type(colors)==type(None):
        colors = vhs.getColors(labels, label_set, cmap)
        colors = np.array([colors[label] for label in label_set])

    # Counting!!! #
    if type(cell_counts) == type(None):
        cell_counts = []  # Label type frequencies
        for j, label in enumerate(label_set):
            counts = sum(labels == label)
            cell_counts.append(counts)
        cell_counts = np.array(cell_counts)

    if rank:
        order = np.argsort(-cell_counts)
    else:
        order = np.array(list(range(len(cell_counts))))
    order = order if type(n_top)==type(None) else order[0:n_top]
    cell_counts = cell_counts[order]
    colors = colors[order]
    label_set = label_set[order]
    xs = np.array(list(range(len(label_set))))

    # Plotting bar plot #
    if type(fig)==type(None) or type(ax)==type(None):
        fig, ax = plt.subplots(figsize=figsize)

    if not horizontal:
        ax.bar(xs, cell_counts, color=colors)
    else:
        ax.barh(xs, cell_counts, color=colors)
    #ax.set_facecolor('white')
    text_dist = max(cell_counts) * 0.015
    fontdict = {"fontweight": "bold", "fontsize": label_size}
    if label_size > 0:
        for j in range(len(xs)):
            ax.text(
                xs[j],
                cell_counts[j] + text_dist,
                label_set[j],
                rotation=90,
                fontdict=fontdict,
            )
    axis_text_fp = {"fontweight": "bold", "fontsize": axis_text_size}
    ax.set_ylabel(f"{label_name} frequency", color="black", **axis_text_fp)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(f"{label_name} rank", **axis_text_fp)
    fig.suptitle(fig_title)

    if show:
        plt.show()
    else:
        return fig, ax

def rank_scatter(
    items,
    y,
    y_label: str = "",
    x_label: str = "",
    highlight_items=None,
    show_text=True,
    color="gold",
    alpha=0.5,
    lr_text_fp=None,
    axis_text_fp=None,
    ax=None,
    show=True,
    highlight_color="red",
    rot: float = 90,
    point_sizes: np.array = None,
    pad=0.2,
    figsize=None,
    width_ratio=7.5 / 50,
    height=4,
    point_size_name="Sizes",
    point_size_exp=2,
    show_all: bool = False,
):
    """General plotting function for showing ranked list of items."""
    ranks = np.array(list(range(len(items))))

    highlight = type(highlight_items) != type(None)
    if type(lr_text_fp) == type(None):
        lr_text_fp = {"weight": "bold", "size": 8}
    if type(axis_text_fp) == type(None):
        axis_text_fp = {"weight": "bold", "size": 12}

    if type(ax) == type(None):
        if type(figsize) == type(None):
            width = width_ratio * len(ranks) if show_text and not highlight else 7.5
            if width > 20:
                width = 20
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)

    # Plotting the points #
    scatter = ax.scatter(
        ranks,
        y,
        alpha=alpha,
        c=color,
        s=None if type(point_sizes) == type(None) else point_sizes ** point_size_exp,
        edgecolors="none",
    )
    y_min, y_max = ax.get_ylim()
    y_max = y_max + y_max * pad
    ax.set_ylim(y_min, y_max)
    if type(point_sizes) != type(None):
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
        [handle.set_markeredgecolor("none") for handle in handles]
        starts = [label.find("{") for label in labels]
        ends = [label.find("}") + 1 for label in labels]
        sizes = [
            float(label[(starts[i] + 1) : (ends[i] - 1)])
            for i, label in enumerate(labels)
        ]
        counts = [int(size ** (1 / point_size_exp)) for size in sizes]
        labels2 = [
            label.replace(label[(starts[i]) : (ends[i])], "{" + str(counts[i]) + "}")
            for i, label in enumerate(labels)
        ]
        legend2 = ax.legend(
            handles,
            labels2,
            frameon=False,
            # bbox_to_anchor=(0.1, 0.05, 1., 1.),
            handletextpad=1.6,
            loc="upper right",
            title=point_size_name,
        )

    if show_text:
        if highlight:
            ranks_ = ranks[[np.where(items == item)[0][0] for item in highlight_items]]
            ax.scatter(
                ranks_,
                y[ranks_],
                alpha=alpha,
                c=highlight_color,
                s=None
                if type(point_sizes) == type(None)
                else (point_sizes[ranks_] ** point_size_exp),
                edgecolors=color,
            )
            ranks = ranks_ if not show_all else ranks

        for i in ranks:
            ax.text(i - 0.2, y[i], items[i], rotation=rot, fontdict=lr_text_fp)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, axis_text_fp)
    ax.set_ylabel(y_label, axis_text_fp)

    if show:
        plt.show()
    else:
        return ax

def plot_go(go_results, n_top=12, lr_text_fp=None, highlight_go=None,
            highlight_color=None, figsize=(6,4.8), rot=45, show=True,
            max_text=25):
    """Plots go terms outputted from beautifulcells.tools.go.go.
    """

    gos = go_results.loc[:, "Description"].values.astype(str)
    y = -np.log10(go_results.loc[:, "p.adjust"].values)
    sizes = go_results.loc[:, "Count"].values
    rank_scatter(
        gos[0:n_top],
        y[0:n_top],
        point_sizes=sizes[0:n_top],
        highlight_items=highlight_go,
        lr_text_fp=lr_text_fp,
        highlight_color=highlight_color,
        figsize=figsize,
        y_label="-log10(padjs)",
        x_label="GO Rank",
        height=6,
        color="deepskyblue",
        rot=rot,
        width_ratio=0.4,
        show=show,
        point_size_name="n-genes",
        show_all=n_top <= max_text,
    )

def pie(labels: np.array, label_set: np.array=None,
        colors: dict=None, min_perc: float=3, autopct: str='',
        labeldistance=1.05, explode_f=0.8, show=True):
    """Piechart, but formatted nicely, e.g. will order pies & explode out
        low frequency labels.
    """
    if type(label_set)==type(None):
        label_set = np.unique(labels)

    sizes = np.array([len(np.where(labels == dataset)[0])
                      for dataset in label_set])
    order = np.argsort(-sizes)
    sizes = sizes[order]
    label_set = label_set[order]
    percs = np.array([(size/sum(sizes))*100 for size in sizes])

    ### Getting the colors, will add this to scripts.helpers so consistent ###
    if type(colors)==type(None):
        colors = vhs.getColors(label_set)

    colors_ = [colors[dataset] for dataset in label_set]
    end = sum(percs<min_perc)
    explode = [0] * (len(label_set) - end) + [explode_f * (i + 1) for i in list(
        range(end))]  # Expand out less numerous data...

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=label_set, autopct=autopct,  # '%0.1f%%',
            shadow=False, startangle=90, colors=colors_, explode=explode,
            labeldistance=labeldistance)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if show:
        plt.show()
    else:
        return fig1, ax1





