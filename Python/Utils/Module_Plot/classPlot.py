#"-*- coding: utf-8 -*-
"""
Created on Wen May 3 2023

@author: Adèle Douin
"""

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import least_squares
from scipy import linalg
from Python.Utils.Module.fct_stats import *
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import logging

from Python.Utils.Module_Plot.fct_plot import *


# ------------------------------------------
class PaperPlot():
    """ Module_Plot clean & automatic plot
    """
    # ---------------------------------------------------------#
    def __init__(self, remote: bool, latex=False) -> None:
        """ Initialisation Paper Plot

        :param remote: deal with plot render => if True only save plot without render
        """
        self.remote = remote

        self.plt = self.version_pylib()

        if latex:
            self.plt.rcParams.update({
                "text.usetex": True,
                'text.latex.preamble': r'\boldmath',
                "font.family": "serif",
                "font.serif": "Computer Modern Roman",
                "font.size": 20,
                "font.weight": "extra bold"
            })
        else:
            self.plt.rcParams.update({
                "text.usetex": False,
                "font.size": 20,
                "font.weight": "bold"
            })
        self.plt.rcParams.update({'axes.titleweight': 'bold'})
        self.plt.rcParams.update({'axes.labelweight': 'bold'})

        self.plt.rc('legend', fontsize=15)
        self.legend_properties = {'weight': 'bold'}
        #
        self.plt.rcParams['figure.figsize'] = (8, 6)

        # ---------------------------------------------------------#

    def version_pylib(self):
        """Deal with pylib version import to prevent render issue when remote analysis
        """
        if self.remote:
            mpl.use('pdf')
            import matplotlib.pyplot as plt
        else:
            # matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt

        return plt

    # ---------------------------------------------------------#
    def belleFigure(self, ax1: str, ax2: str, figsize: (int, int) = None, nfigure: int = None):
        """ Automatic creation of Figure oject

        :param ax1: x axis legend
        :param ax2: y axis legend
        :param figsize: Figure size ; default (8, 6)
        :param nfigure:
        :return: Figure and ax object
        """
        if figsize is None:
            fig = self.plt.figure(nfigure)
        else:
            fig = self.plt.figure(nfigure, figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(ax1, fontdict=dict(weight='bold'))
        ax.set_ylabel(ax2, fontdict=dict(weight='bold'))
        ax.tick_params(axis='both', which='major', width=1)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_weight('bold')
        fig.set_tight_layout(True)
        return fig, ax

    # ---------------------------------------------------------#
    def belleImage(self, ax1, ax2, nrows: int, ncols: int, show_tick: bool = False,
                   figsize: (int, int) = None, nfigure: int = None):
        """ Automatic creation of Figure oject

        :param ax1: x axis legend
        :param ax2: y axis legend
        :param figsize: Figure size ; default (8, 6)
        :param nfigure:
        :return: Figure and ax object
        """
        if isinstance(ax1, np.ndarray):
            ax1 = ax1.astype(str)
        if isinstance(ax2, np.ndarray):
            ax2 = ax2.astype(str)

        if nrows == 1 and ncols == 1:
            if figsize is None:
                fig = self.plt.figure(nfigure)
            else:
                fig = self.plt.figure(nfigure, figsize)
            ax = fig.add_subplot(nrows, ncols, 1)
        else:
            fig, ax = self.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            logging.info('cas 1 - 1')
            if show_tick:
                ax.tick_params(axis='both', which='major', width=1)
                for tick in ax.xaxis.get_ticklabels():
                    tick.set_weight('bold')
                for tick in ax.yaxis.get_ticklabels():
                    tick.set_weight('bold')
            else:
                ax.get_xaxis().set_ticks([])
                ax.get_xaxis().set_ticklabels([])
                ax.get_yaxis().set_ticks([])
                ax.get_yaxis().set_ticklabels([])

            ax.set_xlabel('${}$'.format(ax1), fontdict=dict(weight='bold'))
            ax.set_ylabel('${}$'.format(ax2), fontdict=dict(weight='bold'))

        elif nrows == 1:
            logging.info('cas 1 - ncols')
            logging.debug(ax)
            if show_tick:
                for c, col in enumerate(range(ncols)):
                    ax[c].tick_params(axis='both', which='major', width=1)
                    for tick in ax[c].xaxis.get_ticklabels():
                        tick.set_weight('bold')
                    for tick in ax[c].yaxis.get_ticklabels():
                        tick.set_weight('bold')
            else:
                for c, col in enumerate(range(ncols)):
                    ax[c].get_xaxis().set_ticks([])
                    ax[c].get_xaxis().set_ticklabels([])
                    ax[c].get_yaxis().set_ticks([])
                    ax[c].get_yaxis().set_ticklabels([])

            for c, col in enumerate(range(ncols)):
                ax[c].set_xlabel('${}$'.format(ax1[c]), fontdict=dict(weight='bold'))
            ax[0].set_ylabel('${}$'.format(ax2), fontdict=dict(weight='bold'))

        elif ncols == 1:
            logging.info('cas nrows - 1')
            if show_tick:
                for r, row in enumerate(range(nrows)):
                    ax[r].tick_params(axis='both', which='major', width=1)
                    for tick in ax[r].xaxis.get_ticklabels():
                        tick.set_weight('bold')
                    for tick in ax[r].yaxis.get_ticklabels():
                        tick.set_weight('bold')
            else:
                for r, row in enumerate(range(nrows)):
                    ax[r].get_xaxis().set_ticks([])
                    ax[r].get_xaxis().set_ticklabels([])
                    ax[r].get_yaxis().set_ticks([])
                    ax[r].get_yaxis().set_ticklabels([])

            for r, row in enumerate(range(nrows)):
                ax[r].set_ylabel('${}$'.format(ax2[r]), fontdict=dict(weight='bold'))
            ax[-1].set_xlabel('${}$'.format(ax1), fontdict=dict(weight='bold'))

        else:
            logging.info('cas nroxs - ncols')
            if show_tick:
                for r, row in enumerate(range(nrows)):
                    for c, col in enumerate(range(ncols)):
                        ax[r, c].tick_params(axis='both', which='major', width=1)
                        for tick in ax[r, c].xaxis.get_ticklabels():
                            tick.set_weight('bold')
                        for tick in ax[r, c].yaxis.get_ticklabels():
                            tick.set_weight('bold')
            else:
                for r, row in enumerate(range(nrows)):
                    for c, col in enumerate(range(ncols)):
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_xaxis().set_ticklabels([])
                        ax[r, c].get_yaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticklabels([])

            if isinstance(ax1, np.ndarray):
                for c, col in enumerate(range(ncols)):
                    ax[-1, c].set_xlabel('${}$'.format(ax1[c]), fontdict=dict(weight='bold'))
            if isinstance(ax2, np.ndarray):
                for r, row in enumerate(range(nrows)):
                    ax[r, 0].set_ylabel('${}$'.format(ax2[r]), fontdict=dict(weight='bold'))

        fig.set_tight_layout(True)
        return fig, ax

    # ---------------------------------------------------------#
    def belleMultiFigure(self, N, legend, dict_legends, list_signals, figsize: (int, int) = None):
        """ Automatic creation of Figure oject

        :return: Figure and ax object
        """
        if figsize is None:
            fig, axs = self.plt.subplots(N)
        else:
            fig, axs = self.plt.subplots(N, figsize=figsize)

        axs[-1].set_xlabel('$t \ (s)$', fontdict=dict(weight='bold'))
        for sub in range(N):
            axs[sub].set_ylabel('${}{}$'.format(legend, dict_legends[list_signals[sub]]),
                                fontdict=dict(weight='bold'))
        for j in range(np.size(axs)):
            axs[j].tick_params(axis='both', which='major', width=1)
            for tick in axs[j].xaxis.get_ticklabels():
                tick.set_weight('bold')
            for tick in axs[j].yaxis.get_ticklabels():
                tick.set_weight('bold')
        fig.set_tight_layout(True)
        for j in range(np.size(axs)):
            for tick in axs[j].get_xticklabels():
                tick.set_weight('bold')
            axs[j].grid(True, which='both')
            axs[j].legend()
            self.plt.show()
        return fig, axs

    # ---------------------------------------------------------#
    def make_colors(self, size):
        """Make rainbow colors for

        :param size: nb of sample
        :return: color for each sample
        """
        return [cm.rainbow(i) for i in np.linspace(0, 1, size)]

    # ---------------------------------------------------------#
    def fioritures(self, ax, fig, title, label, grid, save, major=None):
        if title is not None:
            self.plt.title(title)
        if label is not None:
            self.plt.legend(prop=self.legend_properties)
        if grid is not None:
            grid_x_ticks_minor = grid
            ax.set_xticks(grid_x_ticks_minor, minor=True)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            ax.grid(axis='x', which='minor', linestyle='-')
        if major is not None:
            grid_x_ticks_minor = grid
            ax.set_xticks(major)
            ax.set_xticks(grid_x_ticks_minor, minor=True)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            ax.grid(axis='x', which='minor', linestyle='-')
        if save is not None:
            # print(save)
            fig.set_tight_layout(True)
            self.plt.savefig(save + '.pdf')
            self.plt.savefig(save + '.png')
            # self.plt.savefig(save + '.svg')
        if not self.remote:
            self.plt.show()
        else:
            self.plt.close(fig)

    # ---------------------------------------------------------#
    def belleFigureCouple(self, t_label: str, ax0_label: str, ax1_label: str, figsize: (int, int) = None):
        if figsize is None:
            fig, (ax1, ax2) = self.plt.subplots(2)
        else:
            fig, (ax1, ax2) = self.plt.subplots(2, figsize=figsize)
        ax = [ax1, ax2]
        ax[-1].set_xlabel(t_label, fontdict=dict(weight='bold'))
        ax[0].set_ylabel(ax0_label, fontdict=dict(weight='bold'))
        ax[1].set_ylabel(ax1_label, fontdict=dict(weight='bold'))
        for j in range(np.size(ax)):
            ax[j].tick_params(axis='both', which='major', width=1)
            for tick in ax[j].xaxis.get_ticklabels():
                tick.set_weight('bold')
            for tick in ax[j].yaxis.get_ticklabels():
                tick.set_weight('bold')
        fig.set_tight_layout(True)
        for j in range(np.size(ax)):
            for tick in ax[j].get_xticklabels():
                tick.set_weight('bold')
            ax[j].grid(True, which='both')
            ax[j].legend()
        return fig, ax


    def plot_circles(self, X_min, X_max, Y_min, Y_max, X, Y, R, image, sub_set):
        square = self.plt.Rectangle((X_min, Y_min), X_max - X_min, Y_max - Y_min, fill=None, edgecolor='r')
        fig, axs = self.plt.subplots(1, 2)
        axs[0].imshow(image)
        axs[0].add_patch(square)
        axs[0].text(X_max, Y_min, '[{},{}]'.format(X, Y), color='red', ha='right', va='bottom')

        axs[1].imshow(sub_set)
        axs[1].scatter(X, Y)
        circ = Circle((X, Y), R, alpha=1, ec='k', fill=0)
        axs[1].add_patch(circ)
        self.plt.plot()