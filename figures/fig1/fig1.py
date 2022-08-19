#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:27:00 2022

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager


def PC(X, q=1, beta=1):
    nt, nf = X
    return q * nt**beta


def H2(X, alpha=1, h=1.5):
    nt, nf = X
    HV = alpha*nt
    return HV * (1+alpha*h*nt)**(-1)


def H3(X, alpha=1, h=1.5):
    nt, nf = X
    HV = alpha*nt**2
    return HV * (1+alpha*h*nt**2)**(-1)


def rPC(pop, beta):
    q = 5/160**beta
    return PC((pop, np.full(pop.shape, 0)), q=q, beta=beta)


def rH2(pop, h=1):
    alpha = 5 / (160*(1-h*5))
    return H2((pop, np.full(pop.shape, 0)), alpha=alpha, h=h)


def rH3(pop, h=1):
    alpha = 5 / (160**2*(1-h*5))
    return H3((pop, np.full(pop.shape, 0)), alpha=alpha, h=h)


def plot(ax, pop=np.arange(0, 160), fs=21, lw=4,
         hfont={'fontname': 'Helvetica'}):
    font = font_manager.FontProperties(family=hfont['fontname'],
                                       weight='bold',
                                       style='normal', size=fs)

    ax.set_ylabel('CPUE (day$^{-1}$)', fontsize=fs, **hfont)
    ax.set_xlabel('tuna population size ($N$)', fontsize=fs, **hfont)
    custom_lines = [Line2D([0], [0], color='k', lw=4),
                    Line2D([0], [0], linestyle='--', color='r', lw=4),
                    Line2D([0], [0], linestyle='--', color='b', lw=4)]
    ax.legend(custom_lines, ['PC', 'H2', 'H3'],
              prop=font, loc='upper left', bbox_to_anchor=(1, 1))
    ax.text(-3, 5, 'Hyperstability', fontsize=fs,
            bbox=dict(boxstyle="square",
                      ec=(.5, 0.5, 0.5),
                      fc=(.85, 0.85, 0.85)))
    ax.text(120, 0, 'Hyperdepletion', fontsize=fs,
            bbox=dict(boxstyle="square",
                      ec=(.5, 0.5, 0.5),
                      fc=(.85, 0.85, 0.85)))

    if(True):
        cpuePC = rPC(pop, beta=0.1)
        cpuePC02 = rPC(pop, beta=0.35)
        cpuePC2 = rPC(pop, beta=1)
        cpuePC3 = rPC(pop, beta=8)
        cpuePC03 = rPC(pop, beta=2.5)
        ax.plot(pop, cpuePC, color='k', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.plot(pop, cpuePC02, color='k', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.plot(pop, cpuePC2, color='k', label=r'$\beta=1$',
                linewidth=lw)
        ax.plot(pop, cpuePC3, color='k', label=r'$\beta=4$',
                linewidth=lw)
        ax.plot(pop, cpuePC03, color='k', label=r'$\beta=4$',
                linewidth=lw)
        ax.text(-3, 4, r'$\beta$=0.1', fontsize=fs)
        ax.text(20, 3.1, r'$\beta$=0.35', fontsize=fs)
        ax.text(70, 2, r'$\beta$=1', fontsize=fs)
        ax.text(83, 0.8, r'$\beta$=2.5', fontsize=fs)
        ax.text(80, -0.2, r'$\beta$=8', fontsize=fs)
    if(True):
        # h<0.2 and h>0 for hyperstability
        cpue = rH2(pop, h=0.19)
        cpue02 = rH2(pop, h=0.15)
        # linear h=2
        cpue2 = rH2(pop, h=0)
        ax.plot(pop, cpue, '--', color='r', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.plot(pop, cpue02, '--', color='r', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.plot(pop, cpue2, '--', color='r', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.text(80, 5, r'$h$=0.19', fontsize=fs, color='r')
        ax.text(57, 4, r'$h$=0.15', fontsize=fs, color='r')
        ax.text(70, 2.6, r'$h$=0', fontsize=fs, color='r')
    if(True):
        cpue = rH3(pop, h=0.)
        ax.plot(pop, cpue, '--', color='b', label=r'$\beta=0.25$',
                linewidth=lw)
        ax.text(60, 1, r'$h$=0', fontsize=fs, color='b')


fig, ax = plt.subplots(figsize=(12, 10))
plot(ax)
plt.savefig('figure1.png', bbox_inches='tight')
plt.show()
