#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:28:55 2022

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from copy import copy
sns.set()


def init_pDir():
    pDir = {'trophic function': np.array([]),
            'behaviour': np.array([]),
            'AIC': np.array([])}
    return pDir


def update_pDir(pDir, typ, con, tb, p, GOFt='AIC', bootstrapping=False,
                removeF1=False):
    if(bootstrapping):
        file = np.load('Poutput/GOF_%s_%s_%s_p%.1f_its%d.npz' % (typ,
                                                                 con,
                                                                 tb,
                                                                 p,
                                                                 2000))
    elif(removeF1):
        file = np.load('Poutput/GOF_gF1_median_%s_%s_%s_p%.1f_its%d.npz' % (typ,
                                                                        con,
                                                                        tb,
                                                                        p,
                                                                        1))
    else:
        file = np.load('Poutput/GOF_median_%s_%s_%s_p%.1f_its%d.npz' % (typ,
                                                                        con,
                                                                        tb,
                                                                        p,
                                                                        1))
    if(GOFt == 'AIC'):
        gof = np.mean(file['aics'])
    elif(GOFt == 'NRMSE'):
        gof = np.mean(file['nrmses'])
    elif(GOFt == 'BIC'):
        gof = np.mean(file['bics'])
    pDir['AIC'] = np.append(pDir['AIC'], gof)
    pDir['trophic function'] = np.append(pDir['trophic function'], typ)
    if(tb == 'PFeq'):
        pDir['behaviour'] = np.append(pDir['behaviour'],
                                      'Equal forage, FAD')
    elif(tb == 'Pdom'):
        pDir['behaviour'] = np.append(pDir['behaviour'],
                                      'Forage dominant')
    elif(tb == 'Fdom'):
        pDir['behaviour'] = np.append(pDir['behaviour'],
                                      'FAD dominant')
    return pDir


if(__name__ == '__main__'):
    fs = 16
    GOFt = 'AIC'
    removeF1 = False
    assert GOFt in ['AIC', 'NRMSE', 'BIC']
    typs = ['LV', 'PC', 'H2', 'H3', 'BDA', 'GRD2', 'g1',
            'g2', 'g3',
            'g4', 'g5',
            'g6']
    fig = plt.figure(figsize=(16, 5))
    outer_grid = fig.add_gridspec(1, 4, wspace=0.12,
                                  width_ratios=[1, 1, 1, 0.1])

    axso = outer_grid.subplots()
    axso[0].set_title('(a) Random Walk\n', fontsize=fs)
    axso[1].set_title('(b) Bickley Jet\n', fontsize=fs)
    axso[2].set_title('(c) Double Eddy\n', fontsize=fs)
    axso[0].axis('off')
    axso[1].axis('off')
    axso[2].axis('off')
    axso[3].tick_params(labelsize=fs-3)

    if(GOFt == 'AIC'):
        cmap = 'Greys'
        fmt = ".0f"
        vs = [20, 1000]
        axso[3].set_title('AIC', fontsize=fs)
    elif(GOFt == 'NRMSE'):
        cmap = 'cividis_r'
        fmt = ".2f"
        vs = [0.35, 3]
        axso[3].set_title('NRMSE', fontsize=fs)
    elif(GOFt == 'BIC'):
        cmap = 'Greys'
        fmt = ".0f"
        vs = [50, 2600]
        axso[3].set_title('BIC', fontsize=fs)
    for ci, con in enumerate(['RW', 'BJ', 'DG']):
        inner_grid = outer_grid[ci].subgridspec(1, 2, wspace=0.03, hspace=0)
        axs = inner_grid.subplots()
        for pi, p in enumerate([0., 0.95]):
            pDir = init_pDir()
            for typ in typs:
                if(p == 0):
                    pDir = update_pDir(pDir, typ, con, 'PFeq', p, GOFt=GOFt,
                                       removeF1=removeF1)
                else:
                    for tb in ['Fdom', 'PFeq', 'Pdom']:
                        pDir = update_pDir(pDir, typ, con, tb, p, GOFt=GOFt,
                                           removeF1=removeF1)

            pDir = pd.DataFrame.from_dict(pDir)
            pDir = pDir.pivot("trophic function", "behaviour", "AIC")
            # set the order of the heatmap rows:
            pDir.index = pd.CategoricalIndex(pDir.index, categories=typs)
            pDir.sort_index(level=0, inplace=True)
            if(GOFt == 'NRMSE'):
                g = sns.heatmap(data=pDir, ax=axs[pi],
                                annot=True, fmt=fmt,
                                vmin=vs[0], vmax=vs[1], cbar_ax=axso[-1],
                                cmap=cmap,
                                linewidths=.5, linecolor='k')
            else:
                pDirnc = copy(pDir)
                # normalize every column individually to plot their colorscale
                if(len(pDirnc.keys()) > 1):
                    for key in pDirnc.keys():
                        pDirnc[key] = (pDirnc[key]-pDirnc[key].mean()) / pDirnc[key].std() * 100
                if(p == 0):
                    if(con == 'RW'):
                        vm = 200
                    elif(con == 'BJ'):
                        vm = 800
                    else:
                        vm = 1500
                else:
                    vm = 1000
                g = sns.heatmap(data=pDirnc,
                                ax=axs[pi],
                                annot=False,
                                cbar_ax=axso[-1],
                                cmap=cmap,
                                linewidths=.5, linecolor='k')
                g = sns.heatmap(data=pDir, ax=axs[pi],
                                annot=True, fmt=fmt,
                                alpha=0,
                                cbar_ax=axso[-1],
                                cmap=cmap, vmax=vm,
                                linewidths=.5, linecolor='k')
            if(pi == 0):
                axs[pi].set_title('FSrandom', fontsize=fs)
            elif(pi == 1):
                axs[pi].set_title('FSinfo', fontsize=fs)
            if(ci > 0 or pi > 0):
                axs[pi].set_ylabel('')
                axs[pi].set_yticklabels(['']*len(typs))
            else:
                axs[pi].set_ylabel('trophic function', fontsize=fs)
                axs[pi].set_yticklabels(['LV', 'PC', 'H2', 'H3', 'BDA',
                                         'GRD2', 'g$_1$', 'g$_2$', 'g$_3$',
                                         'g$_4$', 'g$_5$', 'g$_6$'],
                                        fontsize=fs)
                g.set_yticklabels(g.get_yticklabels(), rotation=0,
                                  horizontalalignment='right')
            axs[pi].set_xlabel('behaviour', fontsize=fs)
            axs[pi].tick_params(labelsize=fs)
    if(GOFt != 'NRMSE'):
        axso[-1].remove()
    if(removeF1):
        plt.savefig('figure4_Fg0_%s.pdf' % (GOFt), bbox_inches='tight')
    else:
        plt.savefig('figure4_%s.pdf' % (GOFt), bbox_inches='tight')
    plt.show()
