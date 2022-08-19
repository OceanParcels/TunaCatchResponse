#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:45:42 2022

@author: nooteboom
"""

import numpy as np
from scipy.optimize import curve_fit
from netCDF4 import Dataset
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as stats
sns.set()


def switch_pred_prey(pred, prey):
    return prey, pred


def NRMSE(ar1, ar2):
    #  Return root mean square deviation between 2D arrays ar1, ar2
    ar1 = ar1.flatten()
    ar2 = ar2.flatten()
    res = 0
    for i in range(len(ar1)):
        res += ((ar1[i] - ar2[i])**2 / len(ar1))**(0.5)
    res /= (max(ar1)-min(ar2))
    return res
#%%%  Functions to calculate AIC


def loglik(params, sd, yPred, yObs):
    #  Calculate the negative log-likelihood as the negative sum of the log of a normal
    #  PDF where the observed values are normally distributed around the mean (yPred)
    #  with a standard deviation of sd
    logLik = np.sum(stats.norm.logpdf(yObs, loc=yPred, scale=sd))
    logLik = np.sum(stats.norm.logpdf(yObs, loc=yPred, scale=sd))

    #  Tell the function to return the NLL (this is what will be minimized)
    return(logLik)


def AIC(func, params, data, X):
    k = len(params)
    if(k == 1):
        pred = func(X, params[0])
    elif(k == 2):
        pred = func(X, params[0], params[1])
    sd = np.std(data-pred)

    #  log likelihood estimation:
    logLik = loglik(params, sd, pred, data)

    AIC = 2*(k+2) - 2*logLik
    return AIC


def LV(X, a):
    nt, nf = X
    return a * nt


def PC(X, a=1, beta=1):
    nt, nf = X
    return a * nt**beta


def remove_nans(nt, nf, catch):
    valid = ~(np.isnan(catch))
    return nt[valid], nf[valid], catch[valid]


def chooseTF(typ, nf, nt, catch):
    nt, nf, catch = remove_nans(nt, nf, catch)
    #  choose TF for curve fitting

    if(typ == 'LV'):
        p0 = 1
        cf = curve_fit(LV, (nt, nf), catch, p0,
                          #  h, w, m, alpha, a
                       bounds = (np.array([0]),
                                 np.array([np.inf])),
                       maxfev=3000)
    elif(typ == 'PC'):
        p0 = 1, 1
        cf = curve_fit(PC, (nt, nf), catch, p0,
                          #  h, w, m, alpha, a
                       bounds = (np.array([0, 0]),
                                 np.array([np.inf, np.inf])),
                       maxfev=3000)
    else:
        print('typ incorrect TF')
        cf = (np.nan, np.nan)

    return cf[0]


def choosep(typ, nf, nt, catch, p):
    if(typ == 'LV'):
        return LV((nt, nf), p[0])
    elif(typ == 'PC'):
        return PC((nt, nf), p[0], p[1])
    else:
        print('typ incorrect p')


def createXY(fads, tuna, catch, its=20):
    x, y = np.meshgrid(tuna, fads)
    x = np.tile(x, (its, 1, 1))
    y = np.tile(y, (its, 1, 1))
    return x.flatten(), y.flatten(), catch.flatten()


def create_bs_data(it, catch):
    np.random.seed(it)
    ind = np.random.randint(0, catch.shape[0], (catch.shape[1],
                                                catch.shape[2]))
    catchit = np.zeros((catch.shape[1], catch.shape[2]))
    for i in range(catch.shape[1]):
        for j in range(catch.shape[2]):
            catchit[i, j] = catch[ind[i, j], i, j]
    return catchit


def Bootstrap_curve_fit(typ, tuna, fads, catch, bits=1):
    maxbits = bits + 1000
    x, y = np.meshgrid(tuna, fads)
    res = []
    bo = True
    it = 0
    while bo:
        catchb = create_bs_data(it, catch)
        if(not np.isnan(catchb).all()):
            idx = np.where(catchb > 0)
            try:
                popt = chooseTF(typ, y[idx].flatten(), x[idx].flatten(),
                                      catchb[idx].flatten())
                res.append(popt)
            except RuntimeError:
                pass
        it += 1
        if(len(res) == bits):
            bo = False
        elif(it >= maxbits):
            bo = False
            print('number of iterations did not succeed')
    if(typ == 'power'):
        print('min and max beta: ', np.array(res).shape)
    return res


def curve_fit_median(typ, tuna, fads, catch):
    x, y = np.meshgrid(tuna, fads)
    catchb = np.median(catch, axis=0)
    if(not np.isnan(catchb).all()):
        try:
            popt = chooseTF(typ, y.flatten(), x.flatten(),
                              catchb.flatten())
        except:
            popt = [0]
    return popt

def plotTF(ig, typ, tuna, fads, catch, popt, popm, fs=23, title='',
           cmap='summer', sno = 1, popm2 = None, popt2 = None):
    axs = ig

    res = np.zeros((len(fads), len(tuna)))
    tuna2 = np.zeros(res.shape)
    for nfi in range(len(fads)):
        for nti in range(len(tuna)):
            res[nfi, nti] = choosep(typ[:2], fads[nfi], tuna[nti], catch, popm)
            tuna2[nfi, nti] = tuna[nti]
    if(typ == 'LVPC'):
        resPC = np.zeros((len(fads), len(tuna)))
        for nfi in range(len(fads)):
            for nti in range(len(tuna)):
                resPC[nfi, nti] = choosep('PC', fads[nfi], tuna[nti], catch, popm2)
        resPC[np.isnan(resPC)] = -10

    res2 = np.nanmedian(catch, axis=0)
    axs.scatter(tuna2.flatten(), res2.flatten(), c='k', alpha=0.3,
                label='IBM')
    axs.plot(tuna, res[0], c='r', lw=3, label=typ[:2])
    if(typ == 'LVPC'):
        axs.plot(tuna, resPC[0], c='b', lw=3, label='PC')
    if(title[:3 ] == '(d)'):
        axs.legend(prop={'size': fs-9})
    axs.set_xlabel('number of tuna ($N$)', fontsize=fs)
    axs.set_ylim(0, res2.max()+0.05)
    if(typ=='LVPC'):
        axs.set_title(title+ r' NRMSE:          ',
                      fontsize=fs)
        axs.text(95, 1.025*(res2.max()+0.05), '%.3f, '%(NRMSE(res2, res)),
                 color='red',
                 fontsize=fs-3)
        axs.text(130, 1.025*(res2.max()+0.05), '%.3f'%(NRMSE(res2, resPC)),
                 color='blue',
                 fontsize=fs-3)
    axs.set_xticklabels(np.arange(-20, 165, 20), size=fs-7)
    if(title[1] == 'a'):
        axs.set_ylabel('time-mean catch (day$^{-1}$)', fontsize=fs)


def update_Bdir(Bdir, BSpar, tb='Pdom', p=0, typ='LV', no=1):
    if(typ in ['LV']):
        Bdir['$a$'] = np.append(Bdir['$a$'], BSpar[:, 0])
    if(typ in ['PC']):
        Bdir['$q$'] = np.append(Bdir['$q$'], BSpar[:, 0])
        Bdir['$\beta$'] = np.append(Bdir['$\beta$'], BSpar[:, 1])

    if(no == 1):
        if(p == 0):
            Bdir['Strategy'] = np.append(Bdir['Strategy'],
                                         np.full(len(BSpar[:, 0]), 'FSrandom'))
        elif(p == 0.95):
            Bdir['Strategy'] = np.append(Bdir['Strategy'],
                                         np.full(len(BSpar[:, 0]), 'FSinfo'))
        if(tb == 'Pdom'):
            Bdir['Behaviour'] = np.append(Bdir['Behaviour'],
                                         np.full(len(BSpar[:, 0]),
                                                 'Forage dominant'))
        elif(tb == 'Fdom'):
            Bdir['Behaviour'] = np.append(Bdir['Behaviour'],
                                         np.full(len(BSpar[:, 0]),
                                                 'FAD dominant'))
        elif(tb == 'PFeq'):
            Bdir['Behaviour'] = np.append(Bdir['Behaviour'],
                                         np.full(len(BSpar[:, 0]),
                                                 'Equal Forage, FAD'))
    return Bdir


def tot_catch(catch, fads):
    for f in range(len(fads)):
        catch[:, f, :] *= fads[f]
    return catch


def make_subplot(TFgrid, p=0.95, con='BJ', sno=1, fs=22, tb='PFeq', Bdir=None,
                 bits=200, typ='LV'):
    TFgrids = TFgrid.subgridspec(1, 1, wspace=0.1, hspace=0.01)
    axs = TFgrids.subplots()
    if(sno == 0):
        tit = '(a) Equal Forage, FAD\n  FSrandom\n'
    if(sno == 1):
        tit = '(b) Equal Forage, FAD\n  FSinfo\n'
    if(sno == 2):
        tit = '(c) Forage dominant\n  FSinfo\n'
    if(sno == 3):
        tit = '(d) FAD dominant\n  FSinfo\n'
    types = [typ]

    T = 0.0
    if(tb == 'PFeq'):
        P = 1.0
        F = 1.0
    elif(tb == 'Pdom'):
        P = 1.5
        F = 0.5
    elif(tb == 'Fdom'):
        P = 0.5
        F = 1.5

    dirR = 'output/'
    nc = Dataset(dirR+'TF_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T, P, F))
    fads = nc['nFADs'][:]
    tuna = nc['ntuna'][:]
    catch = nc['catch'][:]
    catch = tot_catch(catch, fads)
    tuna2, fads2, catch2 = createXY(fads, tuna, catch)
    
    for typ in types:  
        pops = Bootstrap_curve_fit(typ[:2], tuna, fads, catch,
                                            bits=bits)
        if(typ == 'LVPC'):
            popsPC = Bootstrap_curve_fit('PC', tuna, fads, catch,
                                            bits=bits)
            popmPC = curve_fit_median('PC', tuna, fads, catch)
            poptPC = None
            BSparPC = np.array(popsPC)
        else:
            popmPC = None
            poptPC = None
        BSpar = np.array(pops)
        popm = curve_fit_median(typ[:2], tuna, fads, catch)
        Bdir = update_Bdir(Bdir, BSpar, p=p, tb=tb, typ=typ[:2])
        if(typ == 'LVPC'):
            Bdir = update_Bdir(Bdir, BSparPC, p=p, tb=tb, typ='PC', no=2)
        popt = None
        plotTF(axs, typ, tuna, fads, catch, popt, popm, title=tit,
               cmap='cividis',
               sno=sno, popm2=popmPC, popt2=poptPC)
    return Bdir, fads


def plot_boxplots(ax, Bdir, typ, con='BJ'):
    fs = 22
    if(typ == 'LV'):
        ax[0].remove()
        ax[1].remove()
        ax[3].remove()
        ax[4].remove()
        ax[2].set_title(r'(e) $a$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$a$', hue='Behaviour', x='Strategy',
                    ax=ax[2], showfliers = False)
    elif(typ == 'PC'):
        ax[0].remove()
        ax[3].remove()
        ax[1].set_title(r'(e) $q$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$a$', hue='Behaviour', x='Strategy',
                    ax=ax[1], showfliers = False)
        ax[1].legend([], [], frameon=False)
        ax[2].set_title(r'(f) $\beta$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$\beta$', hue='Behaviour', x='Strategy',
                    ax=ax[2], showfliers = False)
    elif(typ == 'LVPC'):
        ax[0].remove()
        ax[4].remove()
        ax[1].set_title(r'(e) $a$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$a$', hue='Behaviour', x='Strategy',
                    ax=ax[1], showfliers = False)
        ax[1].legend([], [], frameon=False)
        ax[2].set_title(r'(f) $q$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$q$', hue='Behaviour', x='Strategy',
                    ax=ax[2], showfliers = False)
        ax[2].legend([], [], frameon=False)
        ax[3].set_title(r'(g) $\beta$', fontsize=fs)
        sns.boxplot(data=Bdir, y='$\beta$', hue='Behaviour', x='Strategy',
                    ax=ax[3], showfliers = False)


def init_Bdir(typ='LV'):
    Bdir = {
            'Behaviour': np.array([]),
            'Strategy': np.array([])
        }
    if(typ in ['LV', 'LVPC']):
        Bdir['$a$'] = np.array([])
    if(typ in ['PC', 'LVPC']):
        Bdir['$q$'] = np.array([])
        Bdir['$\beta$'] = np.array([])
    return Bdir


if(__name__ == '__main__'):
    fig = plt.figure(figsize=(27, 12))
    outer_grid = fig.add_gridspec(2, 1, wspace=0.5, hspace=0.3)
    con = 'RW'
    its = 2000
    print(con)
    typ = 'LVPC'
    assert typ in ['LV', 'PC', 'LVPC']

    Bdir = init_Bdir(typ)
    TFgrid = outer_grid[0].subgridspec(1, 4, wspace=0.2, hspace=0.01)
    Bdir, fads = make_subplot(TFgrid[1], p=0.95, con=con, tb='PFeq', sno=1,
                        Bdir=Bdir, typ=typ, bits=its)
    Bdir, fads = make_subplot(TFgrid[2], p=0.95, con=con, tb='Pdom', sno=2,
                        Bdir=Bdir, typ=typ, bits=its)
    Bdir, fads = make_subplot(TFgrid[3], p=0.95, con=con, tb='Fdom', sno=3,
                        Bdir=Bdir, typ=typ, bits=its)
    Bdir, fads = make_subplot(TFgrid[0], p=0., con=con, tb='PFeq', sno = 0,
                        Bdir=Bdir, typ=typ, bits=its)

    if(typ in ['LV', 'LVPC']):
        Bgrid = outer_grid[1].subgridspec(1, 5, wspace=0.2, hspace=0.01)
    elif(typ in ['PC']):
        Bgrid = outer_grid[1].subgridspec(1, 4, wspace=0.2, hspace=0.01)
    ax = Bgrid.subplots()
    plot_boxplots(ax, Bdir, typ)

    plt.savefig('figure2.pdf', bbox_inches='tight')
    plt.show()
