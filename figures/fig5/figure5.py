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
    # Return root mean square deviation between 2D arrays ar1, ar2
    ar1 = ar1.flatten()
    ar2 = ar2.flatten()
    nanidx = np.logical_or(np.isnan(ar1), np.isnan(ar2))
    ar1 = ar1[~nanidx]
    ar2 = ar2[~nanidx]
    res = 0
    for i in range(len(ar1)):
        res += ((ar1[i] - ar2[i])**2 / len(ar1))**(0.5)
    res /= (np.nanmax(ar1)-np.nanmin(ar2))
    return res


def loglik(params, sd, yPred, yObs):
    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd
    logLik = np.sum(stats.norm.logpdf(yObs, loc=yPred, scale=sd))
    logLik = np.sum(stats.norm.logpdf(yObs, loc=yPred, scale=sd))

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)


def AIC(func, params, data, X):
    k = len(params)
    if(k == 1):
        pred = func(X, params[0])
    elif(k == 2):
        pred = func(X, params[0], params[1])
    elif(k == 5):
        pred = func(X, params[0], params[1], params[2], params[3],
                    params[4])
    elif(k == 6):
        pred = func(X, params[0], params[1], params[2], params[3],
                    params[4], params[5])
    elif(k == 7):
        pred = func(X, params[0], params[1], params[2], params[3],
                    params[4], params[5], params[6])
    elif(k == 8):
        pred = func(X, params[0], params[1], params[2], params[3],
                    params[4], params[5], params[6], params[7])
    elif(k == 9):
        pred = func(X, params[0], params[1], params[2], params[3],
                    params[4], params[5], params[6], params[7], params[8])
    sd = np.std(data-pred)

    # log likelihood estimation:
    logLik = loglik(params, sd, pred, data)

    AIC = 2*(k+2) - 2*logLik
    return AIC


def g5(X, w=0.2, w2=0.2, m=0.5, m2=0.5, a=1, h=1, n=1, n2=1):
    nt, nf = X
    res2 = a*nt / (1 + a*h*nt*nf/(n*nf**n2+1) + w*(nf**m) + w2*(nf**(-1*m2)))
    return res2


def remove_nans(nt, nf, catch):
    valid = ~(np.isnan(catch))
    return nt[valid], nf[valid], catch[valid]


def chooseTF(typ, nf, nt, catch):
    nt, nf, catch = remove_nans(nt, nf, catch)
    p0 = 200, 1e4, 1, 1, 0.5, 0.1, 0.5, 1
    cf = curve_fit(g5, (nt, nf), catch, p0,
                   #  h, w, m, alpha, a
                   bounds=(np.array([0, 0,
                                     0, 0,
                                     0, 0, 0, 0]),
                           np.array([np.inf, np.inf,
                                     np.inf, 4,
                                     np.inf, 200, np.inf, 20])),
                   maxfev=5000)
    return cf[0]


def choosep(typ, nf, nt, catch, p):
    return g5((nt, nf), p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])


def createXY(fads, tuna, catch, its=20):
    x, y = np.meshgrid(tuna, fads)
    x = np.tile(x, (its, 1, 1))
    y = np.tile(y, (its, 1, 1))
    return x.flatten(), y.flatten(), catch.flatten()


def create_bs_data(it, catch, Pdel=''):
    np.random.seed(it)
    catchit = np.zeros((catch.shape[1], catch.shape[2]))
    if(Pdel != 'Pdel'):
        ind = np.random.randint(0, catch.shape[0], (catch.shape[1],
                                                    catch.shape[2]))
        for i in range(catch.shape[1]):
            for j in range(catch.shape[2]):
                catchit[i, j] = catch[ind[i, j], i, j]
    else:
        for i in range(catch.shape[1]):
            for j in range(catch.shape[2]):
                ct = catch[:, i, j]
                ct = ct[~np.isnan(ct)]
                if(len(ct) == 0):
                    catchit[i, j] = np.nan
                else:
                    ind = np.random.randint(len(ct))
                    catchit[i, j] = ct[ind]
    return catchit


def create_bs_datanz(it, catch):
    np.random.seed(it)
    catchit = np.full((catch.shape[1], catch.shape[2]), np.nan)
    for i in range(catch.shape[1]):
        for j in range(catch.shape[2]):
            for it in range(catch.shape[0]):
                ij = 0
                bo = True
                while(ij < 10 and bo):
                    ij += 1
                    randn = np.random.randint(0, catch.shape[0])
                    if(catch[randn, i, j] > 0):
                        catchit[i, j] = catch[randn, i, j]
                        bo = False
        return catchit


def Bootstrap_curve_fit(typ, tuna, fads, catch, bits=1, Pdel=''):
    maxbits = bits + 1000
    x, y = np.meshgrid(tuna, fads)
    res = []
    bo = True
    it = 0
    while bo:
        catchb = create_bs_data(it, catch, Pdel=Pdel)
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


def Median_curve_fit(typ, tuna, fads, catch):
    x, y = np.meshgrid(tuna, fads)
    catchb = np.median(catch, axis=0)
    popt = chooseTF(typ, y.flatten(), x.flatten(),
                    catchb.flatten())
    return popt


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


def plotTF(ig, ax2, typ, tuna, fads, catch, popt, popm, fs=23, title='',
           cmap='summer',
           sno=1):
    axs = ig
    res = np.zeros((len(fads), len(tuna)))
    for nfi in range(len(fads)):
        for nti in range(len(tuna)):
            res[nfi, nti] = choosep(typ, fads[nfi], tuna[nti], catch, popm)
    res[np.isnan(res)] = -10
    res2 = np.nanmedian(catch, axis=0)
    plotbounds = np.linspace(0, min(max(np.nanmax(res), np.nanmax(res2)), 2e1),
                             10)
    im = axs.contourf(tuna, fads, res2, levels=plotbounds,
                      cmap=cmap)
    cb = plt.colorbar(im, cax=ax2)
    for plb in plotbounds:
        cb.ax.plot([0, 15], [plb]*2, 'k')
    if(sno == 3):
        cb.set_label('catch per day', fontsize=fs)
    axs.contour(tuna, fads, res, colors='k', levels=plotbounds)
    axs.set_xlabel('number of tuna ($N$)', fontsize=fs)
    if(sno == 0):
        axs.set_ylabel('number of FADs ($F$)', fontsize=fs)
        axs.set_title(title+'   NRMSE: %.3f' % (NRMSE(res2, res)), fontsize=fs)
    elif(sno == 1):
        axs.set_title(title+'   NRMSE: %.3f' % (NRMSE(res2, res)), fontsize=fs)
    elif(sno == 2):
        axs.set_title(title+'   NRMSE: %.3f' % (NRMSE(res2, res)), fontsize=fs)
    else:
        axs.set_title(title+'   NRMSE: %.3f' % (NRMSE(res2, res)), fontsize=fs)
    if(sno != 0):
        axs.set_yticks([])
    else:
        axs.set_yticklabels(np.arange(0, 41, 5), size=fs-7)
    axs.set_xticklabels(np.arange(0, 165, 20), size=fs-7)


def update_Bdir(Bdir, BSpar, tb='Pdom', p=0):
    Bdir['$w_1$'] = np.append(Bdir['$w_1$'], BSpar[:, 0])
    Bdir['$w_2$'] = np.append(Bdir['$w_2$'], BSpar[:, 1])
    Bdir['$m_1$'] = np.append(Bdir['$m_1$'], BSpar[:, 2])
    Bdir['$m_2$'] = np.append(Bdir['$m_2$'], BSpar[:, 3])
    Bdir[r'$a$'] = np.append(Bdir[r'$a$'], BSpar[:, 4])
    Bdir['$h$'] = np.append(Bdir['$h$'], BSpar[:, 5])
    Bdir['$n$'] = np.append(Bdir['$n$'], BSpar[:, 6])
    Bdir['$n_2$'] = np.append(Bdir['$n_2$'], BSpar[:, 7])

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
                 bits=200, typ='g5', Pdel=''):
    TFgrids = TFgrid.subgridspec(1, 2, wspace=0.1, hspace=0.01,
                                 width_ratios=[20, 1])
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
    if(Pdel == 'Pdel'):
        nc = Dataset(dirR+'TF_Pdel_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T,
                                                                     P, F))
    else:
        nc = Dataset(dirR+'TF_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T, P,
                                                                F))
    fads = nc['nFADs'][:]
    tuna = nc['ntuna'][:]
    catch = nc['catch'][:]
    if(Pdel != 'Pdel'):
        catch = tot_catch(catch, fads)
    tuna2, fads2, catch2 = createXY(fads, tuna, catch)

    for typ in types:
        pops = Bootstrap_curve_fit(typ, tuna, fads, catch,
                                   bits=bits, Pdel=Pdel)
        BSpar = np.array(pops)
        popm = curve_fit_median(typ, tuna, fads, catch)
        Bdir = update_Bdir(Bdir, BSpar, p=p, tb=tb)
        popt = np.median(BSpar, axis=0)
        popt = Median_curve_fit(typ, tuna, fads, catch)
        plotTF(axs[0], axs[1], typ, tuna, fads, catch, popt, popm, title=tit,
               cmap='cividis',
               sno=sno)
    return Bdir, fads


def plot_boxplots(ax, Bdir, typ, con='BJ'):
    fs = 22
    ax[0].set_title(r'(e) $a$', fontsize=fs)
    sns.boxplot(data=Bdir, y=r'$a$', hue='Behaviour', x='Strategy',
                ax=ax[0], showfliers=False)
    ax[0].legend([], [], frameon=False)

    ax[1].set_title('(f) $w_1$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$w_1$', hue='Behaviour', x='Strategy',
                ax=ax[1], showfliers=False)
    ax[1].legend([], [], frameon=False)

    ax[2].set_title('(g) $w_2$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$w_2$', hue='Behaviour', x='Strategy',
                ax=ax[2], showfliers=False)
    ax[2].legend([], [], frameon=False)

    ax[3].set_title('(h) $m_1$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$m_1$', hue='Behaviour', x='Strategy',
                ax=ax[3], showfliers=False)
    ax[3].legend([], [], frameon=False)

    ax[4].set_title('(i) $m_2$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$m_2$', hue='Behaviour', x='Strategy',
                ax=ax[4], showfliers=False)
    ax[4].legend([], [], frameon=False)

    ax[5].set_title('(j) $h$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$h$', hue='Behaviour', x='Strategy',
                ax=ax[5], showfliers=False)

    ax[5].legend([], [], frameon=False)
    ax[6].set_title('(k) $n$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$n$', hue='Behaviour', x='Strategy',
                ax=ax[6], showfliers=False)

    ax[6].legend([], [], frameon=False)
    ax[7].set_title('(l) $n_2$', fontsize=fs)
    sns.boxplot(data=Bdir, y='$n_2$', hue='Behaviour', x='Strategy',
                ax=ax[7], showfliers=False)

    for i in range(len(ax)):
        if(ax[i]):
            ax[i].set_xticklabels(['FSinfo', 'FSrandom'], size=fs-5)


def init_Bdir():
    Bdir = {
            'Behaviour': np.array([]),
            'Strategy': np.array([])
        }
    Bdir['$a$'] = np.array([])
    Bdir['$h$'] = np.array([])
    Bdir['$w_1$'] = np.array([])
    Bdir['$m_1$'] = np.array([])
    Bdir['$m_2$'] = np.array([])
    Bdir['$w_2$'] = np.array([])
    Bdir['$n$'] = np.array([])
    Bdir['$n_2$'] = np.array([])
    return Bdir


if(__name__ == '__main__'):
    fig = plt.figure(figsize=(27, 12))
    outer_grid = fig.add_gridspec(2, 1, wspace=0.5, hspace=0.3)
    con = 'RW'
    Pdel = ''
    its = 2000
    print(con)
    typ = 'g5'
    assert typ in ['LV', 'PC', 'H2', 'H3', 'g1', 'g2', 'g3',
                   'g4', 'g5', 'g5mod', 'g6']

    Bdir = init_Bdir()
    TFgrid = outer_grid[0].subgridspec(1, 4, wspace=0.2, hspace=0.01)

    Bdir, fads = make_subplot(TFgrid[1], p=0.95, con=con, tb='PFeq', sno=1,
                              Bdir=Bdir, typ=typ, bits=its, Pdel=Pdel)
    Bdir, fads = make_subplot(TFgrid[2], p=0.95, con=con, tb='Pdom', sno=2,
                              Bdir=Bdir, typ=typ, bits=its, Pdel=Pdel)
    Bdir, fads = make_subplot(TFgrid[3], p=0.95, con=con, tb='Fdom', sno=3,
                              Bdir=Bdir, typ=typ, bits=its, Pdel=Pdel)
    Bdir, fads = make_subplot(TFgrid[0], p=0., con=con, tb='PFeq', sno=0,
                              Bdir=Bdir, typ=typ, bits=its, Pdel=Pdel)

    Bgrid = outer_grid[1].subgridspec(1, 8, wspace=0.2, hspace=0.01)
    ax = Bgrid.subplots()
    plot_boxplots(ax, Bdir, typ)
    plt.savefig('figure5.pdf',
                bbox_inches='tight')
    plt.show()
