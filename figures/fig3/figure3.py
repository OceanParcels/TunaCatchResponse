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
from matplotlib.lines import Line2D
sns.set(context='paper', style='whitegrid')

colors1 = sns.color_palette("Set1", 9)
colors2 = sns.color_palette("Set2", 8)


def RMSE(ar1, ar2):
    #  Return root mean square deviation between 2D arrays ar1, ar2
    ar1 = ar1.flatten()
    ar2 = ar2.flatten()
    res = 0
    for i in range(len(ar1)):
        res += ((ar1[i] - ar2[i])**2 / len(ar1))**(0.5)
    return res


def PC(X, q=1, beta=1):
    nt, nf = X
    return nf * q * nt**beta


def PC_der(X, q=1, beta=1):
    nt, nf = X
    return nf * q*nt**(beta-1)


def catch_der(typ, nf, nt, con, pars):
    if(typ == 'power'):
        return PC_der((nt, nf), pars[0], pars[1])
    else:
        print('typ incorrect')


def remove_nans(nt, nf, catch):
    valid = ~(np.isnan(catch))
    return nt[valid], nf[valid], catch[valid]


def chooseTF(typ, nf, nt, catch):
    nt, nf, catch = remove_nans(nt, nf, catch)
    if(len(nt) > 0):
        # choose TF for curve fitting
        if(typ == 'power'):
            p0 = 1, 1  # initial guess
            return curve_fit(PC, (nt, nf), catch, p0)
        else:
            print('typ incorrect')
    else:
        return ([np.nan, np.nan], np.nan)


def choosep(typ, nf, nt, catch, p):
    # choose TF for plotting
    if(typ == 'power'):
        return PC((nt, nf), p[0], p[1])
    else:
        print('typ incorrect')


def total_catch(typ, nf, nt, catch, p):
    if(typ == 'power'):
        return nf * PC((nt, nf), p[0], p[1])
    else:
        print('typ incorrect')


def createXY(fads, tuna, catch, its=20):
    x, y = np.meshgrid(tuna, fads)
    x = np.tile(x, (its, 1, 1))
    y = np.tile(y, (its, 1, 1))
    return x.flatten(), y.flatten(), catch.flatten()


def curve_fit_per_it(typ, tuna, fads, catch):
    res = np.zeros((catch.shape[0], len(fads)))
    for it in range(catch.shape[0]):
        for nf in range(len(fads)):
            catchb = catch[it, nf].flatten()
            x, y = np.meshgrid(tuna, np.array([fads[nf]]))
            try:
                popt, pcov = chooseTF(typ, y.flatten(), x.flatten(),
                                      catchb.flatten())
                res[it, nf] = popt[1]
            except RuntimeError:
                print('curve fit runtime error')
    return np.array(res)


def create_bs_data(it, catch):
    ind = np.random.randint(0, catch.shape[0], (catch.shape[1],
                                                catch.shape[2]))
    catchit = np.zeros((catch.shape[1], catch.shape[2]))
    for i in range(catch.shape[1]):
        for j in range(catch.shape[2]):
            catchit[i, j] = catch[ind[i, j], i, j]
    return catchit


def curve_fit_BS(typ, tuna, fads, catch, bits=200):
    res = np.zeros((catch.shape[0], len(fads)))
    np.random.seed(0)
    for it in range(catch.shape[0]):
        catcha = create_bs_data(it, catch)
        for nf in range(len(fads)):
            catchb = catcha[nf]
            idx = np.where(catchb > 0)
            x, y = np.meshgrid(tuna[idx], np.array([fads[nf]]))
            try:
                popt, pcov = chooseTF(typ, y.flatten(), x.flatten(),
                                      catchb.flatten()[idx])
                res[it, nf] = popt[1]
            except RuntimeError:
                print('curve fit runtime error')
    return np.array(res)


def plot_beta(ax, typ, tuna, fads, betas,  fs=17, title='', tb='Pdom',
              con='BJ', maxC=None, uniform=False, sub=1, p=0.95, fr=1):
    pc = 10

    if(sub == 0):
        if(uniform):
            co = colors2[1]
            label = 'uniform prey'
        elif(maxC):
            co = colors2[0]
            label = 'max %d catch' % (maxC)
        elif(fr > 1):
            co = colors2[3]
            label = '%d events per %d days' % (fr, fr)
        elif(sub == 0):
            label = tb
            co = colors2[2]
    elif(sub):
        label = tb
        if(tb == 'Pdom'):
            co = colors1[0]
        elif(tb == 'PFeq'):
            co = colors1[1]
        else:
            co = colors1[2]

    if(True):
        ax.fill_between(fads, np.percentile(betas, pc, axis=0),
                        np.percentile(betas, 100-pc, axis=0), alpha=0.2,
                        color=co)
        if(p == 0.95):
            ax.plot(fads, np.mean(betas, axis=0), linewidth=3, color=co,
                    label=label)
        elif(p == 0.):
            ax.plot(fads, np.mean(betas, axis=0), '--', linewidth=3, color=co,
                    label=label)
    else:
        for i in range(betas.shape[0]):
            ax.plot(fads, betas[i], color='navy', linewidth=2)
    ax.plot([0, 40], [1, 1], '--', c='k', linewidth=2)


def make_boxplot(res, typ, fs=15):
    dirr = {'data': np.array([]),
            'par': np.array([])}
    print('parameter medians: ', np.median(res, axis=0))
    for par in range(res.shape[1]):
        dirr['data'] = np.append(dirr['data'], res[:, par])
        dirr['par'] = np.append(dirr['par'], np.full(res.shape[0], par))

    fig, ax = plt.subplots(1, 1)
    ax.set_title(typ, fontsize=fs)
    sns.boxplot(x='par', y='data', data=dirr)
    plt.show()


def tot_catch(catch, fads):
    for f in range(len(fads)):
        catch[:, f, :] *= fads[f]
    return catch


def load_nc(con='BJ', p=0.95, tb='Pdom', T=0.0, maxC=None,
            uniform=False, fr=1):
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
    if(maxC):
        nc = Dataset(dirR+'TF_maxC%d_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (maxC,
                                                                       con, p,
                                                                       T, P,
                                                                       F))
    elif(uniform):
        nc = Dataset(dirR+'TF_up_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T,
                                                                   P, F))
    elif(fr > 1):
        nc = Dataset(dirR+'TF_fr%d_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (fr,
                                                                     con, p,
                                                                     T, P, F))
    else:
        nc = Dataset(dirR+'TF_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T,
                                                                P, F))
    fads = nc['nFADs'][:]
    tuna = nc['ntuna'][:]
    catch = nc['catch'][:]
    catch = tot_catch(catch, fads)

    return fads, tuna, catch


def make_subplot_con(ax, con='BJ', typ='power', title='', fs=17):
    fads, tuna, catch = load_nc(con, tb='PFeq')
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, tb='PFeq', con=con)
    fads, tuna, catch = load_nc(con, tb='Pdom')
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, tb='Pdom', con=con)
    fads, tuna, catch = load_nc(con, tb='Fdom')
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, tb='Fdom', con=con)
    fads, tuna, catch = load_nc(con, tb='PFeq', p=0.)
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, tb='PFeq', con=con, p=0.)

    if(con == 'BJ'):
        ax.set_title(title + ' BJet', fontsize=fs)
    elif(con == 'DG'):
        ax.set_title(title + ' DEeddy', fontsize=fs)
    elif(con == 'RW'):
        ax.set_title(title + ' RWalk', fontsize=fs)
    ax.set_xlim(fads[0], fads[-1])
    if(con == 'RW'):
        ax.set_xlabel('number of FADs (F)', fontsize=fs)
        ax.set_ylabel(r'$\beta$', fontsize=fs)
        legend_elements = [Line2D([0], [0], color=colors1[0], lw=3,
                                  label='Prey dominant'),
                           Line2D([0], [0], color=colors1[1], lw=3,
                                  label='Equal Prey, FAD'),
                           Line2D([0], [0], color=colors1[2], lw=3,
                                  label='FAD dominant')]
        ax.legend(handles=legend_elements)
    if(con == 'BJ'):
        legend_elements = [Line2D([0], [0], linestyle='--', color='k', lw=3,
                                  label='FSrandom'),
                           Line2D([0], [0], linestyle='-', color='k', lw=3,
                                  label='FSinfo')]
        ax.legend(handles=legend_elements)


def make_subplot2(ax, con='BJ', typ='power', fs=17, title=''):
    fads, tuna, catch = load_nc(con)
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, title=con, con=con, sub=0)
    fads, tuna, catch = load_nc(con, uniform=True)
    betas = curve_fit_BS(typ, tuna, fads, catch)
    plot_beta(ax, typ, tuna, fads, betas, title=con, con=con, uniform=True,
              sub=0)
    if(con == 'BJ'):
        fads, tuna, catch = load_nc(con, maxC=5)
        betas = curve_fit_BS(typ, tuna, fads, catch)
        plot_beta(ax, typ, tuna, fads, betas, title=con, con=con, maxC=5,
                  sub=0)
        fads, tuna, catch = load_nc(con, fr=3, p=-4)
        betas = curve_fit_BS(typ, tuna, fads, catch)
        plot_beta(ax, typ, tuna, fads, betas, title=con, con=con, fr=3, sub=0)
        ax.set_ylabel(r'$\beta$', fontsize=fs)
    elif(con == 'DG'):
        legend_elements = [Line2D([0], [0], color=colors2[0], lw=3,
                                  label='Max 5 catch'),
                           Line2D([0], [0], color=colors2[1], lw=3,
                                  label='Uniform Prey'),
                           Line2D([0], [0], color=colors2[2], lw=3,
                                  label='Reference'),
                           Line2D([0], [0], color=colors2[3], lw=3,
                                  label='3 events per 3 days')
                           ]
        ax.legend(handles=legend_elements)
    if(con == 'BJ'):
        ax.set_title(title + ' BJet', fontsize=fs)
    elif(con == 'DG'):
        ax.set_title(title + ' DEddy', fontsize=fs)
    ax.set_xlabel('number of FADs (F)', fontsize=fs)
    ax.set_xlim(fads[0], fads[-1])


if(__name__ == '__main__'):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    make_subplot_con(ax[0, 0], con='RW', title='(a)')
    make_subplot_con(ax[0, 1], con='BJ', title='(b)')
    make_subplot_con(ax[0, 2], con='DG', title='(c)')

    make_subplot2(ax[1, 1], con='BJ', title='(d)')
    make_subplot2(ax[1, 2], con='DG', title='(e)')

    ax[1, 0].axis('off')
    plt.savefig('figure3.pdf', bbox_inches='tight')
    plt.show()
