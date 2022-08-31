#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:45:42 2022

@author: nooteboom
"""
import numpy as np
from scipy.optimize import curve_fit
from netCDF4 import Dataset
import seaborn as sns
from time import time
sns.set()


def switch_pred_prey(pred, prey):
    return prey, pred


def NRMSE(ar1, ar2):
    # Return root mean square deviation between 2D arrays ar1, ar2
    ar1 = ar1.flatten()
    ar2 = ar2.flatten()
    res = 0
    for i in range(len(ar1)):
        res += ((ar1[i] - ar2[i])**2 / len(ar1))**(0.5)
    res /= (max(ar1)-min(ar1))
    return res


def loglik(yPred, yObs):
    #  Calculate the negative log-likelihood as the negative sum of the log of a normal
    #  PDF where the observed values are normally distributed around the mean (yPred)
    #  with a standard deviation of sd

    logLik = -np.nansum((yPred-yObs)**2)

    return logLik


def GOF(func, params, data, X):
    k = len(params)
    if(k == 1):
        pred = func(X, params[0])
    elif(k == 2):
        pred = func(X, params[0], params[1])
    elif(k == 3):
        pred = func(X, params[0], params[1], params[2])
    elif(k == 4):
        pred = func(X, params[0], params[1], params[2], params[3])
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

    #  log likelihood estimation:
    logLik = loglik(pred, data)
    AIC = 2*(k) - 2*logLik
    BIC = k*np.log(len(X[0])) - 2*logLik
    nrmse = NRMSE(data, pred)

    return AIC, nrmse, BIC


def LV(X, a):
    nt, nf = X
    return a * nt


def PC(X, a=1, beta=1):
    nt, nf = X
    return a * nt**beta


def H2(X, a=1, h=1):
    nt, nf = X
    return a*nt / (1+a*h*nt)


def BDA(X, a=1, h=1, w=1):
    nt, nf = X
    return a*nt / (1+a*h*nt + a*w*nf)


def GRD2(X, a=1, h=1, p0=1):
    nt, nf = X
    return a*nt / (a*h*nt + nf/p0 + 1/(1 + nf/p0))


def H3(X, a=1, h=1, n=2):
    nt, nf = X
    return a*nt**n / (1+a*h*nt**n)


def g1(X, w=0.2, w2=0.2, m2=0.5, a=1):
    nt, nf = X
    res2 = a*nt / (1 + w*nf + w2*np.e**(-1*m2*nf))
    return res2


def g2(X, w=0.2, w2=0.2, m2=0.5, a=1, h=1):
    nt, nf = X
    res2 = a*nt / (1 + a*h*nt + w*nf + w2*np.e**(-1*m2*nf))
    return res2


def g3(X, w=0.2, w2=0.2, m2=0.5, a=1, h=1, n=1):
    nt, nf = X
    res2 = a*nt**n / (1 + a*h*nt**n + w*nf + w2*np.e**(-1*m2*nf))
    return res2


def g4(X, w=0.2, w2=0.2, m2=0.5, a=1, h=1, n=1):
    nt, nf = X
    res2 = a*nt**n / (1 + w*nf + w2*np.e**(-1*m2*nf))
    return res2


def g5(X, w=0.2, w2=0.2, m2=0.5, a=1, h=1, n=1, n2=1):
    nt, nf = X
    res2 = a*nt / (1 + a*h*nt*nf/(n*nf**n2+1) + w*nf + w2*np.e**(-1*m2*nf))
    return res2


def g6(X, w=0.2, w2=0.2, m2=0.5, a=1, h=1, n=1, n2=1, n3=1):
    nt, nf = X
    res2 = a*nt**n3 / (1 + a*h*nf/(n*nf**n2+1)*nt**n3 + w*nf + w2*np.e**(-1*m2*nf))
    return res2
#%%
# Functions in which specific trophic functions are chosen


def remove_nans(nt, nf, catch):
    valid = ~(np.isnan(catch))
    return nt[valid], nf[valid], catch[valid]


def chooseTF(typ, nf, nt, catch):
    nt, nf, catch = remove_nans(nt, nf, catch)
    if(typ == 'g1'):
        p0 = 2, 1, 1, 1e-5
        cf = curve_fit(g1, (nt, nf), catch, p0,
                       bounds=(np.array([0,
                                         0, 0, 0]),
                               np.array([np.inf,
                                         np.inf, np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g1, cf[0], catch, (nt, nf))
    elif(typ == 'g2'):
        p0 = 2, 1,  1, 0.5, 1
        cf = curve_fit(g2, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0,
                                         0,
                                         0, 0]),
                               np.array([np.inf, np.inf,
                                         4,
                                         np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g2, cf[0], catch, (nt, nf))
    elif(typ == 'g3'):
        p0 = 2, 1, 1, 0.5, 1, 1
        cf = curve_fit(g3, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0,
                                         0,
                                         0, 0, 1]),
                               np.array([np.inf, np.inf,
                                         4,
                                         np.inf, np.inf, 10])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g3, cf[0], catch, (nt, nf))
    elif(typ == 'g4'):
        p0 = 2, 1, 1, 0.5, 1, 1
        cf = curve_fit(g4, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0,
                                         0,
                                         0, 0, 1]),
                               np.array([np.inf, np.inf,
                                         4,
                                         np.inf, np.inf, 3])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g4, cf[0], catch, (nt, nf))
    elif(typ == 'g5'):
        p0 = 2, 1, 1, 0.5, 0.1, 0.5, 1
        cf = curve_fit(g5, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0,
                                         0,
                                         0, 0, 0, 0]),
                               np.array([np.inf, np.inf,
                                         4,
                                         np.inf, 100, np.inf, 10])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g5, cf[0], catch, (nt, nf))
    elif(typ == 'g6'):
        p0 = 2, 1, 1, 0.5, 0.1, 0.5, 1, 1
        cf = curve_fit(g6, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0,
                                         0,
                                         0, 0, 0, 0, 1]),
                               np.array([np.inf, np.inf,
                                         4,
                                         np.inf, 100, np.inf, 10, 10])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(g6, cf[0], catch, (nt, nf))
    elif(typ == 'LV'):
        p0 = 1
        cf = curve_fit(LV, (nt, nf), catch, p0,
                       bounds=(np.array([0]),
                               np.array([np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(LV, cf[0], catch, (nt, nf))
    elif(typ == 'PC'):
        p0 = 1, 1
        cf = curve_fit(PC, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0]),
                               np.array([np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(PC, cf[0], catch, (nt, nf))
    elif(typ == 'H2'):
        p0 = 1, 0
        cf = curve_fit(H2, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0]),
                               np.array([np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(H2, cf[0], catch, (nt, nf))
    elif(typ == 'BDA'):
        p0 = 1, 0, 1
        cf = curve_fit(BDA, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0, 0]),
                               np.array([np.inf, np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(BDA, cf[0], catch, (nt, nf))
    elif(typ == 'GRD2'):
        p0 = 1, 0, 1
        cf = curve_fit(GRD2, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0, 0]),
                               np.array([np.inf, np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(GRD2, cf[0], catch, (nt, nf))
    elif(typ == 'H3'):
        p0 = 1, 0, 2
        cf = curve_fit(H3, (nt, nf), catch, p0,
                       bounds=(np.array([0, 0, 1]),
                               np.array([np.inf, np.inf, np.inf])),
                       maxfev=5000
                       )
        aic, nrmse, bic = GOF(H3, cf[0], catch, (nt, nf))
    else:
        print('typ incorrect')
        cf = (np.nan, np.nan)
        aic = np.nan
        nrmse = np.nan
        bic = np.nan

    return cf[0], aic, nrmse, bic


def choosep(typ, nf, nt, catch, p):
    if(typ == 'g1'):
        return g1((nt, nf), p[0], p[1], p[2], p[3])
    elif(typ == 'g2'):
        return g2((nt, nf), p[0], p[1], p[2], p[3], p[4])
    elif(typ == 'g3'):
        return g3((nt, nf), p[0], p[1], p[2], p[3], p[4], p[5])
    elif(typ == 'g4'):
        return g4((nt, nf), p[0], p[1], p[2], p[3], p[4], p[5])
    elif(typ == 'g5'):
        return g5((nt, nf), p[0], p[1], p[2], p[3], p[4], p[5], p[6])
    elif(typ == 'g6'):
        return g6((nt, nf), p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    elif(typ == 'LV'):
        return LV((nt, nf), p[0])
    elif(typ == 'PC'):
        return PC((nt, nf), p[0], p[1])
    elif(typ == 'H2'):
        return H2((nt, nf), p[0], p[1])
    elif(typ == 'H3'):
        return H3((nt, nf), p[0], p[1], p[2])
    elif(typ == 'BDA'):
        return BDA((nt, nf), p[0], p[1], p[2])
    elif(typ == 'GRD2'):
        return GRD2((nt, nf), p[0], p[1], p[2])
    else:
        print('typ incorrect')


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


def Bootstrap_curve_fit(typ, tuna, fads, catch, bits=1):
    maxbits = bits + 1000
    x, y = np.meshgrid(tuna, fads)
    res = []
    AICl = []
    BICl = []
    NRMSEl = []
    bo = True
    it = 0
    while bo:
        catchb = create_bs_data(it, catch)
        if(not np.isnan(catchb).all()):
            idx = np.where(catchb > 0)
            try:
                popt, aic, nrmse, bic = chooseTF(typ, y[idx].flatten(),
                                                 x[idx].flatten(),
                                                 catchb[idx].flatten())
                res.append(popt)
                AICl.append(aic)
                BICl.append(bic)
                NRMSEl.append(nrmse)
            except RuntimeError:
                pass
            except ValueError:
                print('one ValueError')
        it += 1
        if(len(res) == bits):
            bo = False
        elif(it >= maxbits):
            bo = False
            print('number of iterations did not succeed')
    if(typ == 'power'):
        print('min and max beta: ', np.array(res).shape)
    return res, AICl, NRMSEl, BICl


def Median_curve_fit(typ, tuna, fads, catch):
    x, y = np.meshgrid(tuna, fads)
    catchb = np.median(catch, axis=0)
    popt, aic, nrmse, bic = chooseTF(typ, y.flatten(), x.flatten(),
                                     catchb.flatten())
    return popt, aic, nrmse, bic


def tot_catch(catch, fads):
    for f in range(len(fads)):
        catch[:, f, :] *= fads[f]
    return catch


def calc_config(p=0.95, con='BJ', sno=1, fs=17, tb='PFeq',
                bits=1, typ='LV', sub=0, removeF1=False):
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
    if(sub == 0):
        nc = Dataset(dirR+'TF_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T,
                                                                P, F))
    elif(sub < 0):
        nc = Dataset(dirR+'TF_fr%d_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (-1*sub,
                                                                     con, p, T,
                                                                     P, F))
    elif(sub > 0):
        nc = Dataset(dirR+'TF_maxC%d_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (sub,
                                                                       con,
                                                                       p, T, P,
                                                                       F))
    else:
        nc = Dataset(dirR+'TF_up_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T,
                                                                   P, F))
    fads = nc['nFADs'][:]
    tuna = nc['ntuna'][:]
    catch = nc['catch'][:]
    if(removeF1):
        catch = catch[:, 1:]
        fads = fads[1:]
    catch = tot_catch(catch, fads)
    tuna2, fads2, catch2 = createXY(fads, tuna, catch)

    for typ in types:
        pops, aics, nrmses, bics = Median_curve_fit(typ, tuna, fads, catch)
        BSpar = np.array(pops)
        aics = np.array(aics)
        bics = np.array(bics)
        nrmses = np.array(nrmses)
        if(removeF1):
            names = 'Poutput/GOF_gF1_median'
        else:
            names = 'Poutput/GOF_median'
        if(sub == 0):
            np.savez(names + '_%s_%s_%s_p%.1f_its%d.npz' % (types[0],
                                                            con,
                                                            tb, p,
                                                            bits),
                     BSpar=BSpar, fads=fads, aics=aics, nrmses=nrmses,
                     bics=bics)
        elif(sub < 0):
            np.savez(names + '_fr%d_%s_%s_%s_p%.1f_its%d.npz' % (-1*sub,
                                                                 types[0],
                                                                 con,
                                                                 tb, p,
                                                                 bits),
                     BSpar=BSpar, fads=fads, aics=aics, nrmses=nrmses,
                     bics=bics)
        elif(sub > 0):
            np.savez(names + '_maxC%d_%s_%s_%s_p%.1f_its%d.npz' % (sub,
                                                                   types[0],
                                                                   con,
                                                                   tb,
                                                                   p,
                                                                   bits),
                     BSpar=BSpar, fads=fads, aics=aics, nrmses=nrmses,
                     bics=bics)
        else:
            np.savez(names + '_up_%s_%s_%s_p%.1f_its%d.npz' % (types[0],
                                                               con,
                                                               tb, p,
                                                               bits),
                     BSpar=BSpar, fads=fads, aics=aics, nrmses=nrmses,
                     bics=bics)


if(__name__ == '__main__'):
    for typ in ['LV', 'PC', 'H2', 'H3', 'BDA', 'GRD2',
                'g1', 'g2', 'g3', 'g4', 'g5', 'g6']:
        for con in ['BJ', 'RW', 'DG']:
            if(True):
                ti = time()
                removeF1 = True
                print('start %s, %s ' % (typ, con))
                calc_config(p=0.95, con=con, tb='PFeq', typ=typ,
                            removeF1=removeF1)
                calc_config(p=0.95, con=con, tb='Pdom', typ=typ,
                            removeF1=removeF1)
                calc_config(p=0.95, con=con, tb='Fdom', typ=typ,
                            removeF1=removeF1)
                calc_config(p=0., con=con, tb='PFeq', typ=typ,
                            removeF1=removeF1)
                print('finish time (min) %.2f \n' % ((time()-ti)/60))
    for con in ['BJ']:
        for typ in ['LV', 'PC', 'H2', 'H3', 'BDA',
                    'GRD2', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6']:
            if(False):
                calc_config(p=0.95, con=con, tb='Pdom', typ=typ)
                calc_config(p=0.95, con=con, tb='Pdom', typ=typ, sub=5)
                calc_config(p=-4, con=con, tb='Pdom', typ=typ, sub=-3)

                calc_config(p=0.95, con=con, tb='Pdom', typ=typ, sub=np.nan)
                print('finished %s, %s' % (con, typ))
