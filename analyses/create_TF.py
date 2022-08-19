import numpy as np
from netCDF4 import Dataset
import sys


def avg_catch(nc, nf):
    # Determine average catch per day per dFAD from .nc file
    # Only load the FAD particles and the last 85 / 100 days
    caught = nc['caught'][:nf+2, 30:]
    cmax = np.max(caught, axis=1)
    caught = np.sum(cmax) / (caught.shape[1] / 2) / nf
    return caught

if(__name__ == '__main__'):
    p = 0.
    T = 0.0
    con = str(sys.argv[1])
    tb = str(sys.argv[2])
    if(tb == 'PFeq'):
        F = 1.
        P = 1.0
    elif(tb == 'Fdom'):
        F = 1.5
        P = 0.5
    elif(tb == 'Pdom'):
        F = 0.5
        P = 1.5
    else:
        assert False, 'behaviour incorrect'
    
    assert p in [-1, 0, 0.95]
    assert T in [0, 0.01]
    assert con in ['RW', 'DG', 'BJ']
    if(p == 0):
        FS = 'FS2'
    elif(p == 0.95):
        FS = 'FS3'
    if(F > P):
        dirR = '/nethome/3830241/tuna_project/MC/output%s/%s/Fdom/' % (con, FS)
    elif(F == P):
        dirR = '/nethome/3830241/tuna_project/MC/output%s/%s/PFeq/' % (con, FS)
    elif(F < P):
        dirR = '/nethome/3830241/tuna_project/MC/output%s/%s/Pdom/' % (con, FS)
    
    its = 21
    nfs = np.append(np.array([1, 2, 3, 4, 5]),
                    np.arange(7, 41, 3))
    nts = np.arange(5, 161, 5)
    ncw = Dataset('output/TF_%s_p%.2f_T%.2f_P%.1f_F%.1f.nc' % (con, p, T, P, F),
                  mode='w')
    
    nfss = ncw.createDimension('nFADs', len(nfs))
    ntss = ncw.createDimension('ntuna', len(nts))
    itss = ncw.createDimension('iteration', 21)
    
    nFADs = ncw.createVariable('nFADs', np.float32, ('nFADs', ))
    nFADs[:] = nfs
    ntuna = ncw.createVariable('ntuna', np.float32, ('ntuna', ))
    ntuna[:] = nts
    catch = ncw.createVariable('catch', np.float32, ('iteration', 'nFADs',
                                                     'ntuna', ))
    for it in range(its):
        for nfi, nf in enumerate(nfs):
            for nti, nt in enumerate(range(5, 161, 5)):
                npart = nf+nt+1
                try:
                    ncr = Dataset(dirR + 'FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I0.01_p%.1f_Pa0.1.nc' % (con,
                                                                                                                   it,
                                                                                                                   npart,
                                                                                                                   nf,
                                                                                                                   T,
                                                                                                                   F,
                                                                                                                   P,
                                                                                                                   p))
                    catch[it, nfi, nti] = avg_catch(ncr, nf)
                except:
                    try:
                        T = 0.01
                        print('T1, nf%d, nt%d,  ' % (nf, nt), catch[it, nfi, nti])
                        ncr = Dataset(dirR + 'FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I0.01_p%.1f_Pa0.1.nc' % (con,
                                                                                                                       it,
                                                                                                                       npart,
                                                                                                                       nf,
                                                                                                                       T,
                                                                                                                       F,
                                                                                                                       P,
                                                                                                                       p))
                        catch[it, nfi, nti] = avg_catch(ncr, nf)
                        T = 0
                    except:
                        T = 0.0
                        print('exception')
                        print(dirR+'FADPrey%s_no%d_npart%d_nfad%d_T%.2f_F%.2f_P%.2f_I0.01_p%.1f_Pa0.1.nc' % (con,
                                                                                                             it,
                                                                                                             npart,
                                                                                                             nf,
                                                                                                             T,
                                                                                                             F,
                                                                                                             P,
                                                                                                             p))
                        catch[it, nfi, nti] = np.nan
    
    ncw.close()
