
#Copyright 2023 Fabio Mazza
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from numba import njit

@njit()
def make_list_stub(deg,pr=False):
    ltot=deg.sum()
    if ltot % 2 == 1:
        ## odd number, need to remove one
        
        imax=np.argmax(deg)
        deg[imax]-=1
        ltot=deg.sum()
        if pr: print(ltot, "is not even, remove from ",imax)
    c = 0
    deglu=np.empty(ltot,dtype=np.int_)
    for i,d in enumerate(deg):
        nc = c+d
        deglu[c:nc] = i
        c = nc
    assert nc == ltot
    return deglu
#@nb.njit()
def make_conf_mod(deg,rng=np.random,trials=10):
    
    stubs = make_list_stub(deg)
    nsedg=np.inf
    ncopy=np.inf
    edg_sav = None
    for i in range(trials):
        rng.shuffle(stubs)
    
        edges = stubs.reshape((2,-1))
        nself=(edges[0]==edges[1]).sum()
        #print(nself)
        if nself == 0:
            edg_sav = edges
            break
        elif nself < nsedg:
            nsedg = nself
            edg_sav = edges
    #print(nsedg)
    edg_sav.sort(0)
    #edges=np.sort(edges,0)
    
    return edg_sav[:,(edges[0]!=edges[1])], edg_sav

def make_contacts(edges):
    edd=np.rec.fromarrays((edges[0],edges[1]),names=["i","j"])
    eu,d=np.unique(edd, return_counts=True)
    return np.rec.fromarrays((eu["i"],eu["j"],d),names=["i","j","c"])

def duplicate_contacts(edges):
    cc= np.concatenate((edges,np.rec.fromarrays(
        (edges["j"],edges["i"], edges["c"]),names=["i","j","c"]
    )))
    cc.sort()
    return cc

def conf_model_days(days,func_gen, seed):
    rng=np.random.RandomState(np.random.PCG64(seed))
    all_cts=[]
    for d in days:
        #s=rng.lognormal(mean=np.log(8),sigma=0.6,size=500)
        s=func_gen(rng)
        sdisc = np.floor(s).astype(int)
        edges,ns = make_conf_mod(sdisc, rng, trials=20)
        con=make_contacts(edges)
        cts=duplicate_contacts(con)
        cts_t = np.rec.fromarrays([cts[n] for n in cts.dtype.names]+[np.full(len(cts),d)],names=list(cts.dtype.names)+["t"])
        all_cts.append(cts_t)
    return all_cts