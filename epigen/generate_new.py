import numpy as np
import pandas as pd
import numba as nb
from .base import EpInstance
from .propagate import set_seed_numba

COLUMNS_CONT = ["t","i","j","lamb"]

CONTACT_DTYPES= {"t":int, "i":int, "j":int, "lamb": np.float}

def get_contacts_pd(contacts):
    return pd.DataFrame(contacts,columns=COLUMNS_CONT).astype(CONTACT_DTYPES)

def make_epidemy_pd(T:int, N:int, mu: float, cont_pd: pd.core.frame.DataFrame,
                    ):
    """
    Generate epidemy with pandas
    """
    track = np.zeros((T,N),np.int8)
    source = np.random.randint(0,N)
    track[0,source] = 1

    inf = np.full(N, -2,np.int)
    inf[source] = -1
    succ_cont = np.random.rand(len(cont_pd))< cont_pd.lamb


    for t, df in cont_pd[COLUMNS_CONT[:-1]].groupby(COLUMNS_CONT[0]):
        #print(t)
        for idx,con in df.iterrows():
            if track[t,con.i] == 1 and succ_cont[idx]:
                if track[t, con.j] == 0 and inf[con.j]<-1:
                    track[t+1, con.j] =1
                    inf[con.j] = con.i
        #print(t,track[t])
        #print(t+1,track[t+1])
        for i in range(N):
            if track[t,i] == 1:
                if np.random.rand() < mu:
                    track[t+1,i] = 2
                else:
                    track[t+1,i] = 1
            elif track[t,i] == 2:
                track[t+1,i] = 2
    
    return track, inf

## Numba stuff

nbint = nb.intc
nblong = nb.int_
@nb.njit(nb.void(nbint[:,:], nblong, nb.float_, nblong ))
def move_new(track, N,  mu,  time):
    for i in range(N):
        if track[time,i] == 1:
            if np.random.rand() < mu:
                track[time+1,i] = 2
            else:
                track[time+1,i] = 1
        elif track[time,i] == 2:
            track[time+1,i] = 2
@nb.njit()
def run_epidemy_numba(N, mu,  contacts, track, inf_v, src
                    ):
    """
    Generate epidemy with numba
    """
    num_con = contacts.shape[0]
    if src < 0:
        source = np.random.randint(0,N)
        track[0,source] = 1

        inf_v[source] = -1
    else:
        track[0, src] = 1
        inf_v[src] = 1
    succ = (np.random.rand(num_con)< contacts[:,-1]).astype(np.intc)
    
    time = 0
    for c_i in range(num_con):
       
    
        if time > contacts[c_i,0]:
            raise ValueError("Contacts have to be ordered in time")
        elif time < contacts[c_i,0]:
            move_new(track, N, mu, time)
            time = int(contacts[c_i,0])
        i = int(contacts[c_i,1])
        j = int(contacts[c_i,2])
        #print(t)
        #print(c_i,time,i,j)
        if track[time,i] == 1 and succ[c_i]:
            if track[time, j] == 0 and inf_v[j]<-1:
                track[time+1, j] =1
                inf_v[j] = i
    
    move_new(track, N, mu, time)
    

def make_epidemy_nb(T, N, mu,  contacts, src=-2
                    ):
    """
    Generate epidemy with numba
    """
    track = np.zeros((T,N),np.intc)
    inf = np.full(N, -2,np.int)
    
    run_epidemy_numba(N, mu, contacts, track, inf, src)
    
    return track, inf

@nb.njit(parallel=True)
def _run_many_epids_numba(N, mu, contacts, tracks, inf, n_epi, src):

    for r in nb.prange(n_epi):
        run_epidemy_numba(N, mu, contacts, tracks[r,:,:], inf[r,:], src)


def make_many_epids_nb(T:int, N:int, mu:float, contacts: np.ndarray, n_epi:int, src:int = -2):
    
    tracks = np.zeros((n_epi, T,N),np.intc)
    inf = np.zeros((n_epi,N),np.int)
    inf[:,:] = -2

    _run_many_epids_numba(N, mu, contacts, tracks, inf, n_epi, src)

    return tracks, inf

def make_many_epids_nb_inst(instance: EpInstance, contacts: np.ndarray, n_epi:int, mu = None, src=-2):
    
    T = instance.t_limit + 1
    N = instance.n
    mu = instance.mu if mu is None else float(mu)
    tracks = np.zeros((n_epi, T,N),np.intc)
    inf = np.zeros((n_epi,N),np.int)
    inf[:,:] = -2

    _run_many_epids_numba(N, mu, contacts, tracks, inf, n_epi, src)

    return tracks, inf