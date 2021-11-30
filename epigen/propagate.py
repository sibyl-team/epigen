import numba
import numpy as np
from numba import jit, int32, float64, boolean, void, int8, int64, njit

@numba.njit( numba.void (numba.int_) )
def set_seed_numba(seed):
    """
    Set the seed for the numba random
    """
    np.random.seed(seed)

# 1 susceptible 2 infected 3 recovery
@numba.jit(float64(float64, float64, float64), nopython=True)
def state_numba(time, inf_time, delay_time):
    """
    Calculate the state, based on discrete-time dynamics
    """
    if time <= inf_time:
        return 1  # su
    if time >= inf_time + delay_time+1: #min value of delay_time=1
        return 3
    return 2

def init_arrays(n, mu, Tinf,discrete_time=True, num_epids=1):
    delay = 0
    if num_epids == 1:
        shape = n
    else:
        shape = (num_epids,n)
    if mu > 0:
        #delay = np.full(n, random.expovariate(mu), dtype=np.float64)
        if discrete_time:
            # The dynamics is discrete, therefore I have to use the geometric distribution
            delay = np.array(np.random.geometric(mu, shape), dtype=np.float64)
        else:
            delay = np.random.exponential(1/mu,shape)
        
    else:
        delay = np.full(shape, Tinf+1, dtype=np.float64)

    epidemy = np.full(shape, float(Tinf), dtype=np.float64)
    fathers = np.full(shape, -1, dtype=np.float64)
    return delay, epidemy, fathers


@numba.jit(float64[:, :](float64[:, :], float64[:], float64[:], float64[:]),  nopython=True)
def propagate_discrete_epi(contacts, epidemy, delay, fathers):
    """
    Propagates epidemy with contacts' times, discrete time
    Uses internal numba generator, need to use @set_numba_seed for the seed
    """
    # print(len(contacts))
    count = 0
    tot = len(contacts)
    count_inf = 0
    #old_states = np.zeros(len(fathers),dtype=np.int16)

    for c in range(len(contacts)):
        time = float(contacts[c][0])
        n1, n2 = int(contacts[c][1]), int(contacts[c][2])
        lambd = contacts[c][3]
        state = state_numba(time, epidemy[n1], delay[n1])
        if state == 2:
            if state_numba(time, epidemy[n2], delay[n2]) == 1:
                if np.random.random() < lambd:
                    count_inf += 1
                    epidemy[n2] = time
                    fathers[n2] = n1
                    #print(n1, " infects ",n2)
        # if state == 3 and old_states[n1] == 2:
        #    print("Node ",n1," recovered at time", time)
        count += 1
        #old_states[n1] = state
    return np.stack((epidemy, fathers))

def get_full_epidemy_trace(times, inf_t, rec_t):
    """
    Get the trace of the epidemy, assuming that rec_t is the recovery delay
    IMPORTANT:
    Infection times are defined as when the nodes BECOME infected

    """
    rec_t = rec_t[:,np.newaxis]
    inf_t = inf_t[:,np.newaxis]
    #res = torch.zeros(times.shape[0],inf_t.shape[0])
    res = (times >= inf_t).astype(int)
    res += (times >= (inf_t+rec_t)).astype(int)
    return res


### SIS model

@numba.jit(void(int64, float64, int64, int8[:, :], int64[:, :]), nopython=True)
def mark_recovered_sis(t, mu, N, track, infecter_track):
    for k in range(N):
        if track[t-1, k] == 1:
            if np.random.rand() < mu:
                track[t, k] = 0
                infecter_track[t, k] = -3
            else:
                track[t, k] = 1

@numba.jit(void(int64, int64, float64, float64[:, :], int8[:, :], int64[:, :]), nopython=True)
def gen_epidemy_sis_numba(N, T, mu, contacts, track, infecter_track):

    #track = np.zeros((T, N), dtype=np.int8)
    #infecter_track = np.full((T, N), -1, dtype=np.int64)
    last_t = 0
    for l in range(len(contacts)):
        t = int(contacts[l][0])
        if(t > last_t):
            # print(track[t-1])
            mark_recovered_sis(t, mu, N, track, infecter_track)
            last_t = t
        elif t < last_t:
            raise ValueError("Contacts have to be ordered in time")
        n1, n2 = int(contacts[l][1]), int(contacts[l][2])
        lambd = contacts[l][3]
        # print(track[t,n1],track[t,n2],infecter_track[t,n2])
        if track[t, n1] == 1 and track[t, n2] == 0 \
                and infecter_track[t+1, n2] < 0:
            if np.random.rand() < lambd:
                track[t+1, n2] = 1
                infecter_track[t+1, n2] = n1
    # set the last t:
    mark_recovered_sis(t+1, mu, N, track, infecter_track)
    # return

def get_recovery_times(trace):
    n = trace.shape[2]
    v = np.cumsum(trace[1] >= 0, axis=0)
    res = v-v*(trace[1] == -3)
    t_rec = []
    for i in range(n):
        int_res = np.unique(res[:, i], return_counts=True)[1]
        if trace[0, -1, i] == 1:
            t_rec.append(int_res[1:-1])
        else:
            t_rec.append(int_res[1:])
    return t_rec

def get_times_infecter(trace):
    times, nodes = np.where(trace[1] >= 0)
    vals = trace[1, times, nodes]
    # time, node index, infected by
    return np.stack((times-1, nodes, vals), -1)

def gen_epidemy_sis(N, t_limit, mu, contacts, n_seed):
    T = t_limit + 1
    track = np.zeros((T, N), dtype=np.int8)
    infecter_track = np.full((T, N), -1, dtype=np.int64)
    source_n = np.random.randint(0, N, n_seed)
    track[0, source_n] = 1
    infecter_track[[0]*n_seed, source_n] = source_n
    gen_epidemy_sis_numba(N, T, mu, contacts, track, infecter_track)
    return np.stack((track, infecter_track), axis=0)


### Numba optimized functions (FOR SOFT MARGIN)

@jit(nopython=True)
def init_arrays_inplace(n, mu, Tinf,delay,epidemy,fathers):    

    epidemy.fill(float(Tinf))
    fathers.fill(-1)
    if mu > 0:
        #delay = np.full(n, random.expovariate(mu), dtype=np.float64)
        # The dynamics is discrete, therefore I have to use the geometric distribution

        for i in range(n):
            delay[i] = np.random.geometric(mu)
    
    else:
        delay.fill(Tinf+1)
@njit
def make_epidemy_inplace(source,n,mu,contacts,epidemy,delay,fathers):
    init_arrays_inplace(n,mu,np.inf,delay,epidemy,fathers)
    epidemy[source] = -1
    epidemy_res = propagate_discrete_epi(contacts,
                                            epidemy, delay,
                                            fathers)
    return epidemy_res

@njit
def get_status_t_numba(t,inf_t,delays):
    """
    Get the state of the epidemy at time t,
    assuming that delays is the recovery delay

    Infection times are defined as in @state_numba 
    (the time before they become infected)
    """
    n = len(inf_t)
    res_arr = np.empty(n,np.int8)
    for i in range(n):
        res_arr[i] = state_numba(t,inf_t[i],delays[i])-1
    return res_arr