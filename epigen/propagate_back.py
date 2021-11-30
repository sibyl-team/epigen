import random
import math
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

# 1 susceptible 2 infected 3 recovery
@numba.jit(float64(float64, float64, float64), nopython=True)
def state_numba_cont(time, inf_time, delay_time):
    """
    Calculate the state, based on discrete-time dynamics
    """
    if time <= inf_time:
        return 1  # su
    if time >= inf_time + delay_time:
        return 3
    return 2


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

@numba.njit(parallel=True)
def _run_many_epids_discrete_nb(contacts, epidemy, delay, fathers, sources):

    n_epi = len(sources)
    T0 = -1
    for i in numba.prange(n_epi):
        epidemy[i, sources[i]] = T0
        propagate_discrete_epi(contacts, epidemy[i], delay[i], fathers[i])

def make_epi_discrete(N:int, mu:float, contacts:np.ndarray, n_epi:int=1, source:int=-2):
    """
    Generate one or many discrete epidemies, with optional source `source`
    """

    delay, epidemy, fathers = init_arrays(N, mu, np.inf,num_epids=n_epi)
    T0 = -1
    if n_epi > 1:
        if source > 0:
            sources = np.full(n_epi,source)
        else:
            sources = np.random.randint(0, N, n_epi)

        _run_many_epids_discrete_nb(contacts, epidemy, delay, fathers, sources)

        res = np.stack((epidemy,fathers,delay),axis=1)

    else:
        if source < 0:
            source = np.random.randint(0,N)
        epidemy[source] = T0

        propagate_discrete_epi(contacts,
                               epidemy, delay, fathers)

        res = np.stack((epidemy,fathers,delay))
    
    return res

@numba.njit
def propagate_both_numba(contacts, epidemy, delay, fathers,state_fun,seed=None):
    """propagates epidemy with contacts' times, 
    state_fun decides the timing (discrete or continuous)"""
    #print("in functions")
    # print(len(contacts))
    count = 0
    tot = len(contacts)
    count_inf = 0
    #old_states = np.zeros(len(fathers),dtype=np.int16)
    if seed != None:
        np.random.seed(seed)
    
    for c in range(len(contacts)):
        time = float(contacts[c][0])
        n1, n2 = int(contacts[c][1]), int(contacts[c][2])
        lambd = contacts[c][3]
        state = state_fun(time, epidemy[n1], delay[n1])
        if state == 2:
            if state_fun(time, epidemy[n2], delay[n2]) == 1:
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


def convertContactsNumpy(contacts, mu):
    node2pos = {}
    pos2node = {}
    pos = 0
    pos2node[-1] = "not_infected"
    max_time = 0
    last_time = -1e15
    ordered = True
    for c in contacts:
        if c[1] not in node2pos:
            node2pos[c[1]] = pos
            pos2node[pos] = c[1]
            pos += 1
        if c[2] not in node2pos:
            node2pos[c[2]] = pos
            pos2node[pos] = c[2]
            pos += 1
        if max_time < c[0]:
            max_time = c[0]
        if c[0] < last_time:
            ordered = False
            # return False
        last_time = c[0]
    if not ordered:
        print("Contacts not ordered in time of contact, ordering...")
        contacts = sorted(contacts, key=itemgetter(0))

    contacts_pos = [(float(c[0]), float(node2pos[c[1]]),
                     float(node2pos[c[2]]), float(c[3])) for c in contacts]
    contacts_pos = np.array(contacts_pos, dtype=np.float64)
    # contacts.sort(axis=0)
    Tinf = 1 + max_time

    return contacts_pos, node2pos, pos2node, Tinf


def convertNumpy2Dict(pos2node, epidemy_, delay_pos):
    epidemy_pos = epidemy_[0]
    fathers_pos = epidemy_[1]
    epidemy = {}
    fathers = {}
    delay = {}
    for pos, s in enumerate(epidemy_pos):
        epidemy[pos2node[pos]] = s
    for pos, f in enumerate(fathers_pos):
        fathers[pos2node[pos]] = pos2node[f]
    for pos, d in enumerate(delay_pos):
        delay[pos2node[pos]] = d

    return epidemy, fathers, delay


def simulate_epidemy_not_numpy(sources, contacts, mu, T0=-1, random_seed=0, print_=True):

    contacts_pos, node2pos, pos2node, Tinf = convertContactsNumpy(contacts, mu)

    n_tot = len(node2pos)
    for source in sources:
        if source not in node2pos:
            pos2node[n_tot] = source
            node2pos[source] = n_tot
            n_tot += 1

    delay_pos = 0

    if mu > 0:
        delay_pos = np.random.geometric(p=mu, size=n_tot).astype(np.float64)
    else:
        delay_pos = np.full(n_tot, Tinf+1, dtype=np.float64)
    epidemy_pos = np.full(n_tot, float(Tinf), dtype=np.float64)
    fathers_pos = np.full(n_tot, -1, dtype=np.float64)

    for source in sources:
        source_pos = node2pos[source]
        epidemy_pos[source_pos] = T0
        fathers_pos[source_pos] = source_pos

    if print_:
        print("starting simulation .. ")
    epidemy_ = propagate_discrete_epi(
        contacts_pos, epidemy_pos, delay_pos, fathers_pos)
    # print(epidemy_)
    epidemy, fathers, delay = convertNumpy2Dict(pos2node, epidemy_, delay_pos)

    return (epidemy, fathers, delay, Tinf)


def convert_epi_confs(infection_times, delay, t_limit,
                      one_hot=False, num_states=3, out_dtype=np.int_):
    """
    Convert the epidemic cascade from infection times and delays
    into the full trace (state at each time, 0 -> S, 1 -> I, 2 -> R).
    Give one hot flag to convert the trace in one hot enconding.

    t_limit: last time instant
    
    """
    num_nodes = len(infection_times)

    num_times = int(t_limit)+1

    conf = np.empty((num_times, num_nodes),dtype=np.int_)
    for node, inf_time in enumerate(infection_times):
        for time in range(0,num_times):
            conf[time][node] = state_numba(
                float(time), inf_time, delay[node]) - 1

    if one_hot:
        conf = np.eye(num_states,dtype=out_dtype)[conf]
    return conf

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

def get_status_time(t,inf_t,rec_t):
    """
    Get the state of the epidemy at time t,
    assuming that rec_t is the recovery delay
    IMPORTANT:
    Infection times are defined as when the nodes BECOME infected

    """
    res = (t >= inf_t).astype(int)
    res += (t >= (inf_t+rec_t)).astype(int)
    return res
@njit
def get_status_time_numba_iplc(t,inf_t,rec_t,res_arr):
    """
    Get the state of the epidemy at time t,
    assuming that rec_t is the recovery delay
    IMPORTANT:
    Infection times are defined as in @state_numba 
    (the time before they become infected)

    """
    n = len(res_arr)
    for i in range(n):
        res_arr[i] = state_numba(t,inf_t[i],rec_t[i])-1


def make_epidemy(source,n,mu,contacts):
    delay, epidemy, fathers = init_arrays(n, mu, np.inf)
    epidemy[source] = -1
    epidemy_res = propagate_discrete_epi(contacts,
                                            epidemy, delay,
                                            fathers)
    return epidemy_res, delay




