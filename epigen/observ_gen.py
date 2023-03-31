import warnings
import numpy as np
import pandas as pd

from .generators import calc_epidemies

STATE_VALUE_CONVERT= {
    0: "S", 1: "I", 2:"R"
}

DEFAULT_COLS = {
    "o": "obs",
    "i": "node",
    "t": "time"
}
def make_test_delay_draw(values,probs):
    def get_test_delay(rng=None):
        if rng is None:
            return np.random.choice(values,p=probs)
        else:
            return rng.choice(values, p=probs)
    return get_test_delay

def make_test_delay_poisson(mean):
    def get_delay(rng=None):
        if rng is None:
            return np.random.poisson(mean)
        else:
            return rng.poisson(mean)
    return get_delay


def give_test_delay_fun(p_delay_1):
    return make_test_delay_draw([1,2],[p_delay_1,1-p_delay_1])

def convert_obs_list_numeric(obs_str_list, map_obs=None):
    if map_obs is None:
        map_obs = {r:k for k,r in STATE_VALUE_CONVERT.items()}
    return map(lambda x: map_obs[x],obs_str_list)

def create_observations(infect_t, all_epidemies, num_test_t:int,
            p_asym:float, test_delay_draw, seed=None,
            debug:bool=False, 
            verbose:bool=False,
            get_state_int:bool=False,min_t_inf:int=0, v_old=None):
    """
    Function to make the observations, both in table format and JSON
    infect_t : time instant of infection (BEFORE showing up as infected)
    num_test_t: number of tests at each time instant
    param: p_asym: probability of node being asymptomatic, which means, it does not show symptoms,
        therefore, it is not observed around the time it is infected
    param: test_delay_draw: function to draw the delays for the observation of the symptomatic individuals
    """
    num_test_conf = len(all_epidemies)
    obs_all=[]
    obs_df = []
    
    for inst in range(num_test_conf):

        if seed is None:
            randsg = np.random
        else:
            randsg = np.random.default_rng(np.random.PCG64(seed + 42*inst))
        
        epid_trace = all_epidemies[inst]
        ##
        inf_times = infect_t[inst]
        num_nodes = len(inf_times)
        t_limit = epid_trace.shape[0]-1

        inf = (inf_times != np.inf)
        obs_idx_mat = np.zeros(epid_trace.shape, dtype=np.bool_)
        obs_sympt_times = {}
        for i in inf.nonzero()[0]:
            if randsg.random() < p_asym:
                inf[i]=False
            else:
                ## this guy is symptomatic
                delay_test = test_delay_draw(rng=randsg)
                t_obs = int(inf_times[i])+1+delay_test
                t_obs = max(t_obs, min_t_inf)
                
                ignore = False
                if v_old is not None:
                    if v_old == 1 or v_old == "1":
                        ## old behavior
                        if t_obs > t_limit:
                            ignore = True
                if ignore:
                    ## skip rest
                    continue
                t_obs = t_limit if t_obs > t_limit else t_obs
                #if t_obs <=t_limit:
                if t_obs not in obs_sympt_times.keys():
                    obs_sympt_times[t_obs] = set()
                obs_idx_mat[t_obs,i] = True
                obs_sympt_times[t_obs].add(i)
        #print("Symptomatic nodes", obs_inf_sympto)
        ## from now on, inf contains the nodes excluded from ranking
        untested = np.ones_like(inf)
        for t in range(t_limit+1):
            if t in obs_sympt_times:
                sympt_today = obs_sympt_times[t]
                untested[list(sympt_today)] = False
            idx_choose = np.where(untested)[0]
            randsg.shuffle(idx_choose)
            chosen = idx_choose[:num_test_t]
            ## save tested
            obs_idx_mat[np.full(len(chosen),t),chosen] = True

            vals_nodes = epid_trace[t][chosen]
            idx_g = chosen[vals_nodes!=0]
            untested[idx_g] = False
            if debug:
                print(t, len(idx_choose),chosen, vals_nodes, idx_g)

        obs_t_i = np.where(obs_idx_mat)
        ## Create observations in csv
        df = pd.DataFrame(np.stack(obs_t_i).T,columns=["time","node"])
        if get_state_int:
            df["obs"] = np.array(epid_trace[obs_t_i])
        else:
            df["obs"] = list(map(lambda x: STATE_VALUE_CONVERT[x],epid_trace[obs_t_i]))
        obs_df.append(df)

        ## Create observation in JSON
        observ = {"S":{},"I":{},"R":{}}
        for t,i,val in zip(*obs_t_i,epid_trace[obs_t_i]):
            stat = STATE_VALUE_CONVERT[val]
            tt = int(t)
            ii = int(i)
            if t not in observ[stat].keys():
                observ[stat][tt] = []
            observ[stat][tt].append(ii)
        obs_all.append(observ)
        if verbose:
            obsperN = {}
            obsinf = set()
            for l in range(num_nodes):
                obsdone = obs_idx_mat[:,l]
                if np.any(obsdone):
                    obsperN[l] = {}
                    obsperN[l] = tuple(zip(np.where(obsdone)[0], epid_trace[obsdone,l]))
                    if np.any(epid_trace[obsdone,l] == 1):
                        obsinf.add(l)
            #print("INF NODES:", sorted(obsinf))
            #print("ALL OBS NODES: ",obsperN.keys())
            #print(obsperN)

    return obs_df, obs_all


def make_sparse_obs_default(data_, t_limit, ntests, pr_sympt, p_test_delay,
    seed=None, verbose=False, numeric_obs=False,min_t_inf:int=-1, old_ver=None):
    """
    Make observations given the data produced by epidemy gen
    """
    gen_epi_data = data_["epidemy"]
    infect_times = [g[0][0] for g in gen_epi_data]
    full_epi = calc_epidemies(gen_epi_data,
            data_["test"],t_limit)
    fun_test_delay = make_test_delay_draw(np.arange(len(p_test_delay)), p_test_delay)

    obs_df, obs_json = create_observations(
            infect_times, full_epi, ntests, 1. - pr_sympt, fun_test_delay,
            seed=seed,
            debug=False, verbose=verbose,
            get_state_int=numeric_obs,
            min_t_inf=min_t_inf, v_old=old_ver
        )

    return obs_df, obs_json

def define_delay_probs(p_test_delay):
    if p_test_delay[0] == "uniform":
        delay_max = int(p_test_delay[1])
        print(f"Uniform observ days: {delay_max}")
        probs = np.ones(delay_max+1)
        return probs/probs.sum()
    else:
        
        try:
            #v=float(p_test_delay[0])
            probs = np.array([float(x) for x in p_test_delay])
            return probs/probs.sum()
        except ValueError as e:
            raise ValueError("Have to give probabilities") from e

def create_obs_for_inf_last(infect_t, all_epidemies,
            p_asym:float, seed=None,
            debug:bool=False, 
            verbose:bool=False,
            get_state_int:bool=False,):
    """
    Function to make the observations, both in table format and JSON
    infect_t : time instant of infection (BEFORE showing up as infected)
    num_test_t: number of tests at each time instant
    param: p_asym: probability of node being asymptomatic, which means, it does not show symptoms,
        therefore, it is not observed around the time it is infected
    param: test_delay_draw: function to draw the delays for the observation of the symptomatic individuals
    """
    num_test_conf = len(all_epidemies)
    obs_all=[]
    obs_df = []
    T_obs = -1
    
    for inst in range(num_test_conf):

        if seed is None:
            randsg = np.random
        else:
            randsg = np.random.default_rng(np.random.PCG64(seed + 42*inst))
        
        epid_trace = all_epidemies[inst]
        ##
        inf_times = infect_t[inst]
        num_nodes = len(inf_times)

        inf = (inf_times != np.inf)
        obs_idx_mat = np.zeros(epid_trace.shape, dtype=np.bool_)
        for i in inf.nonzero()[0]:
            if randsg.random() < p_asym:
                inf[i]=False
            else:
                ## this guy is symptomatic
                #delay_test = test_delay_draw(rng=randsg)
                ### Only test at the this time
                obs_idx_mat[T_obs,i] = True
        #print("Symptomatic nodes", obs_inf_sympto)
        ## from now on, inf contains the nodes excluded from ranking

        obs_t_i = np.where(obs_idx_mat)
        ## Create observations in csv
        df = pd.DataFrame(np.stack(obs_t_i).T,columns=["time","node"])
        if get_state_int:
            df["obs"] = np.array(epid_trace[obs_t_i])
        else:
            df["obs"] = list(map(lambda x: STATE_VALUE_CONVERT[x],epid_trace[obs_t_i]))
        obs_df.append(df)

        ## Create observation in JSON
        observ = {"S":{},"I":{},"R":{}}
        for t,i,val in zip(*obs_t_i,epid_trace[obs_t_i]):
            stat = STATE_VALUE_CONVERT[val]
            tt = int(t)
            ii = int(i)
            if t not in observ[stat].keys():
                observ[stat][tt] = []
            observ[stat][tt].append(ii)
        obs_all.append(observ)
        if verbose:
            obsperN = {}
            obsinf = set()
            for l in range(num_nodes):
                obsdone = obs_idx_mat[:,l]
                if np.any(obsdone):
                    obsperN[l] = {}
                    obsperN[l] = tuple(zip(np.where(obsdone)[0], epid_trace[obsdone,l]))
                    if np.any(epid_trace[obsdone,l] == 1):
                        obsinf.add(l)
            #print("INF NODES:", sorted(obsinf))
            #print("ALL OBS NODES: ",obsperN.keys())
            #print(obsperN)

    return obs_df, obs_all


def make_sparse_obs_last_t(data_, t_limit, pr_sympt, seed=None, verbose=False, numeric_obs=False):
    """
    Make observations given the data produced by epidemy gen
    """
    gen_epi_data = data_["epidemy"]
    infect_times = [g[0][0] for g in gen_epi_data]
    full_epi = calc_epidemies(gen_epi_data,
            data_["test"],t_limit)


    obs_df, obs_json = create_obs_for_inf_last(
            infect_times, full_epi, 1. - pr_sympt,
            seed=seed,
            debug=False, verbose=verbose,
            get_state_int=numeric_obs,
        )

    return obs_df, obs_json

def make_obs_new_trace(trace_epi:np.ndarray, p_test_delay, 
            p_sympt:float, n_test_rnd:int,
            seed=None,
            tobs_inf_min:int=-1,
            tobs_rnd_lim:tuple=(0, None),
            allow_testing_pos=False):
    """
    New function to generate observations, from the trace
    trace_epi: trace of epidemy, T x N
    p_test_delay: list of probabilities for different times

    """

    T,N = trace_epi.shape
    if seed == None:
        rng = np.random
    else:
        ## FIX
        rng = np.random.default_rng(np.random.PCG64(seed))

    obs_free = np.ones((N,T),dtype=np.int8)

    p_test_delay = np.array(p_test_delay)/sum(p_test_delay)
    
    ## filter out infs (not infected)
    infected = (trace_epi!=0).sum(0) >0
    inf_idx = np.where(infected)[0]
    ## infection times
    tinf_c = np.argmax(trace_epi[:,inf_idx]!=0,0)
    ## select observed infected
    is_sel = rng.random(tinf_c.shape) < p_sympt
    sel = inf_idx[is_sel]
    #print(sel)
    tinf_sel = tinf_c[is_sel].astype(int) ## time they become infected
    
    # extract delays
    c = rng.random(len(tinf_sel))[:,np.newaxis]
    delays = np.argmax(c < p_test_delay.cumsum(), axis=1)
    times_obs_inf = tinf_sel+delays
    ## if some inf_t_obs are > T, they will not be observed
    if tobs_inf_min >= 0:
        times_obs_inf[times_obs_inf < tobs_inf_min] = tobs_inf_min
    #( sel, times_obs_inf )
    if not allow_testing_pos:
        for i, tobs in zip(sel, tinf_sel):
            obs_free[i][tobs:] = 0

    obs_final = []
    max_t_rnd = tobs_rnd_lim[1] if tobs_rnd_lim[1]!=None else T
    for t in range(T):
        infobs = sel[times_obs_inf==t]
        #print(t, infobs)
        obs_final += list((i,trace_epi[t,i], t) for i in infobs)
        if n_test_rnd==0 or t<tobs_rnd_lim[0] or t >max_t_rnd:
            ## skip the random obs
            continue
        rnd_obs_allow = np.where(obs_free[:,t])[0]
        rng.shuffle(rnd_obs_allow)
        rnd_obs_idx = rnd_obs_allow[:n_test_rnd]
        #print(len(rnd_obs))
        if len(rnd_obs_idx) < n_test_rnd:
            warnings.warn("Number of available people to to test < n_test_rnd = {}".format(n_test_rnd))
        
        for k in rnd_obs_idx:
            if not allow_testing_pos and trace_epi[t,k] != 0:
                ### don't allow testing
                obs_free[k,t:] = 0
            
            obs_final.append((k, trace_epi[t,k], t))

    return obs_final

def gen_obs_new_data(epidata, epInstance, p_test_delay, 
            p_sympt:float, n_test_rnd:int,
            seed=None,
            tobs_inf_min:int=-1,
            tobs_rnd_lim:tuple=(0, None),
            allow_testing_pos=False):
    
    epidemies = calc_epidemies(epidata["epidemy"], epidata["test"], epInstance.t_limit)

    #all_obs = []
    for i, trace in enumerate(epidemies):
        s = seed + 42*i
        
        yield  make_obs_new_trace(trace, p_test_delay=p_test_delay,
            p_sympt=p_sympt, n_test_rnd=n_test_rnd,seed=s,
            tobs_inf_min=tobs_inf_min,
            tobs_rnd_lim=tobs_rnd_lim,
            allow_testing_pos=allow_testing_pos)


def gen_obs_custom(epidata, epInstance,gen_funct, seed, *args, **kwargs):
    
    epidemies = calc_epidemies(epidata["epidemy"], epidata["test"], epInstance.t_limit)

    #all_obs = []
    for i, trace in enumerate(epidemies):
        s = seed + 42*i
        kwargs["seed"] = s
        yield  gen_funct(trace,*args,**kwargs)
