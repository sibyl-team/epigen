import numpy as np
import pandas as pd
import random

from . import propagate

def set_seed(seed):
    propagate.set_seed_numba(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_contacts(G, t_limit, lambda_, p_edge=1, seed=1):
    contacts = []
    random.seed(seed)
    for t in range(t_limit):
        for e in G.edges():
            if random.random() <= p_edge:
                contacts.append((t, e[0], e[1], lambda_))
                contacts.append((t, e[1], e[0], lambda_))
    contacts = np.array(contacts, dtype=[("t","f8"), ("i","f8"), ("j","f8"), ("lam", "f8")])
    contacts.sort(axis=0, order=("t","i","j"))
    return contacts.view("f8").reshape(-1,4) #.astype(np.float32)


def generate_one_sim(contacts, mu, t_limit, n, n_seed, sources=None):
    """
    MC simulations of SIR, return configuration at T0 and Tend

     contacts : list([time, node_i, node_j, lambda_ij])
     mu: recovery probability
     t_limit: total numbers of epoch
     n: number of nodes
     random_seed: seed for random library, [default -1, random_seed]
    """
    if sources == None:
        sources = set()  #[ for _ in range(n_seed)]
        while len(sources) < n_seed:
            src = random.randint(0, n - 1)
            if src not in sources:
                sources.add(src)
    print(f" Sources {sources}")

    #sources = np.random.randint(0,n, n_seed)
    T0 = -1
    delay, epidemy, fathers = propagate.init_arrays(n, mu, np.inf)
    epidemy[list(sources)] = T0
    start_conf = (epidemy == T0).astype(int)
    epidemy_res = propagate.propagate_discrete_epi(contacts,
                                                    epidemy, delay,
                                                    fathers)
    infected = (epidemy_res[0] < t_limit).astype(int)
    recovery = (infected * ((epidemy_res[0] + delay) < t_limit)).astype(int)
    end_conf = infected + recovery
    num_inf = np.sum(infected)
    num_recovery = np.sum(recovery)
    return {"confs": [start_conf.tolist(), end_conf.tolist()], "inf_rec": [infected, recovery],
            "num_infected": num_inf, "num_recovery": num_recovery, "epidemy": epidemy_res, "delays": delay}

def generate_sis_sim(contacts, mu, t_limit, n, n_sources, sources = None):
    """
    Generate SIS epidemy
    """
    trc = propagate.gen_epidemy_sis(n, t_limit, mu, contacts, n_sources)
    recv_times = propagate.get_recovery_times(trc)
    epidemy = propagate.get_times_infecter(trc)
    conf_start = trc[0, 0]
    conf_end = trc[0, -1]
    num_rec = len(np.hstack(recv_times))
    num_rec_per_node = [len(x) for x in recv_times]
    return {"confs": [conf_start.tolist(), conf_end.tolist()], "num_infected": len(epidemy),
            "num_recovery": num_rec, "num_recov_nodes": num_rec_per_node,
            "epidemy": epidemy, "delays": [x.tolist() for x in recv_times]}


def generate_configurations(n, contacts,
                            mu, t_limit,
                            num_conf=10000,
                            num_source=1,
                            use_SIS=False,
                            give_rec_times=True,
                            lim_infected = 2,
                            max_infected = None,
                            print_ = True,
                            unique_nI_nR=False,
                            seed=1,
                            sources = None):
    """
    MC simulations of SIR, return "num_conf" configuration at time zero and t_limit
     contacts : list([time, node_i, node_j, lambda_ij])
     mu: recovery probability
     t_limit: total numbers of epoch
     n: number of nodes
     random_seed: seed for random library, [default -1, random_seed]

     IMPORTANT: the SIS generation returns a different format for the epidemies
    """
    np.random.seed(seed)
    random.seed(seed)
    propagate.set_seed_numba(seed)
    # choose generator function
    generator_fun = generate_sis_sim if use_SIS else generate_one_sim

    configurations = []
    count = 0
    mean_inf = 0
    max_inf = 0
    mean_recover = 0
    refused_c = 0
    tot_conf_gen = 0
    epidemies = []
    set_ninf = set()
    max_infected = n+10 if max_infected is None else max_infected
    print("Num sources: ", num_source)
    for i in range(num_conf):
        count += 1
        not_accept = True
        ## use counter variable to avoid infinite loops
        tries = 0
        while (not_accept and tries < 100):
            ### keep generating epidemies until we get a satisfactory one
            results = generator_fun(contacts, mu, t_limit, n, num_source, sources=sources)
            tries += 1
            tot_conf_gen += 1
            ninf = results["num_infected"]
            nrec = results["num_recovery"]
            if ninf <= lim_infected:
                if print_:
                    print("\nnum_infected {} < {} (lim_infected)".format(ninf, lim_infected))
                i -= 1
                refused_c +=1
            elif ninf > max_infected:
                if print_:
                    print("\nnum_infected {} > {} (max_infected)".format(ninf, max_infected))
                i -= 1
                refused_c +=1
            elif unique_nI_nR and (ninf,nrec) in set_ninf:
                if print_:
                    print(f"\nnum_infected {ninf},{nrec} already obtained")
                i -= 1
                refused_c +=1
            else:
                configurations.append(results["confs"])
                not_accept = False
                if (not use_SIS) and give_rec_times:
                    epidemies.append((results["epidemy"],results["delays"]))
                else:
                    epidemies.append(results["epidemy"])
                if unique_nI_nR:
                    set_ninf.add((ninf,nrec))
                mean_inf += results["num_infected"]
                mean_recover += results["num_recovery"]
                max_inf = max(max_inf, ninf)

        if print_:
            text_ = "\r # conf {0},".format(count)
            text_ += "mean infected: {0:.1f}, ".format(mean_inf / count)
            text_ += "mean recover: {0:.1f}, ".format(mean_recover / count)
            text_ += "max num infected: {0}".format(max_inf)
            print(text_, end="")
    print(f"\n{refused_c} ({refused_c/tot_conf_gen:.2%}) epidemies refused")
    return configurations, epidemies


def calc_epidemies(epid_test_output,init_fin_confs,t_limit,no_delays=False):
    """
    Construct matrix of states by time
    """
    num_test_conf = len(epid_test_output)
    full_epidemies = []
    t_arrs = np.arange(t_limit+1)
    for inst in range(num_test_conf):
        
        if no_delays:
            inf_times = epid_test_output[inst][0]
            delays = np.full_like(inf_times,np.inf)
        else:
            inf_times = epid_test_output[inst][0][0]
            delays = epid_test_output[inst][1]
        try:
            full_epi = propagate.get_full_epidemy_trace(t_arrs,inf_times+1,delays).T
        except:
            print(t_arrs,inf_times,delays)
            raise InterruptedError
        assert np.all(full_epi[-1] == np.array(init_fin_confs[inst][1]))
        full_epidemies.append(full_epi)
    return full_epidemies

