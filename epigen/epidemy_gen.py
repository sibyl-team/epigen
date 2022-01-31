import networkx as nx
import numpy as np
import pandas as pd
from . import generators, net_gen
from . import dynamic

def cut_contacts_list(contacts_, start_time, end_time, shift_t=True, small_lambda_limit = 0):
    contacts_to_save = [cc for cc in contacts_ if (cc[0] >= start_time and cc[0] < end_time and cc[3] > small_lambda_limit)]
    contacts_to_save = np.array(contacts_to_save)    
    nodes_names = np.unique(contacts_to_save[:,1:3])
    rename_nodes = {}
    count = 0
    for i in nodes_names:
        rename_nodes[i] = count
        count += 1
    contacts_rename = np.array([[x[0], rename_nodes[x[1]], rename_nodes[x[2]], x[3]] for x in contacts_to_save])
    
    t_min = contacts_rename[:,0].min()
    if shift_t:
        contacts_rename[:,0] -= t_min
    contacts_rename.astype(contacts_.dtype)
    return contacts_rename

def get_t_limit(contacts):
    return int(np.max(contacts[:,0]))+1

def epidemy_gen_epinstance(inst, lim_infected=None,
                        max_infected = None,
                        num_conf=10,
                        extra_gen=None,
                        verbose=True,
                        print_out=True,
                        unique_ninf=False,
                        sources=None
                        ):
    """
    Generate epidemies from libsaving.EpInstance object
    """
    if extra_gen is not None:
        data_gen = dict(extra_gen)
    else:
        data_gen = {}

    data_gen["N"] = inst.n
    data_gen["lambda_"] = inst.lambda_
    data_gen["d"] = inst.d
    data_gen["p_edge"] = inst.p_edge

    num_sources = inst.n_src

    data_gen.update({
        "start_time":0,
        "shift_t":True,
    })

    t_limit=inst.t_limit
    data_res = epidemy_gen_new(inst.type_graph,
        t_limit, mu=inst.mu, lim_infected=lim_infected,
        max_infected=max_infected, seed=inst.seed,
        num_conf=num_conf, data_gen=data_gen,
        verbose=verbose,
        print_out=print_out,
        unique_ninf=unique_ninf,
        num_sources=num_sources,
        sources=sources
    )
    try:
        inst.n = len(data_res["G"].nodes)
    except AttributeError:
        print("No graph")
        conts = data_res["contacts"]
        inst.n = int(np.max(conts[:,1:3]))+1
    
    t_lim_c = int(np.max(data_res["contacts"][:,0]))+1
    if t_limit > t_lim_c:
        if print_out: print("Fixing t_limit to {}".format(t_lim_c))
        inst.t_limit = t_lim_c

    assert inst.t_limit == t_lim_c

    return data_res

def load_cut_contacts(data_gen, t_limit):
    """
    Load contacts from file and cut them, if specified
    """
    with np.load(data_gen["path_contacts"], allow_pickle=True) as f:
        contacts = f["contacts"]
    
    cts = cut_contacts_list(contacts.astype(float), 
                             data_gen["start_time"], 
                             t_limit, 
                             shift_t=data_gen["shift_t"], 
                             small_lambda_limit = data_gen["small_lambda_limit"])

    t2 = get_t_limit(cts)
    if t2 != t_limit:
        print(f"GEN: Fixing t_limit to {t2}")
        t_limit = t2
    
    return cts, t_limit


def epidemy_gen_new(type_graph:str = "RRG",
                    t_limit:int=15,
                    mu:float=1e-10,
                    lim_infected:int = None,
                    max_infected:int = None,
                    seed:int=1,
                    num_conf:int=10,
                    num_sources:int=1,
                    data_gen:dict=None,
                    verbose=True,
                    print_out=True,
                    unique_ninf=False,
                    sources=None):
    if data_gen is None:
        data_gen = {
            "N" : 10,
            "d" : 3,
            "p_edge" : 1,
            "h" : 3,
            "scale":1,
        }
    N = data_gen["N"]
    data_extend = {}
    contacts = []
    G = {}
    if type_graph.split("_")[-1] == "dyn":
        ##dynamical graph
        dynamic_graph = True
        if verbose: print("Try with dynamical graph")
        r = type_graph.split("_")[:-1]
        type_graph = "_".join(r)
    else:
        dynamic_graph = False
    
    if type_graph == "RRG":
        
        d = data_gen["d"]
        lambda_ = data_gen["lambda_"]
        p_edge = data_gen["p_edge"]
        if dynamic_graph:
            RRGgen = lambda n, d, rng, **kwargs: nx.random_regular_graph(d,n,seed=rng, **kwargs)

            graphs = dynamic.dynamic_random_graphs(
                N, d, t_limit=t_limit, seed=seed,
                nxgen=RRGgen,
            )
            rng = np.random.RandomState(np.random.PCG64(seed))
            contacts = dynamic.gen_contacts_t(graphs,
            lambda_gen= lambda rng, num : [lambda_]*num,
            t_limit=t_limit, p_edge=p_edge, rng=rng,
            shuffle=False,
            )
        else:
            G = nx.random_regular_graph(d, N, seed=seed)
            print(f"nodes:{N}, edges:{len(G.edges())}")
            contacts = generators.generate_contacts(G, t_limit, lambda_, 
                                                p_edge=p_edge, seed=seed)
    
    elif type_graph == "TREE":
        
        d = data_gen["d"]
        h = data_gen["h"]
        lambda_ = data_gen["lambda_"]
        p_edge = data_gen["p_edge"]
        G = nx.balanced_tree(d-1, h)
        N = G.number_of_nodes()
        print(f"nodes:{N}, edges:{len(G.edges())}")
        if dynamic_graph: raise ValueError("Wrong dynamical graph on trees")
        contacts = generators.generate_contacts(G, t_limit, lambda_, 
                                            p_edge=p_edge, seed=seed)
        
    elif type_graph == "data_deltas" or type_graph == "i_bird":
        contacts, t_limit = load_cut_contacts(data_gen, t_limit=t_limit)
        data_extend["deltas"] = np.copy(contacts)
        gamma=data_gen["gamma"]
        contacts[:, 3] = 1 - np.exp(-gamma * contacts[:,3])
        if dynamic_graph: raise ValueError("Wrong dynamical graph")

    elif type_graph == "data":
        contacts, t_limit = load_cut_contacts(data_gen=data_gen, t_limit=t_limit)
        if dynamic_graph: raise ValueError("Wrong dynamical graph")

    elif type_graph == "data_deltas_2_gamma":
        if dynamic_graph: raise ValueError("Wrong dynamical graph")
        rnd_gen = np.random.RandomState(seed=seed)
        contacts, t_limit = load_cut_contacts(data_gen=data_gen, t_limit=t_limit)
        #gamma=data_gen["gamma"]
        #data_extend["contacts_inference"][:, 3] = 1 - np.exp(-gamma * contacts[:,3])
        N = int(np.max(contacts[:, [1,2]])+1)
        gamma1=data_gen["gamma1"]
        gamma2=data_gen["gamma2"]
        split_nodes=int(N*data_gen["fraction_nodes1"])
        list_N = np.arange(N)
        rnd_gen.shuffle(list_N)
        nodes_1, nodes_2 = list_N[:split_nodes], list_N[split_nodes:]
        data_extend["nodes_1"] = nodes_1
        data_extend["nodes_2"] = nodes_2
        for cc in range(len(contacts)):
            norm_err=1
            if "gauss_mean" and "gauss_sigma" in data_gen:
                norm_err = (1+rnd_gen.normal(data_gen["gauss_mean"], data_gen["gauss_sigma"]))
            if int(contacts[cc][2]) in nodes_1:
                contacts[cc][3] = 1-np.exp(-gamma1 * contacts[cc][3] * norm_err)
            elif int(contacts[cc][2]) in nodes_2:
                contacts[cc][3] = 1-np.exp(-gamma2 * contacts[cc][3] * norm_err)
            else:
                print("ERROR nodes not found in split arrays EXIT")
                return False
        data_extend["contacts_run"] = np.copy(contacts)
        
    elif type_graph == "proximity":
        if dynamic_graph: raise NotImplementedError()
        rng = np.random.RandomState()
        rng.seed(seed)
        N = data_gen["N"]
        scale = data_gen["scale"]
        lambda_ = data_gen["lambda_"]
        p_edge = data_gen["p_edge"]
        x_pos = np.sqrt(N)*rng.random(N)
        y_pos = np.sqrt(N)*rng.random(N)
        # for soft geometric graph generation
        pos  = {i: (x, y) for i, (x, y) in enumerate(zip(x_pos,y_pos))}
        radius = 5*scale
        def p_dist(d):
            return np.exp(-d/scale)
        G=net_gen.soft_random_geometric_graph(
            N, radius=radius, p_dist=p_dist, pos=pos, seed = rng
        )
        print(f"nodes:{N}, edges:{len(G.edges())}")
        contacts = generators.generate_contacts(G, t_limit, lambda_, 
                                            p_edge=p_edge, seed=seed)

    elif type_graph == "BA":
        N = data_gen["N"]
        d= data_gen["d"]
        lambda_ = data_gen["lambda_"]
        p_edge = data_gen["p_edge"]
        if dynamic_graph:
            
            graphs = dynamic.dynamic_random_graphs(
                N, d, t_limit=t_limit, seed=seed,
                nxgen=dynamic._barabasi_albert,
            )
            rng = np.random.RandomState(np.random.PCG64(seed))
            contacts = dynamic.gen_contacts_t(graphs,
            lambda_gen= lambda rng : lambda_,
            t_limit=t_limit, p_edge=p_edge, rng=rng,
            shuffle=True,
            )
        else:
            G = nx.barabasi_albert_graph(n=N, m=d, seed=seed)

            contacts = generators.generate_contacts(G, t_limit, lambda_,
                p_edge=p_edge, seed=seed)

    else:
        print(f"graph {type_graph} not yet implemented")
        print("END")
        return None    

        
    print(f"number of contacts: {len(contacts)}")
    ### rewrite N
    N = int(np.max(contacts[:, [1,2]])+1)
    if lim_infected == None:
        lim_infected = int(N/20)
    elif lim_infected < 1 and lim_infected > 0:
        ## considered fraction of infected
        lim_infected = max(int(N*lim_infected),1)
    # same for max infected
    if max_infected == None:
        max_infected = N+1
    elif max_infected <= 0:
        raise ValueError("Invalid value for max infected (0 or less)")
    elif max_infected < 1:
        ## considered fraction of infected
        max_infected = int(N*max_infected)
    print(f"Lim infected: {lim_infected}, Lim max infected: {max_infected}")
    #print(N)
    test, epidemies = generators.generate_configurations(N, contacts, 
                                mu, t_limit,num_source=num_sources,
                                num_conf = num_conf, seed=seed,
                                lim_infected=lim_infected, max_infected=max_infected,
                                print_=verbose, unique_nI_nR=unique_ninf,  sources=sources)
    print()
    if print_out:
        for tt in range(len(test)):
            ttt = np.array(test[tt][1])
            print(f"S:{len(ttt[ttt==0])}, I:{len(ttt[ttt==1])}, R:{len(ttt[ttt==2])}")
    elif not verbose:
        confs = np.array(test)
        ninf = (confs[:,1]>=1).sum(-1)
        print(f"num infected: min: {ninf.min()} max: {ninf.max()}, avg: {ninf.min()}")

    data_ = {
        "epidemy":epidemies,
        "test" : test,
        "contacts" : contacts,
        "params":{
            "seed" : seed,
            "type_graph" : type_graph,
            "t_limit" : t_limit, # Numbers of epoch of our epidemics spreading [0,1,...,T_limit-1]
            "mu" : mu, # probability of recovery
            "data_gen" : data_gen,
            "num_sources": num_sources,
        },
        "G":G
    }
    
    data_.update(data_extend)
    
    return data_

def epidemy_gen_graph(graph,instance, num_conf,
                      lim_infected=1,
                      verbose=False):
    """
    Generate epidemies on a pre-existing graph, all parameters 
    are given in the `instance` object
    """
    N = len(graph.nodes)
    instance.n = N
    if lim_infected == None:
        lim_infected = int(N/20)
    contacts = []
    t_limit = instance.t_limit
    lamda = instance.lambda_
    seed = instance.seed

    contacts = generators.generate_contacts(graph, t_limit, lamda, 
                                            p_edge=instance.p_edge, seed=seed)
        
    print(f"number of contacts: {len(contacts)}")
    assert N == int(np.max(contacts[:, [1,2]])+1)
    #print(N)
    test, epidemies = generators.generate_configurations(N, contacts, 
                                instance.mu, t_limit, num_conf = num_conf, seed=seed,
                                lim_infected=lim_infected, print_=verbose)
    print()
    for tt in range(len(test)):
        ttt = np.array(test[tt][1])
        print(f"S:{len(ttt[ttt==0])}, I:{len(ttt[ttt==1])}, R:{len(ttt[ttt==2])}")

    data_ = {
        "epidemy":epidemies,
        "test" : test,
        "contacts" : contacts,
        "params": instance.as_dict(),
        "G":graph
    }
    
    return data_

def make_stats_observ(confs, all_obs_df, obs_key="obs_st"):
    #confs = np.array(data_["test"])
    fin_confs = confs[:,1]
    num_I = [len(np.where(c!=0)[0]) for c in fin_confs]

    num_I_obs = [len(d[(d[obs_key]==1) | (d[obs_key]==2)]) for d in all_obs_df]
    
    df_all_obs = pd.DataFrame(dict(num_inf=num_I, num_inf_obs=num_I_obs))
    df_all_obs["num_inf_guess"] = df_all_obs.num_inf - df_all_obs.num_inf_obs
    sources = [np.where(c==1)[0] for c in confs[:,0]]
    df_all_obs["sources"] = sources

    return df_all_obs
