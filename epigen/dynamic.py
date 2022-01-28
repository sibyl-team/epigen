import numpy as np
import networkx as nx

def gen_contacts_t(graphs, lambda_gen, t_limit, p_edge=1, rng=None, shuffle=True):
    contacts = []
    #random.seed(seed)
    if rng is None:
        rng = np.random
    else:
        try:
            rng.random()
        except AttributeError:
            rng = np.random.Generator(np.random.PCG64(rng))
    for t in range(t_limit):
        assert isinstance(graphs[t], nx.Graph)
        copy_back = True
        if isinstance(graphs[t], nx.DiGraph):
            copy_back = False
        
        nodes = np.arange(len(graphs[t].nodes))
        if shuffle:
            rng.shuffle(nodes)
        for e in graphs[t].edges():
            if rng.random() < p_edge:
                contacts.append((t, nodes[e[0]], nodes[e[1]], lambda_gen(rng)))
                if copy_back:
                    contacts.append((t, nodes[e[1]], nodes[e[0]], lambda_gen(rng)))
    contacts = np.array(contacts, dtype=[("t","f8"), ("i","f8"), ("j","f8"), ("lam", "f8")])
    contacts.sort(axis=0, order=("t","i","j"))
    return contacts.view("f8").reshape(-1,4) #.astype(np.float32)

def _barabasi_albert(n,d,rng, **kwargs):
    return nx.barabasi_albert_graph(n,d,seed=rng,
        initial_graph=nx.generators.cycle_graph(d))

def dynamic_random_graphs(n, d, t_limit,nxgen, seed:int=None, **kwargs):

    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed)

    graphs = []
    for t in range(t_limit):
        G = nxgen(n,d,rng, **kwargs)
        graphs.append(G)

    return graphs


        

    

    