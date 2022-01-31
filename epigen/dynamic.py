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
        edges = np.array(graphs[t].edges)
        ## sort edges in both directions
        edges.sort(axis=-1)
        edges.view("i8,i8").sort(axis=0,order=("f0","f1"))
        ## important for shuffling nodes
        if shuffle: edges = nodes[edges]
        #chosen = 
        edges_chosen = edges[rng.random(len(edges)) < p_edge]
        nedges = len(edges_chosen)
        lambdas = np.array(
            lambda_gen(rng, nedges)
        )[:,np.newaxis]
        
        times = np.full((nedges,1), t)
        contacts.append(np.hstack((times, edges_chosen, lambdas)))
        if copy_back:
            contacts.append(np.hstack(
                (times, edges_chosen[:,::-1], lambdas)) )

        #for e in sorted(graphs[t].edges()):
        #    if rng.random() < p_edge:
        #        contacts.append((t, nodes[e[0]], nodes[e[1]], lambda_gen(rng)))
        #        if copy_back:
        #            contacts.append((t, nodes[e[1]], nodes[e[0]], lambda_gen(rng)))
    f=np.float_
    contacts = np.concatenate(contacts) #dtype=[("t","f8"), ("i","f8"), ("j","f8"), ("lam", "f8")])
    contacts.view(dtype=[("t",f), ("i",f), ("j",f), ("lam",f)]).sort(axis=0, order=("t","i","j"))
    return contacts

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


        

    

    