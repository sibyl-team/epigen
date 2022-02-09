"""
Copyright 2022 Fabio Mazza

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
    ## get all the indices
    nodes = set()
    for g in graphs:
        nodes.update(g.nodes)
    nodes = np.array(tuple(nodes))

    for t in range(t_limit):
        assert isinstance(graphs[t], nx.Graph)
        copy_back = True
        if isinstance(graphs[t], nx.DiGraph):
            copy_back = False
        
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
        ## generate lambdas
        lambdas = np.array(extract_lambdas(
            lambda_gen, nedges)
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

def dynamic_random_graphs(n, d, t_limit,nxgen, seed:int=None, p_drop_node=0., **kwargs):

    if seed is None:
        rng = np.random
    else:
        try:
            seed.random()
            rng = seed
        except AttributeError:
            rng = np.random.RandomState(seed)
    if p_drop_node >= 1:
        raise ValueError("Drop of nodes must be in range [0,1)")
    graphs = []
    for t in range(t_limit):
        G = nxgen(n,d,rng, **kwargs)
        N = len(G.nodes)
        if p_drop_node > 0:
            idx_remove = np.where(rng.rand(N)<p_drop_node)[0]
            G.remove_nodes_from(idx_remove)
        graphs.append(G)

    return graphs


def extract_lambdas(callable_rand, n):

    lambdas = np.array(callable_rand(n))
    assert len(lambdas) == n
    trials = 0
    MAX_TRIALS = 20000
    while((lambdas>=1).sum()>0):
        idc = (lambdas>=1)
        #print(idc.sum())
        lambdas[idc] = np.array(callable_rand(idc.sum()))
        trials+=1
        if trials > MAX_TRIALS:
            print("ERROR: MAX TRIALS FOR lambda generation reached")
            break

    return lambdas
    

    