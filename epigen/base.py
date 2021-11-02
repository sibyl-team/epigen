def make_instance_dict(type_graph, n, d, t_lim, lamda, mu, seed, p_edge):
    return {
        "n": n,
        "d": d,
        "t_limit": t_lim,
        "lambda": lamda,
        "mu": mu,
        "seed": seed,
        "p_edge": p_edge,
        "type_graph": type_graph
    }


class EpInstance:
    """
    Parameters for the generated epidemies
    type_graph: type of the graph (tree, RRG, Barabasi-Albert,...)
    n: number of nodes
    d: degree of node(where applicable)
    t_limit: Numbers of epoch of our epidemics spreading [0,1,...,T_limit-1]
    lamda: probability of infection (SI/SIR model)
    mu: probability of recovery (SIR model)
    seed: seed used to generate the epidemies
    p_edge: probability of usage of each edge, used in the generation of epidemies
    """
    def __init__(self, type_graph, n, d, t_limit, lamda, mu, seed, p_edge, n_source=1):
        self.type_graph = type_graph
        self.n = n
        self.d = d
        self.t_limit = t_limit
        self.lambda_ = lamda
        self.mu = mu
        self.seed = seed
        self.p_edge = p_edge
        self.n_src = n_source
        self.old_print = False if n_source > 1 else True

    def as_dict(self):
        return {
            "n": self.n,
            "d": self.d,
            "t_limit": self.t_limit,
            "lambda": self.lambda_,
            "mu": self.mu,
            "seed": self.seed,
            "p_edge": self.p_edge,
            "type_graph": self.type_graph,
            "n_source": self.n_src
        }
    def legacy(self, yes):
        self.old_print = yes
    
    @staticmethod
    def from_dict(input_dict):
        return EpInstance(
            type_graph=input_dict["type_graph"],
            n=input_dict["n"],
            d=input_dict["d"],
            t_limit=input_dict["t_limit"],
            lamda=input_dict["lambda"],
            mu=input_dict["mu"],
            seed=input_dict["seed"],
            p_edge=input_dict["p_edge"],
            n_source=input_dict["n_source"]
        )
    def __hash__(self):
        return hash((self.type_graph,
                     self.n,
                     self.t_limit,
                     self.d,
                     self.lambda_,
                     self.mu,
                     self.seed,
                     self.p_edge,
                     self.n_src
                    ))
    def __repr__(self):
        reprstring = f"""EpInstance:
    type: {self.type_graph}, d: {self.d}, N: {self.n}, seed: {self.seed},
    lambda: {self.lambda_}, mu: {self.mu}, p_edge: {self.p_edge}, num src: {self.n_src}"""
        return reprstring

    def __eq__(self, o):
        return hash(o) == hash(self)


    def __str__(self):
        basestr= "{}_n_{}_d_{}_tlim_{}_lam_{}_mu_{}_s_{}_pe_{}".format(
            self.type_graph, self.n, self.d, self.t_limit, self.lambda_,
            0 if self.mu == 0 else self.mu, self.seed, self.p_edge
        )
        if self.old_print:
            return basestr
        else:
            return basestr+f"_nsrc_{self.n_src}"
