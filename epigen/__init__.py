from .base import EpInstance

from .epidemy_gen import epidemy_gen_epinstance, epidemy_gen_graph, epidemy_gen_new

from .generators import calc_epidemies as gen_calc_epi

from .utils import get_git_revision_hash

def calc_epidemies(data, instance):

    return gen_calc_epi(data["epidemy"], data["test"], instance.t_limit)