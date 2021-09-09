import pickle
import os
import bz2
import json
import gzip
import numpy as np
from pathlib import Path

PICKLE_PROT_VERSION = 4

#TODO: add global names of files with keywords

def save_json(path, obj):
    """
    Save object as json
    """
    with open(path, "wt") as fi:
        json.dump(obj,fi)

def load_json(path):
    """
    Load json file
    """
    with open(path) as f:
        return json.load(f)

def save_json_gzip(path, obj):
    """
    Save object as json, with gzip compression
    """
    with gzip.open(path, "wt") as fi:
        json.dump(obj,fi)

def load_json_gzip(path):
    """
    Load gzip-compressed json file
    """
    with gzip.open(path) as f:
        return json.load(f)

def convert_dtypes_py(d):
    """
    Convert the dtypes of an object to
    pure python.

    Useful for saving.
    """
    if isinstance(d,dict):
        v = {}
        for k in d.keys():
            v[ convert_dtypes_py(k) ]=convert_dtypes_py(d[k])
        return v
    elif isinstance(d,list):
        return [convert_dtypes_py(el) for el in d]
    elif isinstance(d, np.ndarray):
        if len(d.shape) > 1:
            raise NotImplementedError("Can't convert arrays with 2 dimensions or more")
        else:
            return [convert_dtypes_py(el) for el in d]
    elif isinstance(d,np.int) or isinstance(d,np.int_):
        return int(d)
    elif isinstance(d,np.float_):
        return float(d)
    elif isinstance(d,np.bool_):
        return bool(d)
    else:
        return d

def save_pickle_zip(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """

    f = bz2.BZ2File(filename, 'wb')

    pickle.dump(myobj, f, protocol=PICKLE_PROT_VERSION)
    f.close()

def save_pickle_bz2_sec(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """

    f = bz2.open(filename, 'wb')

    pickle.dump(myobj, f, protocol=PICKLE_PROT_VERSION)
    f.close()


def load_pickle_zip(filename):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str
    """

    f = bz2.BZ2File(filename, 'rb')

    myobj = pickle.load(f)
    f.close()
    return myobj

def load_data_instance(instance, path="./data/", with_confs = False, 
              with_obs = False,
              train = False):
    """
    Load data using the "instance" object instead of manually entering
    all parameters.
    See libsaving.EpInstance
    """
    return load_data(type_graph = instance.type_graph,
            n = instance.n,
            d = instance.d,
            lambda_ = instance.lambda_,
            mu = instance.mu,
            t_limit = instance.t_limit,
            p_edge=instance.p_edge,
            seed=instance.seed,
            path=path,
            with_confs = with_confs, 
            with_obs = with_obs,
            train = train)

def load_data(type_graph = "TREE", n = 100, d = 3, lambda_ = 1, mu = 0, 
              t_limit = 5, p_edge = 1,
              seed = 1,
              path = "./data/", 
              with_confs = False, 
              with_obs = False,
              train = False,
              ):
    "load data generate with save_data"
    data_ = {}
    path_data = Path(path)
    dir_path = path_data / type_graph

    name_template_file ="{0}_n_{1}_d_{2}_lambda_{3}_mu_{4}_tlim_{5}_p_edge_{6}_seed_{7}".format(type_graph, 
                                                     n, d, lambda_, 
                                                     mu, t_limit,
                                                     p_edge, seed,)
    name_file_test = name_template_file + "_test.zip"
    name_file_train = name_template_file + "_train.zip"
    
    data_["test"] = load_pickle_zip(dir_path / name_file_test)
    if train:
        data_["train"] = load_pickle_zip(dir_path / name_file_train)   

    name_file_G = name_template_file + "_G.zip"
    data_["G"] = load_pickle_zip(dir_path / name_file_G)   
    name_file_contacts =  name_template_file + "_contacts.zip"
    data_["contacts"] = load_pickle_zip(dir_path / name_file_contacts)
    
    if with_confs and train:
        try:
            name_file_epi_train = name_template_file + "epi_train.zip"
            data_["epidemy_train"] = load_pickle_zip(dir_path / name_file_epi_train)   
        except:
            name_file_epi_train = name_template_file + "_epi_train.zip"
            data_["epidemy_train"] = load_pickle_zip(dir_path / name_file_epi_train)   


    if with_confs:
        try:
            name_file = name_template_file + "epi_test.zip"
            data_["epidemy_test"] = load_pickle_zip(dir_path / name_file)
        except:
            name_file = name_template_file + "_epi_test.zip"
            data_["epidemy_test"] = load_pickle_zip(dir_path / name_file)

            
    if with_obs:
        name_file = name_template_file + "_obs.zip"
        try:
            data_["obs"] = load_pickle_zip(dir_path / name_file)
        except FileNotFoundError:
            print("No observations found")

    try:
        name_file = name_template_file + "_params.zip"
        data_["params"] = load_pickle_zip(dir_path / name_file)
    except:
        print("Couldn't find parameters")


    return data_

def save_data_instance(data_, instance, path="./", params=None):
    """
    Save data using the "instance" object instead of manually entering
    all parameters.
    See libsaving.EpInstance
    """
    save_data(data_, type_graph = instance.type_graph,
            n = instance.n,
            d = instance.d,
            lambda_ = instance.lambda_,
            mu = instance.mu,
            t_limit = instance.t_limit,
            p_edge=instance.p_edge,
            seed=instance.seed,
            path=path,
            params=params)

def save_data(data_, type_graph = "TREE", n = 100, d = 3, lambda_ = 1,  t_limit = 5,
              p_edge = 1,
              mu = 0.1,
              seed=1,
              p_obs_inf = 0.5,
              p_obs_susc = 0.1,
             path = "./data/",params=None):
    "save data"
    if params is None:
        params = {
            "type_graph": type_graph,
            "n":n,
            "d":d,
            "lambda_":lambda_,
            "p_edge":p_edge,
            "path":path,
            "mu":mu,
            "seed":seed,
            "p_obs_inf":p_obs_inf,
            "p_obs_susc":p_obs_susc,
            
        }

    dir_path = path + type_graph
    try:
        os.mkdir(dir_path)
    except:
        print("No dir")

    
    #train_str = str(data_["train"])
    #test_str = str(data_["test"])
    if "train" in data_: 
        train_str = data_["train"]
    test_str = data_["test"]
        
    name_template = dir_path + "/{0}_n_{1}_d_{2}_lambda_{3}_mu_{4}_tlim_{5}_p_edge_{6}_seed_{7}".format(type_graph, 
                                                     n, d, lambda_, 
                                                     mu, t_limit,
                                                     p_edge, seed,)
    name_file_test = name_template + "_test.zip"
    name_file_train = name_template + "_train.zip"
    name_file_train = name_template + "_train.zip"

    '''zf = zipfile.ZipFile(name_file, 
                         mode='w',
                         compression=zipfile.ZIP_DEFLATED, 
                         )
    try:
        zf.writestr('train.txt', train_str)
        zf.writestr('test.txt', test_str)
    finally:
        zf.close()'''
    
    save_pickle_zip(name_file_test, test_str)   
    if "train" in data_: 
        save_pickle_zip(name_file_train, train_str)   
    name_file_G = name_template + "_G.zip"
    save_pickle_zip(name_file_G, data_["G"])   
    name_file_contacts =  name_template + "_contacts.zip"
    save_pickle_bz2_sec(name_file_contacts, data_["contacts"])
    
    if "epidemy_train" in data_:
        name_file = name_template + "_epi_train.zip"
        save_pickle_zip(name_file, data_["epidemy_train"])   

    if "epidemy_test" in data_:
        name_file = name_template + "_epi_test.zip"
        save_pickle_zip(name_file, data_["epidemy_test"]) 
    if "obs" in data_:
        name_file = name_template + "_obs.zip"
        save_pickle_zip(name_file, data_["obs"]) 
        
    name_file = name_template + "_params.zip"
    save_pickle_zip(name_file, params) 
    
    return "save files on " + dir_path