import os
from optparse import OptionParser,OptionGroup
from cynet.Seq2Seq import params as seq_params
from cynet.util import params as util_params

DESCR = """Cynet: pure Cython API for building dynet Seq2Seq models"""
USAGE = """usage: python -m cynet [options] [--help]"""

global_config = OptionParser(usage=USAGE,description=DESCR)

seq_params(global_config)
util_params(global_config)

## lib_loc
src_loc = os.path.abspath(os.path.dirname(__file__))
lib_loc = os.path.abspath(os.path.join(src_loc,"../"))

## added the global stuff 
GEN = OptionGroup(global_config,"cynet.__main__")

GEN.add_option(
    "--mem",dest="mem",default=512,type=int,
    help="Dynet memory allocation [default=512]"
)

GEN.add_option(
    "--seed",dest="seed",default=2798003128,type=int,
    help="Dynet random seed [default=2798003128]"
)

GEN.add_option(
    "--logger",dest="logger",default="info",type=str,
    help="The logger level [default='info']"
)

global_config.add_option_group(GEN)


## for initializing dynet

def start_dynet(config,params):
    """General function for setting up dynet memory and random seed 

    :param config: the global configuration with settings 
    :param params: the dynet params instance 
    :type params: cynet._dynet.DynetParams
    """
    params.set_mem(config.mem)
    params.set_random_seed(config.seed)
    params.init()
