
import traceback
import sys
import os
import logging
from optparse import OptionGroup,OptionParser
from cynet import _dynet as dy
from cynet import global_config
from cynet.Seq2Seq import run_seq2seq

## general configuration settings
GEN = OptionGroup(global_config,"cynet.__main__")

GEN.add_option(
    "--mem",dest="mem",default=512,type=int,
    help="Dynet memory allocation [default=512]"
)

GEN.add_option(
    "--seed",dest="seed",default=2798003128,type=int,
    help="Dynet random seed [default=2798003128]"
)

global_config.add_option_group(GEN)


def __setup_dynet(config,params):
    """Sets up and initializes dynet from configuration
    
    :param config: the global configuration 
    :param params: the dynet parameter object 
    :rtype: None 
    """
    params.set_mem(config.mem)
    params.set_random_seed(config.seed)
    params.init()

if __name__ == "__main__":

    try:
        ## the main configuration 
        config,_ = global_config.parse_args(sys.argv[1:])

        ## initialize dynet
        __setup_dynet(config,dy.DynetParams())

        ## now decide what to do
        run_seq2seq(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
