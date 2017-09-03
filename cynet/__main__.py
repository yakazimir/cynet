
import traceback
import sys
import os
import logging
from optparse import OptionGroup,OptionParser
from cynet import _dynet as dy
from cynet.Seq2Seq import main as seq2seq_main

DESCR = """Cynet: pure cython API for dynet"""
USAGE = """usage: python -m cynet [options] [--help]""" 

CONFIG = OptionParser(usage=USAGE,description=DESCR)

## general configuration settings
GEN = OptionGroup(CONFIG,"cynet.__main__")

GEN.add_option(
    "--mem",dest="mem",default=512,type=int,
    help="Dynet memory allocation [default=512]"
)

GEN.add_option(
    "--seed",dest="seed",default=2798003128,type=int,
    help="Dynet random seed [default=2798003128]"
)

CONFIG.add_option_group(GEN)


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
        config,_ = CONFIG.parse_args(sys.argv[1:])

        ## initialize dynet
        __setup_dynet(config,dy.DynetParams())

        ## now decide what to do
        seq2seq_main()
        
    except Exception,e:
        traceback.print_exc(file=sys.stdout)
