
import traceback
import sys
import os
import logging
from optparse import OptionGroup,OptionParser
from cynet import _dynet as dy
from cynet import global_config
from cynet import start_dynet 
from cynet.Seq2Seq import run_seq2seq


if __name__ == "__main__":

    try:
        ## the main configuration 
        config,_ = global_config.parse_args(sys.argv[1:])

        ## initialize dynet
        start_dynet(config,dy.DynetParams())

        ## now decide what to do
        run_seq2seq(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stderr)
