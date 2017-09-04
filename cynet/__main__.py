
import traceback
import sys
import os
import logging
from optparse import OptionGroup,OptionParser
from cynet import _dynet as dy
from cynet import global_config
from cynet import start_dynet 
from cynet.Seq2Seq import run_seq2seq


LEVELS = {
    "info"    : logging.INFO,
    "debug"   : logging.DEBUG,
    "error"   : logging.ERROR,
    "warning" : logging.WARNING,
}


if __name__ == "__main__":

    try:
        ## the main configuration 
        config,_ = global_config.parse_args(sys.argv[1:])

        ## setup the global logger
        log_level = LEVELS.get(config.logger,logging.INFO)
        logging.basicConfig(level=log_level)

        ## initialize dynet
        start_dynet(config,dy.DynetParams())

        ## run the seq2seq main function
        run_seq2seq(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stderr)
