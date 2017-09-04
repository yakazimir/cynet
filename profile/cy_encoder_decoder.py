import sys
import os
import cProfile
import pstats 
from cynet.Seq2Seq import run_seq2seq
from cynet import start_dynet,global_config
from cynet import _dynet as dy
from cynet import lib_loc

# python -m profile.cyn_encoder_decoder

config,_ = global_config.parse_args(sys.argv[1:])
config.epochs = 1

def run_decoder():
    run_seq2seq(config)
    
if __name__ == "__main__":

    ## start dynet 
    start_dynet(config,dy.DynetParams())

    run_decoder()
    profiler = os.path.join(lib_loc,"decoder_profile")
    cProfile.runctx("run_decoder()",globals(),locals(),profiler)
    s = pstats.Stats(profiler)
    s.strip_dirs().sort_stats("time").print_stats()
