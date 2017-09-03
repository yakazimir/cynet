from optparse import OptionParser,OptionGroup
from cynet.Seq2Seq import params as seq_params

DESCR = """Cynet: pure Cython API for building dynet Seq2Seq models"""
USAGE = """usage: python -m cynet [options] [--help]"""

global_config = OptionParser(usage=USAGE,description=DESCR)

seq_params(global_config)

