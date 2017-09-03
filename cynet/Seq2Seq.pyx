
from optparse import OptionParser,OptionGroup
import logging
import sys
cimport numpy as np
import numpy as np
cimport _dynet as dy
#import _dynet as dy

## cython class identifiers 
from cynet._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
    get_cg,
)


cdef class LoggableClass:

    @classmethod
    def logger(self):
        """Logger instance associated with object"""
        level = ".".join([__name__,type(self).__name__])
        return logging.getLogger(level)

    @classmethod
    def config_config(cls,config):
        """Create an instance from configuration 

        """
        raise NotImplementedError()
    
## seq2seq models

cdef class Seq2SeqBase(LoggableClass):
    """Base class for Seq2Seq models.

    This is a pure cythonized version of: https://talbaumel.github.io/attention/
    """

    cdef double get_loss(self, x_bold, y_bold):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        raise NotImplementedError

cdef class EncoderDecoder(Seq2SeqBase):
    """Simple encoder-decoder model implementation"""

    def __init__(self,enc_layers,
                     dec_layers,
                     embeddings_size,
                     enc_state_size,
                     dec_state_size):
        """Create a simple encoder decoder instance 

        :param enc_layers: the number of layers used by the encoder RNN 
        :param dec_layers: the number of layers used by the decoder RNN
        :param embedding_size: the size of the embeddings used 
        :param enc_state_size: the size of the encoder RNN state size 
        :parma dec_state_size: the size of decoder RNN state size  
        """
        self.model = ParameterCollection()

    
## learner class

cdef class Seq2SeqLearner(LoggableClass):
    """Class for training Seq2Seq models"""

    def train(self):
        pass

    __call__ = train


def params(config):
    gen_group = OptionGroup(config,"cynet.Seq2Seq","General Seq2Seq settings")

    gen_group.add_option(
        "--loc",dest="loc",default="",
        help="The location of data [default='']"
    )

    config.add_option_group(gen_group)
    


def main():
    """Main execution point for running a seq2seq model 

    """
    e = EncoderDecoder(10,10,10,10,10)
    print e
