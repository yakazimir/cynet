import time 
from optparse import OptionParser,OptionGroup
import logging
import sys
cimport numpy as np
import numpy as np
cimport _dynet as dy
from cynet.util import *

## cython class identifiers 
from cynet._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
    get_cg, ## to get direct access to computation graph 
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

cdef class Seq2SeqModel(LoggableClass):
    """Base class for Seq2Seq models.

    This is a pure cythonized version of: https://talbaumel.github.io/attention/
    """

    cdef double get_loss(self, x_bold, y_bold):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        raise NotImplementedError

cdef class EncoderDecoder(Seq2SeqModel):
    """Simple encoder-decoder model implementation"""

    def __init__(self,enc_layers,
                     dec_layers,
                     embedding_size,
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

        ## embedding parameters 
        self.embeddings = self.model.add_lookup_parameters((10,embedding_size))
        
    @classmethod
    def from_config(cls,config):
        """Create an encoder decoder instance from configuration 

        :param config: the global configuration 
        :rtype: EncodeDecoder 
        """
        instance = cls(config.enc_rnn_layers,
                           config.dec_rnn_layers,
                           config.embedding_size,
                           config.enc_state_size,
                           config.dec_state_size)

        return instance 
        
cdef class AttentionModel(EncoderDecoder):
    
    @classmethod
    def from_config(cls,config):
        """Create an encoder decoder instance from configuration 

        :param config: the global configuration 
        :rtype: EncodeDecoder 
        """
        pass

## learner class

cdef class Seq2SeqLearner(LoggableClass):
    """Class for training Seq2Seq models"""


    def __init__(self,model):
        """Creates a seq2seq learner

        :param model: the underlying neural model 
        :param computation_graph: the global computation graph 
        """
        self.model = <Seq2SeqModel>model 
        self.cg = get_cg()

    @classmethod
    def from_config(cls,config):
        """Create a Seq2SeqLearner from configuration 

        :param config: the main or global configuration 
        """
        pass


def params(config):
    """Parameters for building seq2seq modela 

    :param config: the global configuration instance 
    :rtype: None 
    """
    gen_group = OptionGroup(config,"cynet.Seq2Seq.{EncoderDecoder,AttentionModel}","General Seq2Seq settings")

    gen_group.add_option(
        "--enc_rnn_layers",dest="enc_rnn_layers",
        default=1,
        type=int,
        help="The number of RNN layers to use in encoder [default=1]"
    )

    gen_group.add_option(
        "--dec_rnn_layers",dest="dec_rnn_layers",
        default=1,
        type=int,
        help="The number of RNN layers to use in decoder [default=1]"
    )
    
    gen_group.add_option(
        "--embedding_size",dest="embedding_size",
        default=4,
        type=int,
        help="The size of the embeddings [default=4]"
    )

    gen_group.add_option(
        "--enc_state_size",dest="enc_state_size",
        default=64,
        type=int,
        help="The size of the encoder RNN  states [default=64]"
    )

    gen_group.add_option(
        "--dec_state_size",dest="dec_state_size",
        type=int,
        help="The size of the decoder RNN states [default=64]"
    )

    config.add_option_group(gen_group)

    ##
    learn_group = OptionGroup(config,"cynet.Seq2Seq.Seq2SeqLearner","Settings for Seq2Seq Learner")

    learn_group.add_option(
        "--data",dest="data",
        default=1,
        type=int,
        help="The location of the data [default='']"
    )

    config.add_option_group(learn_group)

def run_seq2seq(config):
    """Main execution point for running a seq2seq model 
    
    :param config: the global configuration 
    """
    e = EncoderDecoder.from_config(config) 
    print e

    
