import time
import traceback
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
    LSTMBuilder,
    Parameters,
    SimpleSGDTrainer,
    MomentumSGDTrainer,
    AdagradTrainer,
    AdamTrainer,
    RNNState,
    get_cg, ## to get direct access to computation graph 
)

cdef class LoggableClass:

    @property
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

    cdef Expression get_loss(self, int[:] x, int[:] z, ComputationGraph cg):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        :param computation graph 
        """
        raise NotImplementedError

    cdef list _embed_x(self,int[:] x,ComputationGraph cg):
        """Embed the given input for use in the neural model

        :param x: the input vector 
        :param cg: the computation graph
        """
        cdef LookupParameters embed = self.enc_embeddings
        cdef int i
        return [cg.lookup(embed,i,True) for i in x]

    cdef list _embed_z(self,int[:] z,ComputationGraph cg):
        """Embed the given input for use in the neural model

        :param z: the input vector 
        :param cg: the computation graph 
        """
        cdef LookupParameters embed = self.dec_embeddings
        cdef int i
        return [cg.lookup(embed,i,True) for i in x]

cdef class RNNSeq2Seq(Seq2SeqModel):
    pass 

cdef class EncoderDecoder(RNNSeq2Seq):
    """Simple encoder-decoder model implementation"""

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     ):
        """Create a simple encoder decoder instance 

        :param enc_layers: the number of layers used by the encoder RNN 
        :param dec_layers: the number of layers used by the decoder RNN
        :param embedding_size: the size of the embeddings used 
        :param enc_state_size: the size of the encoder RNN state size 
        :parma dec_state_size: the size of decoder RNN state size  
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn = LSTMBuilder(dec_layers,embedding_size,dec_state_size,self.model)

        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size))
        self.output_b = self.model.add_parameters((dec_vocab_size))
        
    @classmethod
    def from_config(cls,config):
        """Create an encoder decoder instance from configuration 

        :param config: the global configuration 
        :rtype: EncodeDecoder 
        """
        stime = time.time()
        instance = cls(config.enc_rnn_layers,
                           config.dec_rnn_layers,
                           config.embedding_size,
                           config.enc_state_size,
                           config.dec_state_size,
                           config.enc_vocab_size,
                           config.dec_vocab_size,
                           )

        instance.logger.info('Built model in %f seconds' % (time.time()-stime))
        return instance

    ## c methods

    

    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int w,zlen = z.shape[0]
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## encode the input
        x_encoded = self._embed_x(x,cg)

        ##
        rnn_state = dec_rnn.initial_state()

        ## loop through the
        for w in range(zlen):
            pass
        
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

    def __init__(self,trainer,model,train_data,valid_data,stable):
        """Creates a seq2seq learner

        :param model: the underlying neural model 
        :param train_data: the training data 
        :param valid_data: the validation data
        :param stable: the symbol table 
        """
        self.trainer = <Trainer>trainer 
        self.model   = <Seq2SeqModel>model
        self.train_data = train_data ## do we need this here?
        self.valid_data = valid_data 
        self.stable  = stable
        self.cg = get_cg()

    ## training methods
    cpdef void train(self,config):
        """Trian the model using data 

        :param config: the global configuration 
        """
        self.logger.info('Beginning the training loop')
        stime = time.time()

        ## the main training
        try: 
            self._train(config.epochs,self.train_data,self.valid_data)
        except Exception,e:
            self.logger.info(e,exc_info=True)
        finally: 
            self.logger.info('Finished training in %f seconds' % (time.time()-stime))

    cdef int _train(self,int epochs,
                        ParallelDataset train,
                        ParallelDataset valid,
                        ) except -1:
        """C training loop

        :param epochs: the number of epochs or iterations 
        :param source: the source data input 
        :param target: the target data input 
        """
        cdef int data_point,epoch,data_size = train.size
        cdef ComputationGraph cg = self.cg
        cdef Trainer trainer = <Trainer>self.trainer

        ## training data
        cdef np.ndarray source = train.source
        cdef np.ndarray target = train.target
        
        cdef Expression loss
        cdef double loss_value

        ## neural network model
        cdef Seq2SeqModel model = <Seq2SeqModel>self.model

        ## overall iteration 
        for epoch in range(epochs):

            ## shuffle dataset?
            
            ## go through each data point 
            for data_point in range(data_size):

                ## renew the computation graph

                ## compute loss and back propogate
                loss = model.get_loss(source[data_point],target[data_point],cg)
                #loss = network.get_loss(input_string, output_string)
                # loss_value = loss.value()
                # loss.backward()

                ## do online update 
                # trainer.update()
                
            trainer.update_epoch(1.0)

    ## builder

    @classmethod
    def from_config(cls,config):
        """Create a Seq2SeqLearner from configuration 

        :param config: the main or global configuration 
        :param data: a data instance 
        :type data: cynet.util.DataManager
        """
        cdef Seq2SeqModel model
        cdef ParallelDataset train_data,valid_data
        cdef SymbolTable symbol_table

        ## build the data
        train_data,valid_data,symbol_table = build_data(config)
        config.enc_vocab_size = symbol_table.enc_vocab_size
        config.dec_vocab_size = symbol_table.dec_vocab_size

        # ## find the desired class
        nclass = NeuralModel(config.model)
        model = <Seq2SeqModel>nclass.from_config(config)

        # ## find the desired trainer
        trainer = TrainerModel(config,model.model)

        return cls(trainer,model,train_data,valid_data,symbol_table)

## helper classes

cdef class ParallelDataset(LoggableClass):
    """A class for working with parallel datasets"""

    def __init__(self,np.ndarray source,np.ndarray target,bint shuffle=True):
        """Create a ParallelDataset instance 

        :param source: the source language 
        :param target: the target language 
        :raises: ValueError
        """
        self.source = source
        self.target = target
        self._len = self.source.shape[0]
        self._shuffle = shuffle
        
        ## check that both datasets match in size
        assert self._len == self.target.shape[0],"Bad size!"

    property size:
        """Access information about the dataset size"""
        def __get__(self):
            return <int>self._len

    property shuffle:
        """Turn on and off the shuffling settings"""
        def __get__(self):
            return <bint>self._shuffle
        def __set__(self,bint new_val):
            self._shuffle = new_val

    property is_empty:
        """Deteremines if a given dataset is empty or not"""
        def __get__(self):
            return <bint>(self._len == 0)

    @classmethod
    def make_empty(cls):
        """Make an empty dataset

        :returns: ParallelDataset instance without data
        """
        return cls(np.array([]),np.array([]))

cdef class SymbolTable(LoggableClass):
    """Hold information about the integer symbol mappings"""
    
    def __init__(self,enc_map,dec_map):
        self.enc_map = enc_map
        self.dec_map = dec_map
        
        ##
        
    property enc_vocab_size:
        """Get information about the encoder vocabulary size"""
        def __get__(self):
            return <int>len(self.enc_map)

    property dec_vocab_size:
        """Get information about the decoder vocabulary size"""
        def __get__(self):
            return <int>len(self.dec_map)
    
    
        
## factories

MODELS = {
    "simple" : EncoderDecoder,
}

TRAINERS = {
    "sgd"      : SimpleSGDTrainer,
    "momentum" : MomentumSGDTrainer,
    "adagrad"  : AdagradTrainer,
    #"adam"     : AdamTrainer,
}

def NeuralModel(ntype):
    """Factory method for getting a neural model

    :param ntype: the type of neural model desired 
    :raises: ValueError
    """
    nclass = MODELS.get(ntype,None)
    if nclass is None:
        raise ValueError('Unknown neural model')
    return nclass

def TrainerModel(config,model):
    """Factory method for selecting a particular trainer 

    :param ttype: the type of trainer to use 
    """
    tname = config.trainer
    tclass = TRAINERS.get(tname)
    if tclass is None:
        raise ValueError("Unknown trainer model")

    if tname == "adagrad":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,eps=config.epsilon)
    elif tname == "sgd":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay)
    elif tname == "momentum":
        trainer = tclass(model,e0=config.lrate,edecay=config.weight_decay,mom=config.momentum)
    return trainer 

def params(config):
    """Parameters for building seq2seq modela 

    :param config: the global configuration instance 
    :rtype: None 
    """
    gen_group = OptionGroup(config,"cynet.Seq2Seq.{EncoderDecoder,AttentionModel}","General Seq2Seq settings")

    gen_group.add_option(
        "--trainer",dest="trainer",
        default="sgd",
        type=str,
        help="The type of trainer to use [default=sgd]"
    )

    gen_group.add_option(
        "--lrate",dest="lrate",
        default=0.1,
        type=float,
        help="The main learning rate parameter [default=1.0]"
    )

    gen_group.add_option(
        "--weight_decay",dest="weight_decay",
        default=0.0,
        type=float,
        help="The main weight decay parameter [default=0.0]"
    )

    gen_group.add_option(
        "--epsilon",dest="epsilon",
        default=1e-20,
        type=float,
        help="Epsilon paramater to prevent numerical instability [default=1e-20]"
    )

    gen_group.add_option(
        "--momentum",dest="momentum",
        default=0.9,
        type=float,
        help="Momentum value [default=0.9]"
    )

    gen_group.add_option(
        "--epochs",dest="epochs",
        default=10,
        type=int,
        help="The number of training iterations [default=10]"
    )
    
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
        default=64,
        help="The size of the decoder RNN states [default=64]"
    )

    gen_group.add_option(
        "--model",dest="model",
        type=str,
        default="simple",
        help="The type of seq2seq model to use [default=simple]"
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
    logging.basicConfig(level=logging.INFO)
    
    try: 
        learner = Seq2SeqLearner.from_config(config)
        ## train the model 
        learner.train(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stderr)
