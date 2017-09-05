# cython: profile=True

""" 
Cythonized version of the models implemented in : https://talbaumel.github.io/attention/, which 
itsels seems to be a variation of the attention example in dynet. 

author: Kyle Richardson 

"""

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
    softmax,
    log,
    esum,
    tanh,
    concatenate,
    concatenate_cols,
    transpose,
    colwise_add,
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
        return [cg.lookup(embed,i,True) for i in x if i != -1]

    cdef list _embed_z(self,int[:] z,ComputationGraph cg):
        """Embed the given input for use in the neural model

        :param z: the input vector 
        :param cg: the computation graph 
        """
        cdef LookupParameters embed = self.dec_embeddings
        cdef int i
        return [cg.lookup(embed,i,True) for i in z if i != -1]

    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs):
        """Run the encoder RNN with some initial state and input vector 

        :param init_state: the initial state 
        :param input_vecs: the input vectors
        """
        cdef RNNState s = init_state
        cdef list states,rnn_outputs

        ## cythonize this?
        states = s.add_inputs(input_vecs)
        rnn_outputs = [<Expression>st.output() for st in states]
        return rnn_outputs

    cdef Expression _get_probs(self,Expression rnn_output):
        """Get probabilities associated with RNN output
        
        :param rnn_output: the output of the rnn model 
        """
        cdef Expression output,bias,probs
        cdef Parameters output_w = self.output_w
        cdef Parameters output_b = self.output_b

        output = output_w.expr(True)
        bias = output_b.expr(True)
        probs = softmax(output*rnn_output+bias)
        return probs

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
        self.dec_rnn = LSTMBuilder(dec_layers,enc_state_size,dec_state_size,self.model)

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

    cdef list _encode_string(self,list embeddings):
        """Get the representationf for the input by running through RNN

        :param embeddings: the 
        """
        cdef LSTMBuilder enc_rnn = self.enc_rnn
        cdef RNNState initial_state = enc_rnn.initial_state()
        cdef list hidden_states

        ## annotations or hidden states
        hidden_states = self._run_enc_rnn(initial_state,embeddings)
        return hidden_states

    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int w,zlen = z.shape[0]
        cdef Expression encoded,probs,loss_expr,total_loss
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## encode the input
        x_encoded = self._embed_x(x,cg)
        encoded = self._encode_string(x_encoded)[-1]

        ##
        rnn_state = dec_rnn.initial_state()

        ## loop through the
        for w in range(zlen):
            if z[w] == -1: continue 
            rnn_state = rnn_state.add_input(encoded)
            probs = self._get_probs(rnn_state.output())
            loss_expr = -log(cg.outputPicker(probs,z[w],0))
            loss.append(loss_expr)
            
        total_loss = esum(loss)
        return total_loss
        
cdef class AttentionModel(EncoderDecoder):

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
        EncoderDecoder.__init__(self,enc_layers,dec_layers,
                                    embedding_size,
                                    enc_state_size,dec_state_size,
                                    enc_vocab_size,dec_vocab_size)
        
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size))
        self.attention_w2 = self.model.add_parameters((enc_state_size,dec_state_size))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size

    cdef Expression _attend(self,list input_vectors, RNNState state):
        """Runs the attention network to compute attentions cores

        :param input_vector 
        """
        cdef Parameters w1_o = self.attention_w1
        cdef Parameters w2_o = self.attention_w2
        cdef Parameters v_o = self.attention_v
        cdef Expression w1,w2,v,w2dt,normed
        cdef list weights = [],normalized = []
        cdef int input_vector,vlen = len(input_vectors)

        ## computations 
        cdef Expression attention_weight,new_v
        
        w1 = w1_o.expr(True)
        w2 = w2_o.expr(True)
        v = v_o.expr(True)
        w2dt = w2*((<tuple>state.h())[-1])

        for input_vector in range(vlen):
            attention_weight = v*tanh(w1*input_vectors[input_vector]+w2dt)
            weights.append(attention_weight)

        ## softmax normalization 
        normed = softmax(concatenate(weights))
        for input_vector in range(vlen):
            new_v = input_vectors[input_vector]*normed[input_vector]
            normalized.append(new_v)
        return esum(normalized)

    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int w,zlen = z.shape[0]
        cdef Expression probs,loss_expr,total_loss
        cdef int enc_state_size = self.enc_state_size
        cdef list encoded
        
        ## renew the computation graph directly 
        cg.renew(False,False,None)

        x_encoded = self._embed_x(x,cg)
        encoded = self._encode_string(x_encoded)
        
        rnn_state = dec_rnn.initial_state().add_input(cg.inputVector(enc_state_size))

        for w in range(zlen):
            ## skip over unknown words 
            if z[w] == -1: continue
            attended_encoding = self._attend(encoded,rnn_state)
            rnn_state = rnn_state.add_input(attended_encoding)
            probs = self._get_probs(rnn_state.output())
            loss_expr = -log(cg.outputPicker(probs,z[w],0))
            loss.append(loss_expr)

        total_loss = esum(loss)
        return total_loss

cdef class BiLSTMAttention(AttentionModel):
    """Attention model that uses a bidirectional LSTM model on the source side, 
    more in line with the original Bahdanau paper.

    Follows the attention.py example distributed in dynet/examples/python
    """

    def __init__(self,int enc_layers,
                     int dec_layers,
                     int embedding_size,
                     int enc_state_size,
                     int dec_state_size,
                     int enc_vocab_size,
                     int dec_vocab_size,
                     ):
        """Creates a BiLSTMAttention model instance 

        Note : this constructor doesn't inherit from the previous ones, 
        since it changes a few things, such as the how the decoder RNN 
        work. This should be generalized more...
        
        :param enc_layers: the number of encoder layers 
        :param dec_layers: the number of decoder layers 
        :param enc_state_size: the size of the encoder state 
        :param dec_state_size: the size of the decoder state size 
        :param enc_vocab_size: the size of the encoder vocabulary 
        :param dec_vocab_size: the size of the decoder vocabulary
        """
        self.model = ParameterCollection()

        ## embedding parameters 
        self.enc_embeddings = self.model.add_lookup_parameters((enc_vocab_size,embedding_size))
        self.dec_embeddings = self.model.add_lookup_parameters((dec_vocab_size,embedding_size))

        ## RNN encode and decoder models 
        self.enc_rnn     = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        self.dec_rnn     = LSTMBuilder(dec_layers,enc_state_size*2+embedding_size,dec_state_size,self.model)
        ## the reverse RNN
        self.enc_bwd_rnn = LSTMBuilder(enc_layers,embedding_size,enc_state_size,self.model)
        
        ## output layer and bias for decoder RNN
        self.output_w = self.model.add_parameters((dec_vocab_size,dec_state_size))
        self.output_b = self.model.add_parameters((dec_vocab_size))

        ## attention stuff
        self.attention_w1 = self.model.add_parameters((enc_state_size,enc_state_size*2))
        self.attention_w2 = self.model.add_parameters((enc_state_size,enc_state_size*enc_layers*2))
        self.attention_v = self.model.add_parameters((1,enc_state_size))
        self.enc_state_size = enc_state_size

    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs):
        """Run the encoder RNN with some initial state and input vector 

        :param init_state: the initial state 
        :param input_vecs: the input vectors
        """
        cdef RNNState s = init_state
        cdef list states,rnn_outputs

        # ## cythonize this?
        states = s.add_inputs(input_vecs)
        rnn_outputs = [<Expression>st.output() for st in states]
        return rnn_outputs

    cdef Expression _bi_attend(self,Expression input_matrix,RNNState s,Expression w1dt):
        """Playing around with the 

        :param input_matrix: the input "annotations" in matrix form 
        :param s: the decoder current state 
        :param w1dt: 
        """
        cdef Parameters w2_o = self.attention_w2
        cdef Parameters v_o = self.attention_v
        cdef Expression w2,v,w2dt,unormed,normed,context

        w2 = w2_o.expr(True)
        v = v_o.expr(True)
        w2dt = w2*concatenate(list(s.s()))
        unormed = transpose(v*tanh(colwise_add(w1dt,w2dt)))
        normed = softmax(unormed)
        context = input_matrix*normed
        return context
        

    cdef list _encode_string(self,list embeddings):
        """Get the representationf for the input by running through RNN

        :param embeddings: the 
        """
        cdef LSTMBuilder enc_rnn = self.enc_rnn
        cdef LSTMBuilder enc_bwd_rnn = self.enc_bwd_rnn
        cdef RNNState initial_state = enc_rnn.initial_state()
        cdef RNNState bwd_initial_state = enc_bwd_rnn.initial_state()
        cdef list fw_hidden_states,bwd_hidden_states
        cdef list vectors,sentence_rev = list(reversed(embeddings))
        cdef int i,vsize = len(embeddings)
        cdef Expression con 

        ## run forward
        fw_hidden_states = self._run_enc_rnn(initial_state,embeddings)
        ## run backward and reverse
        bwd_hidden_states = self._run_enc_rnn(bwd_initial_state,sentence_rev)
        bwd_hidden_states = list(reversed(bwd_hidden_states))

        ## create the final vectors
        vectors = []
        for i in range(vsize):
            con = concatenate([fw_hidden_states[i],bwd_hidden_states[i]])
            vectors.append(con)
        return vectors

    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg):
        """Compute loss for a given input and output

        :param x_bold: input representation 
        :param y_bold: the output representation 
        """
        cdef list x_encoded,loss = []
        cdef LSTMBuilder dec_rnn = self.dec_rnn
        cdef RNNState rnn_state
        cdef int i,w,zlen
        cdef Expression probs,loss_expr,total_loss
        cdef int enc_state_size = self.enc_state_size
        cdef list encoded,actual_seq

        ## embedding on the target side 
        cdef Expression last_embed,input_mat

        ## parameterrs
        cdef Parameters w1_o = self.attention_w1
        cdef Parameters output_w = self.output_w
        cdef Parameters output_b = self.output_b
        cdef Expression w1,w1dt,vector,weight,b,out_vector

        ## renew the computation graph directly 
        cg.renew(False,False,None)

        ## parameter expressions
        w1 = w1_o.expr(True)
        weight = output_w.expr(True)
        b = output_b.expr(True)
        
        ## embed input x and run RNNs in both directions 
        x_encoded = self._embed_x(x,cg)
        encoded = self._encode_string(x_encoded)
        ## create matrix of input vectors 
        input_mat = concatenate_cols(encoded)

        ## embed output z
        z_encoded = self._embed_z(z,cg)
        zlen = len(z_encoded)
        last_embed = <Expression>z_encoded[0]

        ## rnn initial state
        rnn_state = dec_rnn.initial_state().add_input(concatenate([cg.inputVector(enc_state_size*2),last_embed]))

        ## rebuild sequence without oov words (to match the embeddings list)
        actual_seq = [i for i in z if i != -1]
        
        ## note : ignores unknown words 
        for w in range(zlen):
            w1dt = w1*input_mat
            vector = dy.concatenate([self._bi_attend(input_mat, rnn_state, w1dt), last_embed])
            rnn_state = rnn_state.add_input(vector)
            out_vector = weight*rnn_state.output()+b
            probs = softmax(out_vector)
            loss_expr = -log(cg.outputPicker(probs,actual_seq[w],0))
            loss.append(loss_expr)
            last_embed = z_encoded[w]

        total_loss = esum(loss)
        return total_loss    

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

    def log_epoch(self,inum,itime,tloss,vtime,vloss):
        """Log information related to the epoch

        :param inum: iteration number
        :param itime: the iteration time 
        :param tloss: the training loss 
        :param vtime: validation time (if available)
        :param vloss: validation lost (if available) 
        :rtype: None
        """
        self.logger.info('Finished iteration %d in %s seconds, ran val in %s seconds, train_loss=%s,val loss=%s' %\
                             (inum,str(itime),str(vtime),str(tloss),str(vloss)))

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
        cdef double loss_value,epoch_loss,val_loss
        cdef double vtime,itime,vstart

        ## neural network model
        cdef Seq2SeqModel model = <Seq2SeqModel>self.model

        ## overall iteration 
        for epoch in range(epochs):
            estart = time.time()
            epoch_loss = 0.0

            ## shuffle dataset?
            
            ## go through each data point 
            for data_point in range(data_size):
                
                ## compute loss and back propogate
                loss = model.get_loss(source[data_point],target[data_point],cg)
                loss_value = loss.value()
                loss.backward()
                epoch_loss += loss_value 
                ## do online update 
                trainer.update()

            ## after epoch information 
            trainer.update_epoch(1.0)
            val_loss = 0.0
            vtime    = 0.0
            itime = time.time()-estart

            ## compute loss on validation 
            if not valid.is_empty:
                vstart = time.time()
                val_loss = compute_val_loss(model,valid,cg)
                vtime = time.time()-vstart

            self.log_epoch(epoch+1,itime,epoch_loss,vtime,val_loss)

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
        
        ## TODO: map these too lookup vectors
        
    property enc_vocab_size:
        """Get information about the encoder vocabulary size"""
        def __get__(self):
            return <int>len(self.enc_map)

    property dec_vocab_size:
        """Get information about the decoder vocabulary size"""
        def __get__(self):
            return <int>len(self.dec_map)

##

cdef double compute_val_loss(Seq2SeqModel model,ParallelDataset data,ComputationGraph cg):
    """Compute loss on a validation dataset given a neural model 

    :param model: the underlying model 
    :param data: the development or held out data 
    :param cg: the computation graph 
    """
    cdef np.ndarray source = data.source
    cdef np.ndarray target = data.target
    cdef int data_point,data_size = data.size
    cdef double total_loss = 0.0

    for data_point in range(data_size):
        loss = model.get_loss(source[data_point],target[data_point],cg)
        total_loss += <double>loss.value()

    return total_loss
        
## factories

MODELS = {
    "simple"    : EncoderDecoder,
    "attention" : AttentionModel,
    "bilstm"    : BiLSTMAttention,
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


def run_seq2seq(config):
    """Main execution point for running a seq2seq model 
    
    :param config: the global configuration 
    """
    
    try: 
        learner = Seq2SeqLearner.from_config(config)
        ## train the model 
        learner.train(config)
        
    except Exception,e:
        traceback.print_exc(file=sys.stderr)
