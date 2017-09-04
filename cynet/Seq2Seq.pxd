cimport numpy as np
import numpy as np

from cynet._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
    LSTMBuilder,
    Parameters,
    RNNState,
)


cdef class LoggableClass:
    pass 

## types of seq2seq implementations

cdef class Seq2SeqModel(LoggableClass):
    cdef ParameterCollection model
    cdef LookupParameters enc_embeddings
    cdef LookupParameters dec_embeddings
    ## methods 
    cdef Expression get_loss(self, int[:] x, int[:] z,ComputationGraph cg)
    cdef list _embed_x(self,int[:] x,ComputationGraph cg)
    cdef list _embed_z(self,int[:]z,ComputationGraph cg)
    cdef list _run_enc_rnn(self,RNNState init_state,list input_vecs)

cdef class RNNSeq2Seq(Seq2SeqModel):
    cdef LSTMBuilder enc_rnn, dec_rnn
    cdef Parameters output_w,output_b
    
cdef class EncoderDecoder(RNNSeq2Seq):
    cdef list _encode_string(self,list embeddings)

cdef class AttentionModel(EncoderDecoder):
    pass 

## learner

cdef class Seq2SeqLearner(LoggableClass):
    cdef ComputationGraph cg
    cdef Seq2SeqModel model
    cdef SymbolTable stable 
    cdef ParallelDataset train_data
    cdef ParallelDataset valid_data 
    cdef Trainer trainer

    ## method
    cpdef void train(self,config)
    cdef int _train(self,int epochs,ParallelDataset train, ParallelDataset valid) except -1

## helper classes

cdef class ParallelDataset(LoggableClass):
    cdef np.ndarray source,target
    cdef int _len
    cdef bint _shuffle

cdef class SymbolTable(LoggableClass):
    cdef dict enc_map,dec_map
    cdef int _vocab_size
