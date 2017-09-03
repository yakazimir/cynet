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
)


cdef class LoggableClass:
    pass 


## types of seq2seq implementations

cdef class Seq2SeqModel(LoggableClass):
    cdef ParameterCollection model
    cdef LookupParameters embeddings
    #cdef LookupParameters embeddings
    ## methods 
    cdef double get_loss(self, x_bold, y_bold)

cdef class RNNSeq2Seq(Seq2SeqModel):
    cdef LSTMBuilder enc_rnn, dec_rnn
    cdef Parameters output_w,output_b
    

cdef class EncoderDecoder(RNNSeq2Seq):
    pass

cdef class AttentionModel(EncoderDecoder):
    pass 

    
## learner

cdef class Seq2SeqLearner(LoggableClass):
    cdef ComputationGraph cg
    cdef Seq2SeqModel model
    cdef object data
    cdef Trainer trainer

    ## method
    cpdef void train(self,config)
    cdef int _train(self,int epochs,np.ndarray source, np.ndarray target) except -1
