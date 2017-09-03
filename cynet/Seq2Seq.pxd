cimport numpy as np
import numpy as np

from cynet._dynet cimport (
    Expression,
    Trainer,
    ComputationGraph,
    ParameterCollection,
    LookupParameters,
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

cdef class EncoderDecoder(Seq2SeqModel):
    pass

cdef class AttentionModel(EncoderDecoder):
    pass 

    
## learner

cdef class Seq2SeqLearner(LoggableClass):
    cdef ComputationGraph cg
    cdef Seq2SeqModel model
    cdef object data
