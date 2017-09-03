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

cdef class Seq2SeqBase(LoggableClass):
    cdef ParameterCollection model
    cdef LookupParameters embeddings
    ## methods 
    cdef double get_loss(self, x_bold, y_bold)


    
## learner

cdef class Seq2SeqLearner(LoggableClass):
    pass 
