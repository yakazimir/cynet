#from cynet.cynet cimport *
from _dynet cimport *
cimport _dynet as dynet
from cynet._dynet cimport Expression

def hi():
    e = Expression()
    print e

if __name__ == "__main__":
    e = Expression()
    print e
