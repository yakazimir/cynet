## utility functions
import numpy as np
from random import choice,randrange,seed
from optparse import OptionParser,OptionGroup
from cynet.Seq2Seq import ParallelDataset,SymbolTable

__all__ = [
    "build_data",
    "DataManager",
]

class DataManager(object):
    """General class for storing lexical info, symbol tables, parallel data ..."""
    
    def __init__(self,vocab_size,source,target,symbol_map):
        """Creates a DataManger instance"""        
        self.vocab_size = vocab_size
        self.source = source
        self.target = target
        self.symbol_map = symbol_map
    
def __build_data(config):
    """Build data for running the seq2seq models

    :param config: the global configuration 
    """
    pass

def __sample_model(min_length, max_lenth,characters):
    random_length = randrange(min_length, max_lenth)
    random_char_list = [choice(characters[:-1]) for _ in range(random_length)]
    random_string = ''.join(random_char_list) 
    return random_string, random_string[::-1]

def __build_demo_data(config):
    """Build demo data for running the seq2seq modela

    :param config: the global configuration 
    """
    seed(a=42)
    characters = list("abcd")
    characters.append("<EOS>")

    int2char = list(characters) 
    char2int = {c:i for i,c in enumerate(characters)}

    train_set = [__sample_model(1,config.max_string,characters) for _ in range(3000)]
    val_set   = [__sample_model(1,config.max_string,characters) for _ in range(50)]

    ## training data 
    source,target = zip(*train_set)
    source =  np.array([np.array([char2int[c] for c in list(i)+["<EOS>"]],dtype=np.int32) for i in source],dtype=object)
    target =  np.array([np.array([char2int[c] for c in list(i)+["<EOS>"]],dtype=np.int32) for i in target],dtype=object)
    train_data = ParallelDataset(source,target)

    ## valid data
    sourcev,targetv = zip(*val_set)
    sourcev = np.array([np.array([char2int[c] for c in list(i)+["<EOS>"]],dtype=np.int32) for i in sourcev],dtype=object)
    targetv =  np.array([np.array([char2int[c] for c in list(i)+["<EOS>"]],dtype=np.int32) for i in targetv],dtype=object)
    valid_data = ParallelDataset(sourcev,targetv)

    ## symbol table
    table = SymbolTable(char2int,char2int)

    return (train_data,valid_data,table)

def __build_wdir(config):
    """Build data from an existing working directory full of data

    :param config: the main configuration 
    """
    pass
    
def build_data(config):
    """Main method for building seq2seq data"""

    if config.wdir:
        pass 
    
    elif config.demo_data:
        return __build_demo_data(config)

    ## build another data
    else: 
        raise ValueError('Unknown option: please specify --wdir or --demo_data')

    
## module parameters

def params(config):
    """Utility parameters"""
    util_group = OptionGroup(config,"cynet.util","General settings for building data")

    util_group.add_option(
        "--demo_data",dest="demo_data",
        default=True,
        action="store_true",
        help="Run the code with the demo data [default=True]"
    )

    util_group.add_option(
        "--max_string",dest="max_string",
        type=int,
        default=15,
        help="Run the code with the demo data [default=True]"
    )

    util_group.add_option(
        "--wdir",dest="wdir",
        type=str,
        default="",
        help="The working directory [default='']"
    )


    util_group.add_option(
        "--source",dest="source",
        type=str,
        default="e",
        help="The source suffix [default='e']"
    )

    util_group.add_option(
        "--target",dest="target",
        type=str,
        default="f",
        help="The target suffix [default='f']"
    )

    util_group.add_option(
        "--name",dest="name",
        type=str,
        default="data",
        help="The name of the data [default='data']"
    )

    config.add_option_group(util_group)
    
