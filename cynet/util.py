## utility functions
import os
import codecs
import logging 
import numpy as np
from random import choice,randrange,seed
from optparse import OptionParser,OptionGroup
from cynet.Seq2Seq import ParallelDataset,SymbolTable

__all__ = [
    "build_data",
]

util_logger = logging.getLogger('cynet.util')

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

def __build_demo_data2(config):
    """Data for running the second demo"""
    eos = "<EOS>"
    characters = list("abcdefghijklmnopqrstuvwxyz ")
    characters.append(eos)
    char2int = {c:i for i,c in enumerate(characters)}

    train_set = [("it is working","it is working")]

    ## training data
    source,target = zip(*train_set)
    source = np.array([np.array([char2int[w] for w in ["<EOS>"]+list(i)+["<EOS>"]],dtype=np.int32) for i in source])
    target = np.array([np.array([char2int[w] for w in ["<EOS>"]+list(i)+["<EOS>"]],dtype=np.int32) for i in target])

    
    train_data = ParallelDataset(source,target)
    table = SymbolTable(char2int,char2int)
    valid = ParallelDataset.make_empty()
    return (train_data,valid,table)

def __read_data(path,symbols=None,lowercase=True):
    """Read the data and extract a symbol map

    :param path: the path to the data 
    """
    total   = []
    encoded = []
    symbol_map = {} if symbols is None else symbols
    symbol_map[u"<EOS>"] = 0
    
    with codecs.open(path,encoding='utf-8') as my_data:
        for line in my_data:
            line = line.strip().lower()
            total.append(line+" "+"<EOS>")
            if symbols is None: 
                for word in line.split():
                    word = word.strip()
                    if word not in symbol_map:
                        symbol_map[word] = len(symbol_map)
    ## now encode the data
    for example in total:
        encoded.append(
            np.array([symbol_map.get(w.strip(),-1) for w in example.split()],
                         dtype=np.int32))
    ## map to numpy
    return (np.array(encoded,dtype=object),symbol_map)
            
def __build_wdir(config):
    """Build data from an existing working directory full of data

    :param config: the main configuration 
    """
    wdir = config.wdir
    name = config.name

    ## train data
    ################
    ################
    source_train = os.path.join(wdir,"%s.%s" % (name,config.source))
    target_train = os.path.join(wdir,"%s.%s" % (name,config.target))

    ## check that the data exists
    if not os.path.isfile(source_train): raise ValueError('Cannot find the source data: %s' % source_train)
    if not os.path.isfile(target_train): raise ValueError('Cannot find the target data: %s' % source_train)

    ## build the data
    source_train_e,enc_symbols = __read_data(source_train)
    target_train_e,dec_symbols = __read_data(target_train)
    train = ParallelDataset(source_train_e,target_train_e)

    symbol_map = SymbolTable(enc_symbols,dec_symbols)

    ## valid data
    ################
    ################
    try:
        source_valid = os.path.join(wdir,"%s_val.%s" % (name,config.source))
        target_valid = os.path.join(wdir,"%s_val.%s" % (name,config.target))
        ## build these datasets
        source_valid_e,_ = __read_data(source_valid,symbols=enc_symbols)
        target_valid_e,_ = __read_data(source_valid,symbols=dec_symbols)
        valid = ParallelDataset(source_valid_e,target_valid_e)

    except Exception:
        util_logger.warning('Error building valid data, or missing, skipping...')
        valid = ParallelDataset.make_empty()

    # util_logger.info('Loaded dataset, source=%d pairs,target=%d pairs, source vocab=%d tokens, target_vocab=%d tokens' % (train.size,valid.size,symbol_map.enc_vocab_size,symbol_map.dec_vocab_size))
    return (train,valid,symbol_map)
        
def build_data(config):
    """Main method for building seq2seq data"""

    if config.wdir:
        return __build_wdir(config)

    elif config.demo_data2:
        return __build_demo_data2(config)
    
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
        help="The maximum string size (for demo) [default=15]"
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

    util_group.add_option(
        "--demo_data2",dest="demo_data2",
        default=False,
        action="store_true",
        help="Run the second demo [default='data']"
    )

    config.add_option_group(util_group)
    
