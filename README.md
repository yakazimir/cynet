Cynet : Playing around with Dynet Python/Cython wrapper 
==================

Sandbox for experimenting with the Python/Cython part of Dynet (link below), and building
pure Cython Dynet applications, in particular related to building
Seq2Seq neural network translation models.

Citations for Dynet: https://github.com/clab/dynet, see paper below

```
@article{dynet,
      title={DyNet: The Dynamic Neural Network Toolkit},
      author={Graham Neubig and Chris Dyer and Yoav Goldberg and Austin Matthews and Waleed Ammar and Antonios Anastasopoulos and Miguel Ballesteros and David Chiang and Daniel Clothiaux and Trevor Cohn and Kevin Duh and Manaal Faruqui and Cynthia Gan and Dan Garrette and Yangfeng Ji and Lingpeng Kong and Adhiguna Kuncoro and Gaurav Kumar and Chaitanya Malaviya and Paul Michel and Yusuke Oda and Matthew Richardson and Naomi Saphra and Swabha Swayamdipta and Pengcheng Yin},
      journal={arXiv preprint arXiv:1701.03980},
      year={2017}
    }
```

One goal is to profile the Cython wrapper in its' current form, and
see how to speed up some components by writing in pure Cython. Let's
see if anything interesting comes out... 

Quick Start 
-----------------

Can be built in place by typing the following:

    python setup.py build_ext --inplace --dynet=/path/to/dynet --eigen=/path/to/eigen --boost=/path/to/boost. 

Then run the following:

    ./run_cynet [option] [--help]

