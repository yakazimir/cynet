import os
import sys
import numpy
import re
import platform
from distutils.core import setup,Command
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.util import get_platform
from optparse import OptionGroup,OptionParser


## configuration stuff

SCONFIG = OptionParser()
DEPS = OptionGroup(SCONFIG,"dependencies")

DEPS.add_option(
    "--dynet",dest="dynet",
    default="",type=str,
    help="Location to dynet"
)

DEPS.add_option(
    "--eigen",dest="eigen",
    default="",type=str,
    help="Location of eigen"
)

DEPS.add_option(
    "--boost",dest="boost",
    default="",type=str,
    help="Location of boost"
)

DEPS.add_option(
    "--inplace=",action="store_true",
)

SCONFIG.add_option_group(DEPS)
CONFIG,_ = SCONFIG.parse_args(sys.argv[1:])


## the main link to all of the dynet c++ stuff

def build_dynet(config):
    """Build the dynet cython extension

    :param config: the build configuration s
    """
    if not config.dynet or not os.path.isdir(config.dynet):
        exit('Cannot find the dynet library! EXITING...')
    if not config.eigen or not os.path.isdir(config.eigen):
        exit('Cannot find the eigen library! EXITING...')
    if not config.boost or not os.path.isdir(config.boost):
        exit('Cannot find the boost library! EXITING...')

    build = os.path.join(config.dynet,"build/dynet")
    
    return [
        Extension('cynet/_dynet',
                    ["cynet/_dynet.pyx"],
                    language="c++",
                    include_dirs = [
                        config.dynet,
                        config.eigen,
                        config.boost,
                    ],
                    libraries=["dynet"],
                    library_dirs=[
                        build
                        ],
                    extra_compile_args=["-std=c++11"],
                    runtime_library_dirs=[build],
                    )
    ]

## cynet sources 
CYNET = [
    Extension("cynet/Seq2Seq",
                  ["cynet/Seq2Seq.pyx"],
                  language="c++",
                  extra_compile_args=["-std=c++11"]
                  ),
]

DYNET = build_dynet(CONFIG)


class build_dynet_ext(build_ext):
    """Customized build_ext to take care of dynet dependencies"""

    user_options = build_ext.user_options + [
        ('dynet=', None, 'Dynet location'),
        ('eigen=', None, 'Eigen location'),
        ('boost=', None, 'Boost location')
        
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.dynet = ""
        self.eigen = ""
        self.boost = ""


if __name__ == "__main__":

    setup(name='cynet',
              cmdclass = {
                  "build_ext": build_dynet_ext,
                  },
              include_dirs=[
                  numpy.get_include(),
                  CONFIG.dynet,
                  CONFIG.eigen,
                  CONFIG.boost,
                ],
              description="Cynet: rewrite of dynet python Cython wrapper for direct use in Cython",
              platforms="any",
              packages=["cynet"],
              scripts=["run_cynet.sh"],
              ext_modules=cythonize(
                  DYNET+CYNET
              )
   )
